"""Compare three communication placements for the two-dot region.

Run on the intended physical GPUs with:

    CUDA_VISIBLE_DEVICES=4,5,6 uv run python compare_sharding_strategy.py
"""

from __future__ import annotations

import argparse
import csv
import functools
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

COLLECTIVE_PATTERN = re.compile(
    r"\s(all-reduce|all-gather|reduce-scatter|collective-permute|all-to-all)\(",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Strategy:
    name: str
    description: str
    hidden_sharding: NamedSharding
    second_weight_sharding: NamedSharding
    output_sharding: NamedSharding


@dataclass
class Benchmark:
    strategy: Strategy
    compiled: jax.stages.Compiled
    arguments: tuple[jax.Array, jax.Array, jax.Array]
    collectives: tuple[str, ...]
    samples_ms: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=3072)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=Path("sharding_results"))
    return parser.parse_args()


def two_dot_region(
    x: jax.Array,
    first_weight: jax.Array,
    second_weight: jax.Array,
    *,
    hidden_sharding: NamedSharding,
) -> jax.Array:
    partial_hidden = x @ first_weight
    hidden = jax.lax.with_sharding_constraint(
        partial_hidden,
        hidden_sharding,
    )
    return jax.nn.relu(hidden) @ second_weight


def collective_sequence(hlo_text: str) -> tuple[str, ...]:
    collectives = []
    for line in hlo_text.splitlines():
        if "=" not in line:
            continue
        match = COLLECTIVE_PATTERN.search(line.split("=", maxsplit=1)[1])
        if match:
            collectives.append(match.group(1).lower())
    return tuple(collectives)


def memory_stats(compiled: jax.stages.Compiled) -> dict[str, int]:
    analysis = compiled.memory_analysis()
    if analysis is None:
        return {}
    names = (
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "temp_size_in_bytes",
        "alias_size_in_bytes",
    )
    return {name: int(getattr(analysis, name)) for name in names}


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def main() -> None:
    args = parse_args()
    if args.size <= 0 or args.size % 3:
        raise ValueError("--size must be positive and divisible by 3")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("--warmup must be non-negative and --iterations positive")

    devices = np.asarray(jax.devices())
    if len(devices) != 3 or jax.default_backend() != "gpu":
        raise RuntimeError(f"Expected exactly 3 GPUs, found {devices.tolist()}")

    mesh = Mesh(devices, ("gpu",))
    row = NamedSharding(mesh, P("gpu", None))
    column = NamedSharding(mesh, P(None, "gpu"))
    replicated = NamedSharding(mesh, P())
    strategies = (
        Strategy(
            "a_final_all_reduce",
            "column-sharded hidden; replicated output",
            column,
            row,
            replicated,
        ),
        Strategy(
            "b_final_reduce_scatter",
            "column-sharded hidden; column-sharded output",
            column,
            row,
            column,
        ),
        Strategy(
            "c_replicate_before_dot2",
            "replicated hidden; column-sharded output",
            replicated,
            column,
            column,
        ),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    x = jax.device_put(jnp.ones((args.size, args.size), jnp.float32), column)
    first_weight = jax.device_put(jnp.eye(args.size, dtype=jnp.float32), row)
    second_weights = {
        row: jax.device_put(jnp.eye(args.size, dtype=jnp.float32), row),
        column: jax.device_put(jnp.eye(args.size, dtype=jnp.float32), column),
    }

    benchmarks: list[Benchmark] = []
    for strategy in strategies:
        function = functools.partial(
            two_dot_region,
            hidden_sharding=strategy.hidden_sharding,
        )
        region = jax.jit(
            function,
            in_shardings=(column, row, strategy.second_weight_sharding),
            out_shardings=strategy.output_sharding,
        )
        arguments = (x, first_weight, second_weights[strategy.second_weight_sharding])
        lowered = region.lower(*arguments)
        compiled = lowered.compile()
        stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
        hlo = compiled.as_text()
        if hlo is None:
            raise RuntimeError(f"No compiled HLO returned for {strategy.name}")
        (args.output_dir / f"{strategy.name}.stablehlo.mlir").write_text(stablehlo)
        (args.output_dir / f"{strategy.name}.optimized.hlo").write_text(hlo)
        benchmarks.append(
            Benchmark(strategy, compiled, arguments, collective_sequence(hlo), [])
        )

    # Warm every executable before timing, then interleave strategies to reduce
    # systematic drift from temperature or background load.
    for benchmark in benchmarks:
        for _ in range(args.warmup):
            benchmark.compiled(*benchmark.arguments).block_until_ready()
    for _ in range(args.iterations):
        for benchmark in benchmarks:
            start = time.perf_counter_ns()
            result = benchmark.compiled(*benchmark.arguments)
            result.block_until_ready()
            benchmark.samples_ms.append((time.perf_counter_ns() - start) / 1e6)

    rows: list[dict[str, object]] = []
    for benchmark in benchmarks:
        result = benchmark.compiled(*benchmark.arguments)
        result.block_until_ready()
        np.testing.assert_allclose(np.asarray(result[0, :16]), 1.0)
        stats = memory_stats(benchmark.compiled)
        row_data: dict[str, object] = {
            "strategy": benchmark.strategy.name,
            "description": benchmark.strategy.description,
            "collectives": " -> ".join(benchmark.collectives) or "none",
            "median_ms": statistics.median(benchmark.samples_ms),
            "mean_ms": statistics.fmean(benchmark.samples_ms),
            "p10_ms": percentile(benchmark.samples_ms, 0.10),
            "p90_ms": percentile(benchmark.samples_ms, 0.90),
            **stats,
        }
        rows.append(row_data)

    csv_path = args.output_dir / "summary.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"JAX {jax.__version__}; devices={devices.tolist()}")
    print(f"shape={args.size}x{args.size}; samples={args.iterations}")
    for row_data in rows:
        temporary_mib = int(row_data.get("temp_size_in_bytes", 0)) / 2**20
        print(
            f"{row_data['strategy']}: median={row_data['median_ms']:.3f} ms, "
            f"p10={row_data['p10_ms']:.3f}, p90={row_data['p90_ms']:.3f}, "
            f"temp={temporary_mib:.1f} MiB, collectives={row_data['collectives']}"
        )
    print(f"Results and HLO written to: {args.output_dir}")


if __name__ == "__main__":
    main()
