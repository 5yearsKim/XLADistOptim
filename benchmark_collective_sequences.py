"""Measure back-to-back collective sequences relevant to the two-dot region."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes-kib", default="1024,16384,65536")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--batch-launches", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("sequence_results"))
    return parser.parse_args()


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[round((len(ordered) - 1) * fraction)]


def main() -> None:
    args = parse_args()
    sizes = [int(value) * 1024 for value in args.sizes_kib.split(",")]
    devices = jax.devices()
    if jax.default_backend() != "gpu" or len(devices) < 2:
        raise RuntimeError(f"Expected at least two GPUs, found {devices}")
    ranks = len(devices)
    mesh = Mesh(np.asarray(devices), ("rank",))
    sharding = NamedSharding(mesh, P("rank", None))

    def rs_ar(x: jax.Array) -> jax.Array:
        reduced = jax.lax.psum_scatter(x, "rank", scatter_dimension=0, tiled=True)
        return jax.lax.psum(reduced, "rank")

    def rs_rs(x: jax.Array) -> jax.Array:
        reduced = jax.lax.psum_scatter(x, "rank", scatter_dimension=0, tiled=True)
        return jax.lax.psum_scatter(reduced, "rank", scatter_dimension=0, tiled=True)

    functions = {
        "reduce_scatter_all_reduce": jax.pmap(rs_ar, axis_name="rank"),
        "reduce_scatter_reduce_scatter": jax.pmap(rs_rs, axis_name="rank"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "environment.json").write_text(
        json.dumps(
            {
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "devices": [str(device) for device in devices],
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "warmup": args.warmup,
                "iterations": args.iterations,
                "batch_launches": args.batch_launches,
            },
            indent=2,
        )
        + "\n"
    )
    rows = []
    samples_rows = []
    for requested_bytes in sizes:
        elements = (requested_bytes + 3) // 4
        divisor = ranks**2
        elements = (elements + divisor - 1) // divisor * divisor
        inputs = jax.device_put(
            jnp.stack(
                [jnp.full((elements,), rank + 1, jnp.float32) for rank in range(ranks)]
            ),
            sharding,
        )
        for name, function in functions.items():
            executable = function.lower(inputs).compile()
            for _ in range(args.warmup):
                executable(inputs).block_until_ready()
            samples = []
            result = None
            for sample_index in range(args.iterations):
                start = time.perf_counter_ns()
                for _ in range(args.batch_launches):
                    result = executable(inputs)
                result.block_until_ready()
                latency = (time.perf_counter_ns() - start) / 1e3 / args.batch_launches
                samples.append(latency)
                samples_rows.append(
                    {
                        "sequence": name,
                        "bytes_per_rank": elements * 4,
                        "sample_index": sample_index,
                        "latency_us": latency,
                    }
                )
            expected = ranks**2 * (ranks + 1) / 2
            np.testing.assert_allclose(
                np.asarray(result[:, [0, result.shape[1] - 1]]), expected
            )
            memory = executable.memory_analysis()
            median = statistics.median(samples)
            rows.append(
                {
                    "sequence": name,
                    "rank_count": ranks,
                    "bytes_per_rank": elements * 4,
                    "median_us": median,
                    "p10_us": percentile(samples, 0.1),
                    "p90_us": percentile(samples, 0.9),
                    "temp_size_in_bytes": int(memory.temp_size_in_bytes),
                }
            )
            hlo = executable.as_text()
            if hlo is not None:
                (args.output_dir / f"{name}_{elements * 4}.optimized.hlo").write_text(
                    hlo
                )
            print(
                f"{name} payload={elements * 4 / 2**20:.1f} MiB median={median:.1f} us"
            )
    for path, data in (
        (args.output_dir / "sequences.csv", rows),
        (args.output_dir / "sequence_samples.csv", samples_rows),
    ):
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)


if __name__ == "__main__":
    main()
