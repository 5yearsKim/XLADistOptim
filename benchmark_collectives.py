"""Measure basic JAX collective costs for Phase C calibration.

Examples:

    CUDA_VISIBLE_DEVICES=4,5 uv run python benchmark_collectives.py
    CUDA_VISIBLE_DEVICES=4,5,6 uv run python benchmark_collectives.py

The reported payload is the input buffer size on each rank.  It is deliberately
kept separate from any claim about physical PCIe traffic: XLA/NCCL may use
different algorithms and transfer volumes for the same logical collective.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import time
from pathlib import Path

import jax
import jaxlib
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

DEFAULT_SIZES_KIB = "4,16,64,256,1024,4096,16384,65536"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes-kib",
        default=DEFAULT_SIZES_KIB,
        help="comma-separated input payload sizes per rank",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument(
        "--batch-launches",
        type=int,
        default=20,
        help="launches per amortized sample before one synchronization",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("collective_results"))
    return parser.parse_args()


def parse_sizes(text: str) -> list[int]:
    try:
        sizes = [int(value.strip()) * 1024 for value in text.split(",")]
    except ValueError as error:
        raise ValueError("--sizes-kib must contain comma-separated integers") from error
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("all payload sizes must be positive")
    return sizes


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[round((len(ordered) - 1) * fraction)]


def collective_functions() -> dict[str, object]:
    def all_reduce(x: jax.Array) -> jax.Array:
        return jax.lax.psum(x, "rank")

    def reduce_scatter(x: jax.Array) -> jax.Array:
        return jax.lax.psum_scatter(x, "rank", scatter_dimension=0, tiled=True)

    def all_gather(x: jax.Array) -> jax.Array:
        return jax.lax.all_gather(x, "rank", axis=0, tiled=True)

    return {
        "all_reduce": jax.pmap(all_reduce, axis_name="rank"),
        "reduce_scatter": jax.pmap(reduce_scatter, axis_name="rank"),
        "all_gather": jax.pmap(all_gather, axis_name="rank"),
    }


def validate_result(name: str, result: jax.Array, rank_count: int) -> None:
    # Each input rank is filled with rank + 1.  Validate only boundary elements
    # so correctness checking does not copy a large benchmark result to the host.
    boundary = np.asarray(result[:, [0, result.shape[1] - 1]])
    rank_sum = rank_count * (rank_count + 1) / 2
    if name in {"all_reduce", "reduce_scatter"}:
        np.testing.assert_allclose(boundary, rank_sum)
    else:
        np.testing.assert_allclose(boundary[:, 0], 1.0)
        np.testing.assert_allclose(boundary[:, 1], float(rank_count))


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.iterations <= 0 or args.batch_launches <= 0:
        raise ValueError(
            "--warmup must be non-negative; --iterations and --batch-launches positive"
        )

    requested_sizes = parse_sizes(args.sizes_kib)
    devices = jax.devices()
    if jax.default_backend() != "gpu" or len(devices) < 2:
        raise RuntimeError(f"Expected at least two visible GPUs, found {devices}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in devices],
        "device_kinds": [device.device_kind for device in devices],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "xla_flags": os.environ.get("XLA_FLAGS"),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "batch_launches": args.batch_launches,
        "requested_sizes_bytes_per_rank": requested_sizes,
    }
    (args.output_dir / "environment.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    rank_count = len(devices)
    functions = collective_functions()
    rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []

    for requested_bytes in requested_sizes:
        # psum_scatter requires equal division by the rank count.  Padding is
        # recorded explicitly, which matters for three-rank power-of-two tests.
        requested_elements = (requested_bytes + 3) // 4
        elements = (requested_elements + rank_count - 1) // rank_count * rank_count
        actual_bytes = elements * 4
        host_shards = [
            np.full((elements,), rank + 1, dtype=np.float32)
            for rank in range(rank_count)
        ]
        mesh = Mesh(np.asarray(devices), ("rank",))
        input_sharding = NamedSharding(mesh, P("rank", None))
        inputs = jax.device_put(np.stack(host_shards), input_sharding)

        compiled = {
            name: function.lower(inputs).compile()
            for name, function in functions.items()
        }
        for name, executable in compiled.items():
            for _ in range(args.warmup):
                executable(inputs).block_until_ready()

            for timing_mode, launches in (
                ("synchronous", 1),
                ("amortized", args.batch_launches),
            ):
                samples_us: list[float] = []
                result = None
                for sample_index in range(args.iterations):
                    start = time.perf_counter_ns()
                    for _ in range(launches):
                        result = executable(inputs)
                    assert result is not None
                    result.block_until_ready()
                    elapsed_us = (time.perf_counter_ns() - start) / 1e3 / launches
                    samples_us.append(elapsed_us)
                    sample_rows.append(
                        {
                            "collective": name,
                            "rank_count": rank_count,
                            "actual_bytes_per_rank": actual_bytes,
                            "timing_mode": timing_mode,
                            "sample_index": sample_index,
                            "launches": launches,
                            "latency_us_per_launch": elapsed_us,
                        }
                    )
                validate_result(name, result, rank_count)

                median_us = statistics.median(samples_us)
                rows.append(
                    {
                        "collective": name,
                        "rank_count": rank_count,
                        "requested_bytes_per_rank": requested_bytes,
                        "actual_bytes_per_rank": actual_bytes,
                        "dtype": "float32",
                        "timing_mode": timing_mode,
                        "median_us": median_us,
                        "mean_us": statistics.fmean(samples_us),
                        "p10_us": percentile(samples_us, 0.10),
                        "p90_us": percentile(samples_us, 0.90),
                        "payload_gbps": actual_bytes / median_us / 1e3,
                    }
                )
                print(
                    f"{name:14s} {timing_mode:11s} ranks={rank_count} "
                    f"payload={actual_bytes / 2**20:8.3f} MiB "
                    f"median={median_us:9.2f} us"
                )

    csv_path = args.output_dir / "collectives.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    samples_path = args.output_dir / "collective_samples.csv"
    with samples_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(sample_rows[0]))
        writer.writeheader()
        writer.writerows(sample_rows)
    print(f"Results written to: {args.output_dir}")


if __name__ == "__main__":
    main()
