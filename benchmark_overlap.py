"""Measure whether an independent matmul and all-reduce overlap in XLA."""

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
    parser.add_argument("--matmul-shape", default="2048x2048x2048")
    parser.add_argument("--sizes-kib", default="1024,16384,65536")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--batch-launches", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("overlap_results"))
    return parser.parse_args()


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[round((len(ordered) - 1) * fraction)]


def time_executable(
    executable: object,
    arguments: tuple[jax.Array, ...],
    warmup: int,
    iterations: int,
    launches: int,
) -> tuple[list[float], object]:
    for _ in range(warmup):
        jax.block_until_ready(executable(*arguments))
    samples = []
    result = None
    for _ in range(iterations):
        start = time.perf_counter_ns()
        for _ in range(launches):
            result = executable(*arguments)
        jax.block_until_ready(result)
        samples.append((time.perf_counter_ns() - start) / 1e3 / launches)
    return samples, result


def main() -> None:
    args = parse_args()
    try:
        m, k, n = (int(value) for value in args.matmul_shape.lower().split("x"))
        sizes = [int(value) * 1024 for value in args.sizes_kib.split(",")]
    except ValueError as error:
        raise ValueError("invalid --matmul-shape or --sizes-kib") from error
    if (
        min(m, k, n, *sizes) <= 0
        or args.warmup < 0
        or args.iterations <= 0
        or args.batch_launches <= 0
    ):
        raise ValueError("shapes/sizes/iterations must be positive")

    devices = jax.devices()
    if jax.default_backend() != "gpu" or len(devices) < 2:
        raise RuntimeError(f"Expected at least two GPUs, found {devices}")
    rank_count = len(devices)
    dtype = jnp.float32 if args.dtype == "float32" else jnp.bfloat16
    mesh = Mesh(np.asarray(devices), ("rank",))
    matrix_sharding = NamedSharding(mesh, P("rank", None, None))
    vector_sharding = NamedSharding(mesh, P("rank", None))
    matrices = jax.device_put(
        jnp.ones((rank_count, m, k), dtype=dtype), matrix_sharding
    )
    weights = jax.device_put(jnp.ones((rank_count, k, n), dtype=dtype), matrix_sharding)

    def compute(x: jax.Array, weight: jax.Array) -> jax.Array:
        return x @ weight

    def communication(x: jax.Array) -> jax.Array:
        return jax.lax.psum(x, "rank")

    def combined(
        x: jax.Array, weight: jax.Array, payload: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        return x @ weight, jax.lax.psum(payload, "rank")

    compute_function = jax.pmap(compute, axis_name="rank")
    communication_function = jax.pmap(communication, axis_name="rank")
    combined_function = jax.pmap(combined, axis_name="rank")
    compute_executable = compute_function.lower(matrices, weights).compile()
    scalar_sharding = NamedSharding(mesh, P("rank"))
    scalar = jax.device_put(jnp.ones((rank_count,), jnp.float32), scalar_sharding)
    baseline_executable = jax.pmap(lambda x: x + 1).lower(scalar).compile()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "environment.json").write_text(
        json.dumps(
            {
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "devices": [str(device) for device in devices],
                "device_kinds": [device.device_kind for device in devices],
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "xla_flags": os.environ.get("XLA_FLAGS"),
                "matmul_shape_per_rank": [m, k, n],
                "dtype": args.dtype,
                "warmup": args.warmup,
                "iterations": args.iterations,
                "batch_launches": args.batch_launches,
            },
            indent=2,
        )
        + "\n"
    )

    compute_samples, compute_result = time_executable(
        compute_executable,
        (matrices, weights),
        args.warmup,
        args.iterations,
        args.batch_launches,
    )
    baseline_samples, _ = time_executable(
        baseline_executable,
        (scalar,),
        args.warmup,
        args.iterations,
        args.batch_launches,
    )
    assert compute_result is not None
    np.testing.assert_allclose(
        np.asarray(compute_result[0, 0, : min(8, n)], dtype=np.float32),
        float(k),
        rtol=2e-2,
    )
    compute_median = statistics.median(compute_samples)
    baseline_median = statistics.median(baseline_samples)

    rows = []
    sample_rows = []
    for requested_bytes in sizes:
        elements = (requested_bytes + 3) // 4
        payload = jax.device_put(
            jnp.stack(
                [
                    jnp.full((elements,), rank + 1, jnp.float32)
                    for rank in range(rank_count)
                ]
            ),
            vector_sharding,
        )
        comm_executable = communication_function.lower(payload).compile()
        combined_executable = combined_function.lower(
            matrices, weights, payload
        ).compile()
        comm_samples, comm_result = time_executable(
            comm_executable,
            (payload,),
            args.warmup,
            args.iterations,
            args.batch_launches,
        )
        combined_samples, combined_result = time_executable(
            combined_executable,
            (matrices, weights, payload),
            args.warmup,
            args.iterations,
            args.batch_launches,
        )
        assert comm_result is not None and combined_result is not None
        expected = rank_count * (rank_count + 1) / 2
        np.testing.assert_allclose(np.asarray(comm_result[:, 0]), expected)
        np.testing.assert_allclose(np.asarray(combined_result[1][:, 0]), expected)

        comm_median = statistics.median(comm_samples)
        combined_median = statistics.median(combined_samples)
        raw_savings = compute_median + comm_median - combined_median
        corrected_compute = compute_median - baseline_median
        corrected_comm = comm_median - baseline_median
        corrected_combined = combined_median - baseline_median
        denominator = min(corrected_compute, corrected_comm)
        corrected_overlap = (
            (corrected_compute + corrected_comm - corrected_combined) / denominator
            if denominator > 0
            else float("nan")
        )
        memory = combined_executable.memory_analysis()
        row = {
            "rank_count": rank_count,
            "m": m,
            "k": k,
            "n": n,
            "dtype": args.dtype,
            "bytes_per_rank": elements * 4,
            "baseline_median_us": baseline_median,
            "compute_median_us": compute_median,
            "communication_median_us": comm_median,
            "combined_median_us": combined_median,
            "raw_combined_savings_us": raw_savings,
            "corrected_compute_us": corrected_compute,
            "corrected_communication_us": corrected_comm,
            "corrected_combined_us": corrected_combined,
            "corrected_overlap_efficiency": corrected_overlap,
            "combined_p10_us": percentile(combined_samples, 0.10),
            "combined_p90_us": percentile(combined_samples, 0.90),
            "argument_size_in_bytes": int(memory.argument_size_in_bytes),
            "output_size_in_bytes": int(memory.output_size_in_bytes),
            "temp_size_in_bytes": int(memory.temp_size_in_bytes),
        }
        rows.append(row)
        for sample_index, (compute_us, comm_us, combined_us) in enumerate(
            zip(compute_samples, comm_samples, combined_samples, strict=True)
        ):
            sample_rows.append(
                {
                    "bytes_per_rank": elements * 4,
                    "sample_index": sample_index,
                    "compute_us": compute_us,
                    "communication_us": comm_us,
                    "combined_us": combined_us,
                }
            )
        hlo = combined_executable.as_text()
        if hlo is not None:
            (args.output_dir / f"combined_{elements * 4}.optimized.hlo").write_text(hlo)
        print(
            f"payload={elements * 4 / 2**20:.1f} MiB compute={compute_median:.1f} us "
            f"comm={comm_median:.1f} us combined={combined_median:.1f} us "
            f"baseline={baseline_median:.1f} us corrected_overlap={corrected_overlap:.3f}"
        )

    for path, data in (
        (args.output_dir / "overlap.csv", rows),
        (args.output_dir / "overlap_samples.csv", sample_rows),
    ):
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    print(f"Results written to: {args.output_dir}")


if __name__ == "__main__":
    main()
