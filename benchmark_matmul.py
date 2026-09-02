"""Measure isolated GPU matmul costs for Phase C calibration."""

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
import jax.numpy as jnp
import jaxlib
import numpy as np

DEFAULT_SHAPES = "1024x1024x1024,2048x2048x2048,1024x3072x3072,4096x3072x3072,4096x3072x12288,4096x12288x3072"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", default=DEFAULT_SHAPES)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--batch-launches", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("matmul_results"))
    return parser.parse_args()


def parse_shapes(text: str) -> list[tuple[int, int, int]]:
    shapes = []
    try:
        for value in text.split(","):
            shape = tuple(int(part) for part in value.lower().split("x"))
            if len(shape) != 3 or any(dimension <= 0 for dimension in shape):
                raise ValueError
            shapes.append(shape)
    except ValueError as error:
        raise ValueError("--shapes must look like MxKxN,MxKxN") from error
    if not shapes:
        raise ValueError("at least one shape is required")
    return shapes


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[round((len(ordered) - 1) * fraction)]


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.iterations <= 0 or args.batch_launches <= 0:
        raise ValueError("--warmup must be non-negative and --iterations positive")
    shapes = parse_shapes(args.shapes)
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"Expected a GPU backend, found {jax.default_backend()}")

    device = jax.devices()[0]
    dtype = jnp.float32 if args.dtype == "float32" else jnp.bfloat16
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "device": str(device),
        "device_kind": device.device_kind,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "xla_flags": os.environ.get("XLA_FLAGS"),
        "dtype": args.dtype,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "batch_launches": args.batch_launches,
        "shapes": shapes,
    }
    (args.output_dir / "environment.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    summaries: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []
    for m, k, n in shapes:
        left = jax.device_put(jnp.ones((m, k), dtype=dtype), device)
        right = jax.device_put(jnp.ones((k, n), dtype=dtype), device)
        executable = jax.jit(lambda x, y: x @ y).lower(left, right).compile()
        for _ in range(args.warmup):
            executable(left, right).block_until_ready()

        mode_samples = {}
        result = None
        for timing_mode, launches in (
            ("synchronous", 1),
            ("amortized", args.batch_launches),
        ):
            samples_us = []
            for sample_index in range(args.iterations):
                start = time.perf_counter_ns()
                for _ in range(launches):
                    result = executable(left, right)
                result.block_until_ready()
                latency_us = (time.perf_counter_ns() - start) / 1e3 / launches
                samples_us.append(latency_us)
                sample_rows.append(
                    {
                        "m": m,
                        "k": k,
                        "n": n,
                        "dtype": args.dtype,
                        "timing_mode": timing_mode,
                        "sample_index": sample_index,
                        "launches": launches,
                        "latency_us_per_launch": latency_us,
                    }
                )
            mode_samples[timing_mode] = samples_us
        assert result is not None
        np.testing.assert_allclose(
            np.asarray(result[0, : min(8, n)], dtype=np.float32),
            float(k),
            rtol=2e-2,
        )
        flops = 2 * m * k * n
        analysis = executable.memory_analysis()
        for timing_mode, samples_us in mode_samples.items():
            median_us = statistics.median(samples_us)
            summaries.append(
                {
                    "m": m,
                    "k": k,
                    "n": n,
                    "dtype": args.dtype,
                    "timing_mode": timing_mode,
                    "flops": flops,
                    "median_us": median_us,
                    "mean_us": statistics.fmean(samples_us),
                    "p10_us": percentile(samples_us, 0.10),
                    "p90_us": percentile(samples_us, 0.90),
                    "tflops": flops / median_us / 1e6,
                    "argument_size_in_bytes": int(analysis.argument_size_in_bytes),
                    "output_size_in_bytes": int(analysis.output_size_in_bytes),
                    "temp_size_in_bytes": int(analysis.temp_size_in_bytes),
                }
            )
            print(
                f"{m}x{k}x{n} {args.dtype} {timing_mode}: "
                f"median={median_us:.2f} us, {flops / median_us / 1e6:.2f} TFLOP/s"
            )

    for path, rows in (
        (args.output_dir / "matmul.csv", summaries),
        (args.output_dir / "matmul_samples.csv", sample_rows),
    ):
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Results written to: {args.output_dir}")


if __name__ == "__main__":
    main()
