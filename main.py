import argparse
import functools
import re
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    profile_group = parser.add_mutually_exclusive_group()
    profile_group.add_argument(
        "--profile",
        action="store_true",
        help="capture an XProf device timeline",
    )
    profile_group.add_argument(
        "--profile-perfetto",
        action="store_true",
        help="capture a Perfetto-compatible device timeline",
    )
    parser.add_argument("--profile-dir", default="profiles")
    parser.add_argument(
        "--dump-hlo",
        nargs="?",
        const="hlo_dumps",
        metavar="DIRECTORY",
        help="write pre-SPMD StableHLO and compiled HLO (default: hlo_dumps)",
    )
    return parser.parse_args()


COLLECTIVE_PATTERN = re.compile(
    r"\b(all-reduce|all-gather|reduce-scatter|collective-permute|all-to-all)\b",
    re.IGNORECASE,
)


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
    hidden = jax.nn.relu(hidden)
    return hidden @ second_weight


def dump_and_verify_hlo(
    lowered: jax.stages.Lowered,
    compiled: jax.stages.Compiled,
    output_directory: str,
) -> tuple[Path, Path, tuple[str, ...]]:
    """Write compiler IR and return collective operations found after SPMD."""
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    stablehlo_path = directory / "two_dot_region.stablehlo.mlir"
    hlo_path = directory / "two_dot_region.optimized.hlo"

    stablehlo_path.write_text(str(lowered.compiler_ir(dialect="stablehlo")))
    hlo_text = compiled.as_text()
    if hlo_text is None:
        raise RuntimeError("The backend did not return compiled HLO text")
    hlo_path.write_text(hlo_text)

    collectives = tuple(
        sorted({match.group(1).lower() for match in COLLECTIVE_PATTERN.finditer(hlo_text)})
    )
    if not collectives:
        raise RuntimeError(
            f"No collective operation found in compiled HLO: {hlo_path}"
        )
    return stablehlo_path, hlo_path, collectives


def main() -> None:
    args = parse_args()
    devices = np.array(jax.devices())
    if len(devices) != 3:
        raise RuntimeError(f"Expected 3 GPUs, found {len(devices)}: {devices}")

    mesh = Mesh(devices, ("gpu",))
    row_sharding = NamedSharding(mesh, P("gpu", None))
    column_sharding = NamedSharding(mesh, P(None, "gpu"))
    replicated = NamedSharding(mesh, P())

    size = 3072
    # Shard Dot_1's contracting dimension: every device computes a partial
    # MxN result.  Requesting a column-sharded hidden value allows SPMD to
    # combine and distribute those partials with a reduce-scatter.
    x = jax.device_put(
        jnp.ones((size, size), dtype=jnp.float32), column_sharding
    )
    first_weight = jax.device_put(jnp.eye(size, dtype=jnp.float32), row_sharding)
    second_weight = jax.device_put(
        jnp.eye(size, dtype=jnp.float32), row_sharding
    )

    region = jax.jit(
        functools.partial(two_dot_region, hidden_sharding=column_sharding),
        in_shardings=(column_sharding, row_sharding, row_sharding),
        out_shardings=replicated,
    )
    lowered = region.lower(x, first_weight, second_weight)
    compiled = lowered.compile()

    if args.dump_hlo:
        stablehlo_path, hlo_path, collectives = dump_and_verify_hlo(
            lowered, compiled, args.dump_hlo
        )
        print(f"StableHLO written to: {stablehlo_path}")
        print(f"Compiled HLO written to: {hlo_path}")
        print(f"Verified collective operations: {', '.join(collectives)}")

    start = time.perf_counter()
    result = compiled(x, first_weight, second_weight)
    result.block_until_ready()
    first_run = time.perf_counter() - start

    start = time.perf_counter()
    result = compiled(x, first_weight, second_weight)
    result.block_until_ready()
    steady_state = time.perf_counter() - start

    if args.profile or args.profile_perfetto:
        with jax.profiler.trace(
            args.profile_dir,
            create_perfetto_trace=args.profile_perfetto,
        ):
            for step in range(5):
                with jax.profiler.StepTraceAnnotation("two_dot_region", step=step):
                    result = compiled(x, first_weight, second_weight)
                    result.block_until_ready()

    np.testing.assert_allclose(np.asarray(result[0, :16]), 1.0)

    print(f"JAX {jax.__version__}; backend={jax.default_backend()}")
    print(f"Devices: {devices.tolist()}")
    print(f"Result sharding: {result.sharding}")
    print(f"Addressable shards: {len(result.addressable_shards)}")
    print(f"First run (precompiled): {first_run * 1e3:.2f} ms")
    print(f"Steady-state run: {steady_state * 1e3:.2f} ms")
    if args.profile or args.profile_perfetto:
        profile_format = "Perfetto" if args.profile_perfetto else "XProf"
        print(f"{profile_format} profile written under: {args.profile_dir}")


if __name__ == "__main__":
    main()
