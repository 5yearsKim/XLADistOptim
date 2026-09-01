import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        action="store_true",
        help="capture an XProf device timeline",
    )
    parser.add_argument("--profile-dir", default="profiles")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    devices = np.array(jax.devices())
    if len(devices) != 3:
        raise RuntimeError(f"Expected 3 GPUs, found {len(devices)}: {devices}")

    mesh = Mesh(devices, ("gpu",))
    row_sharding = NamedSharding(mesh, P("gpu", None))
    replicated = NamedSharding(mesh, P())

    size = 3072
    lhs = jax.device_put(jnp.ones((size, size), dtype=jnp.float32), row_sharding)
    rhs = jax.device_put(jnp.eye(size, dtype=jnp.float32), replicated)

    matmul = jax.jit(lambda x, y: x @ y, out_shardings=row_sharding)

    start = time.perf_counter()
    result = matmul(lhs, rhs)
    result.block_until_ready()
    compile_and_run = time.perf_counter() - start

    start = time.perf_counter()
    result = matmul(lhs, rhs)
    result.block_until_ready()
    steady_state = time.perf_counter() - start

    if args.profile:
        with jax.profiler.trace(args.profile_dir):
            for step in range(5):
                with jax.profiler.StepTraceAnnotation("sharded_matmul", step=step):
                    result = matmul(lhs, rhs)
                    result.block_until_ready()

    np.testing.assert_allclose(np.asarray(result[0, :16]), 1.0)

    print(f"JAX {jax.__version__}; backend={jax.default_backend()}")
    print(f"Devices: {devices.tolist()}")
    print(f"Result sharding: {result.sharding}")
    print(f"Addressable shards: {len(result.addressable_shards)}")
    print(f"Compile + first run: {compile_and_run * 1e3:.2f} ms")
    print(f"Steady-state run: {steady_state * 1e3:.2f} ms")
    if args.profile:
        print(f"Profile written under: {args.profile_dir}")


if __name__ == "__main__":
    main()
