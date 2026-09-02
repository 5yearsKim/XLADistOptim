# XLA Distributed Optimization

## Analytical pipeline simulator

The hardware-independent simulator models a serial producer, communication
resource, and consumer with a bounded number of buffers:

```python
from simulator import simulate_pipeline

result = simulate_pipeline(
    tile_count=3,
    producer_us=2,
    communication_us=3,
    consumer_us=4,
    tile_bytes=100,
    buffer_count=2,
    memory_budget_bytes=200,
)
print(result.makespan_us, result.peak_memory_bytes)
```

Run its tests with `uv run pytest`.

## GPU smoke test

Run the two-dot region on physical GPU IDs 4, 5, and 6:

```bash
CUDA_VISIBLE_DEVICES=4,5,6 uv run python main.py
```

The input columns and first-weight rows shard the first dot's contracting
dimension, so each GPU computes a partial hidden value. An intermediate
column-sharding constraint requests that XLA reduce and scatter those partials
before the activation. The second weight is row-sharded, matching the hidden
value's contracting-dimension sharding.

Dump the pre-SPMD StableHLO and post-SPMD compiled HLO, then verify that the
compiled program contains a collective:

```bash
CUDA_VISIBLE_DEVICES=4,5,6 uv run python main.py --dump-hlo
```

The files are written under `hlo_dumps/` by default. Pass a directory after
`--dump-hlo` to change the destination.

## Phase C calibration

The Phase C benchmark programs are `benchmark_collectives.py`,
`benchmark_collective_sequences.py`, `benchmark_matmul.py`, and
`benchmark_overlap.py`. Run `calibrate_cost_model.py` after collecting data to
write the lookup model and held-out validation report. The current calibrated
results and their applicability limits are documented in `phase_c_results.md`.

## XProf timeline

Capture a warmed-up five-step profile under `profiles/`:

```bash
CUDA_VISIBLE_DEVICES=4,5,6 uv run python main.py --profile
```

Start the XProf viewer:

```bash
uv run xprof --port 8791 profiles
```

Open <http://localhost:8791>, choose the latest run, and select **Trace Viewer**
from the **Tools** menu. If XProf runs on a remote machine, forward the port:

```bash
ssh -L 8791:127.0.0.1:8791 USER@HOST
```

## Perfetto timeline

Capture the same warmed-up five-step profile in Perfetto's trace format:

```bash
CUDA_VISIBLE_DEVICES=4,5,6 uv run python main.py --profile-perfetto
```

This writes `perfetto_trace.json.gz` inside the latest timestamped directory
under `profiles/plugins/profile/`. Open <https://ui.perfetto.dev>, click
**Open trace file**, and select that file. The trace includes the
`two_dot_region` step annotations as well as CPU and GPU activity.

Use `--profile-dir DIRECTORY` with either profiling command to choose a
different output directory. `--profile` and `--profile-perfetto` are mutually
exclusive because JAX can collect only one trace at a time.
