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

Run the sharded matrix multiplication on physical GPU IDs 4, 5, and 6:

```bash
CUDA_VISIBLE_DEVICES=4,5,6 uv run python main.py
```

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
