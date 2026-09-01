# Initial Research Plan: Communication-Aware Distributed Lowering for XLA

## 1. Research objective

Develop a communication-aware distributed lowering approach that jointly selects:

\[
(\text{sharding},\ \text{tile size},\ \text{collective realization},\
\text{pipeline schedule},\ \text{buffering})
\]

for a multi-operation compute/communication region, subject to device-memory and
correctness constraints.

The initial target region is:

```text
Dot_1 -> ReduceScatter -> elementwise activation -> Dot_2
```

After understanding this region, the work can expand to:

```text
Dot_1 -> ReduceScatter -> RMSNorm -> AllGather/reshard -> Dot_2
```

The intended final target is TPU/XLA, most likely TPU7x (Ironwood). A three-NVIDIA-GPU environment will be
used first to develop the methodology, compiler tooling, cost model, and
distributed experiments. TPU hardware will later be required for calibration
and final TPU-specific performance claims.

The intended outcome is a combination of a thesis, research publication, and
potential upstream contribution. The initial implementation need only be a
restricted HLO/JAX-driven prototype; an upstream-quality general XLA C++ pass
is not required.

## 2. Proposed novelty

The novelty is **not** simply chunking a collective or overlapping one
collective with one matrix multiplication. XLA already contains windowed-einsum
lowering, asynchronous collectives, and latency-hiding scheduling.

The proposed novelty is:

> Region-level, memory-constrained co-optimization of sharding, communication
> decomposition, tiling, collective realization, and scheduling across multiple
> producer and consumer operations.

The central hypothesis is that independently making these decisions can produce
a locally fast operation but a globally slow region. For example, the fastest
lowering of `Dot_1` may require an expensive reshard before `Dot_2`, while a
slightly slower `Dot_1` configuration may eliminate that communication and
reduce end-to-end latency.

Communication avoidance must be considered alongside communication overlap.
Candidate decisions may include:

- retaining a sharded intermediate instead of materializing an all-gather;
- changing the consumer's sharding;
- selecting reduce-scatter instead of all-reduce where legal;
- selecting tile size and the number of in-flight tiles;
- choosing single or double buffering;
- caching, recomputing, or rematerializing intermediates;
- selecting among backend-supported collective realizations.

## 3. Core research questions

1. When does region-level optimization outperform stock XLA partitioning,
   windowed einsum, and latency-hiding scheduling?
2. When is avoiding or transforming communication better than overlapping it?
3. How do tile size and pipeline depth trade off collective startup, compute
   efficiency, overlap, and peak memory?
4. Can a cost model predict the best configuration accurately enough to guide
   a compiler transformation?
5. Which parts of the optimizer transfer between GPU and TPU, and which require
   backend-specific models or lowering rules?

## 4. Scope and boundaries

### In initial scope

- JAX-generated StableHLO/HLO.
- XLA SPMD sharding and collective insertion.
- `dot`, `reduce-scatter`, `all-gather`, and an elementwise operation.
- Tile-level producer/communication/consumer pipelines.
- Makespan and peak-memory optimization.
- Three-GPU experiments followed by TPU validation.

### Deferred until the basic region works

- RMSNorm and other cross-tile reductions.
- Full transformer blocks.
- Automatic optimization of arbitrary HLO graphs.
- Large topology-aware collective search spaces.
- Training-scale end-to-end evaluation.

## 5. Development strategy

The project should keep three layers separate:

```text
Region optimizer
  Hardware-independent candidates, constraints, and search

Hardware cost model
  GPU or TPU compute, communication, and memory estimates

Backend realization
  Backend-supported HLO transformations and schedules
```

A candidate configuration can be represented as:

```python
Candidate(
    producer_sharding=...,
    consumer_sharding=...,
    tile_shape=...,
    collective=...,
    pipeline_depth=...,
    buffer_count=...,
)
```

This separation prevents the search algorithm from depending directly on NCCL,
CUDA streams, TPU ICI, or MXU-specific details.

## 6. Phase A: analytical simulator

### Goal

Build a small discrete-event simulator before modifying XLA. Given operation
costs and a memory budget, it should find the lowest-makespan feasible tiled
schedule.

### Initial model

For a tile of size `b`, begin with:

\[
T_{compute}(b) =
\frac{\operatorname{FLOPs}(b)}
     {\operatorname{peak\ FLOPs}\,\eta(b)}
\]

where \(\eta(b)\) models reduced accelerator efficiency for unsuitable tiles.

For a collective transferring `x` bytes across `p` devices, begin with:

\[
T_{comm}(x,p) = \alpha + \frac{x}{\beta} f(p)
\]

where \(\alpha\) is startup latency, \(\beta\) is bandwidth, and \(f(p)\)
approximates the collective realization. Measured lookup tables or piecewise
models can replace this formula later.

### Schedules to compare

1. Serialized compute and communication.
2. Whole-operation asynchronous overlap.
3. Single-buffer tiled pipeline.
4. Double-buffer tiled pipeline.

The simulator must enforce data dependencies, resource availability, buffer
lifetime, and a peak-memory constraint.

### Initial sweep

- tile sizes;
- tensor shapes;
- device count;
- collective startup and bandwidth;
- producer and consumer costs;
- one versus two buffers;
- number of in-flight tiles;
- memory budget.

### Deliverables

- Unit-tested schedule simulator.
- CSV output containing configuration, makespan, and peak memory.
- Makespan-versus-tile-size plot.
- At least one case where tiling wins and one where startup, compute
  underutilization, or memory makes it lose.

## 7. Phase B: understand stock XLA behavior on GPU

### Current GPU environment

- Three GPUs will be used: GPU IDs 4, 5, and 6 (the fifth through seventh
  physical GPUs).
- Each is an NVIDIA RTX PRO 6000 Blackwell Server Edition with approximately
  96 GiB of memory and CUDA compute capability 12.0.
- The GPUs use PCIe rather than NVLink. GPUs 4 and 5 share the closest PCIe path
  (`PIX`); paths from either GPU to GPU 6 are less local (`NODE`).
- All three GPUs are associated with NUMA node 3 (CPU affinity
  `72-95,216-239`).
- The host has two Intel Xeon 6960P sockets, 144 physical CPU cores, and
  approximately 1.5 TiB of system memory.
- Observed software baseline: NVIDIA driver 580.173.02 and CUDA toolkit 12.9.

### Environment record

Record the following with every experiment:

- GPU model and memory capacity;
- NVLink/NVSwitch or PCIe topology;
- whether execution crosses nodes and the inter-node network;
- JAX, `jaxlib`, XLA, CUDA, and NCCL versions;
- compiler flags and device mesh.

### Tasks

1. Implement the initial region in JAX.
2. Run it on a two-device mesh before scaling to all three selected GPUs.
3. Dump StableHLO and HLO before and after SPMD partitioning.
4. Annotate why each collective was inserted.
5. Determine whether collectives become asynchronous.
6. Capture a timeline trace and inspect actual overlap.
7. Compare configurations with relevant XLA optimizations enabled and disabled.

### Baselines

- Stock XLA configuration.
- Stock XLA with applicable windowed-einsum behavior.
- Stock XLA latency-hiding schedule.
- Whole-tensor asynchronous collective.
- Manually specified alternative shardings.

Windowed einsum is both a baseline and a possible realization that a future
region optimizer could select.

### Deliverable

An annotated HLO and trace explaining the sharding, inserted collectives,
dependencies, overlap, and memory behavior of the initial region.

## 8. Phase C: GPU measurements and model calibration

### Microbenchmarks

Measure:

- matrix multiplication across representative shapes;
- reduce-scatter across message sizes;
- all-gather across message sizes;
- back-to-back collectives;
- matmul and collective overlap;
- additional memory caused by overlap and buffering.

Collective message sizes should span latency-bound and bandwidth-bound regimes,
for example from kilobytes through hundreds of megabytes where hardware memory
allows.

### Calibration

Fit the simulator's compute and communication models to these measurements.
Compare predicted and observed:

- makespan;
- selected tile/configuration;
- peak memory;
- compute and communication utilization.

Prediction error should be analyzed rather than hidden: it identifies hardware
effects missing from the compiler model.

## 9. Phase D: joint optimization experiment

Enumerate or search over:

\[
(\text{producer sharding},\ \text{consumer sharding},\ \text{tile size},
\ \text{collective},\ \text{pipeline depth},\ \text{buffers})
\]

The key early objective is to find a counterexample to local optimization:

```text
Local plan:
  fastest Dot_1 -> collective/reshard -> fastest Dot_2

Region plan:
  compatible tiled Dot_1 -> pipelined or avoided communication -> Dot_2
```

The region plan should improve end-to-end makespan under the same memory budget,
even if one individual operation is slower.

### Evaluation metrics

- End-to-end region latency.
- Peak device memory.
- Communication volume.
- Observed compute/communication overlap.
- Compute and communication utilization.
- Compilation/search time.
- Cost-model prediction error.

## 10. Phase E: XLA prototype

Only after the opportunity is demonstrated by measurement:

1. Define the legal HLO pattern and required sharding conditions.
2. Generate candidate tiled forms or attach enough information for scheduling.
3. Express asynchronous collective dependencies correctly.
4. Preserve a fallback to the original HLO when no candidate is profitable.
5. Add correctness tests and structural HLO tests.
6. Compare the generated form with stock windowed-einsum lowering.

The first prototype may use a restricted pattern and an externally supplied
cost table. A general-purpose pass is not required for the initial result.

## 11. Phase F: TPU calibration and validation

TPU access is eventually required for TPU-specific claims. GPU results cannot
establish TPU performance because NCCL, GPU interconnects, Tensor Cores, and
CUDA execution differ from TPU collectives, ICI, MXUs, and the TPU backend.

On a multi-device TPU slice, likely TPU7x (Ironwood):

1. Repeat dot and collective microbenchmarks.
2. Replace the GPU cost profile with a TPU cost profile.
3. Check which candidate realizations the TPU backend supports.
4. Inspect final TPU HLO and execution traces.
5. Validate physical overlap rather than assuming it from HLO structure.
6. Compare with stock TPU XLA, windowed einsum, and latency-hiding scheduling.
7. Evaluate representative LLM dimensions and, later, a larger block.

The final report must present GPU and TPU results separately.

### Likely Ironwood starting configuration

Ironwood exposes two JAX devices per physical chip. Therefore, experiment logs
must distinguish physical chips, JAX devices, hosts, and the physical topology.

- A `2x2x1` slice contains four chips on one host and is suitable for initial
  correctness, HLO, and intra-host collective experiments.
- A `2x2x2` slice contains eight chips across two hosts and is the preferred
  minimum target for studying topology and multi-host communication effects.

The plan must not assume either configuration until actual allocation details
are known. Larger topologies can be added for scaling experiments if access and
budget permit.

## 12. Suggested first four weeks

### Week 1: simulator foundation

- Implement a two-stage and then three-stage discrete-event pipeline.
- Add resource and dependency constraints.
- Add buffer lifetime and peak-memory tracking.
- Plot makespan against tile size using synthetic hardware parameters.

### Week 2: distributed JAX/XLA baseline

- Set up JAX on the GPU environment.
- Run a sharded dot and the initial two-dot region.
- Dump and annotate HLO.
- Capture one execution trace.

### Week 3: GPU microbenchmarks

- Measure dot and collective costs.
- Measure overlap.
- Calibrate the simulator.
- Report predicted-versus-measured errors.

### Week 4: first joint search

- Enumerate shardings, tile sizes, and buffer counts.
- Compare the chosen plan with stock XLA baselines.
- Document one win, one loss, and why each occurs.

## 13. Immediate first task

Implement:

```python
def simulate_pipeline(
    tile_count: int,
    producer_us: float,
    communication_us: float,
    consumer_us: float,
    tile_bytes: int,
    buffer_count: int,
    memory_budget_bytes: int,
) -> ScheduleResult:
    ...
```

The first test should use three tiles and fixed producer, communication, and
consumer durations. Verify dependency ordering, resource serialization, buffer
reuse, makespan, and peak memory by hand.

The first research milestone is reached when the simulator can explain why a
particular tile size is optimal and why both smaller and larger tiles lose.

## 14. Risks and mitigation

| Risk | Mitigation |
|---|---|
| Existing XLA behavior already covers the proposed case | Maintain a capability matrix based on current source and experiments; narrow the contribution to a demonstrated gap. |
| GPU conclusions do not transfer to TPU | Keep optimizer, hardware model, and backend realization separate; make TPU claims only after TPU measurement. |
| Analytical model is inaccurate | Use measured lookup tables and report prediction error. |
| Search space becomes too large | Start with enumeration over a restricted region and add pruning only when needed. |
| More overlap causes excessive memory use | Treat memory as a hard constraint and model buffer lifetimes explicitly. |
| RMSNorm complicates tile independence | Begin with an elementwise activation and add cross-tile reductions later. |

## 15. Confirmed decisions and open details

### Confirmed

- Intended outcome: a thesis, research publication, and potential upstream
  contribution.
- Initial implementation scope: a restricted prototype is sufficient; a full
  upstream XLA pass is not required.
- Likely final TPU target: TPU7x (Ironwood).

### Still unknown

These questions do not block Phase A, but should be resolved before GPU
experiments are finalized:

1. Which exact Ironwood slice topology, host count, allocation mechanism, and
   usage budget will be available?
2. Can experiments reserve a stable topology long enough for reproducible
   measurements?
