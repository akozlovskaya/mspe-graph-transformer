# Profiling Framework

Framework for profiling efficiency, memory usage, and scalability of Graph Transformer models.

## Structure

```
src/profiling/
├── __init__.py      # Public API
├── runtime.py       # Runtime profiling
├── memory.py        # Memory profiling
├── flops.py         # FLOPs estimation
├── scaling.py       # Scaling experiments
├── utils.py         # Utilities
└── README.md        # Documentation
```

## Quick Start

### Runtime Profiling

```python
from src.profiling import RuntimeProfiler, profile_forward

# Simple forward pass
stats = profile_forward(model, batch, num_runs=100, warmup_runs=10)
print(f"Forward: {stats.mean:.2f} ± {stats.std:.2f} ms")

# Full profiling
profiler = RuntimeProfiler(model, device, num_runs=100)
results = profiler.profile_all(batch, optimizer=optimizer)
print(f"Forward: {results['forward']}")
print(f"Backward: {results['backward']}")
print(f"Training step: {results['training_step']}")
```

### Memory Profiling

```python
from src.profiling import MemoryProfiler, get_memory_breakdown

# Peak memory
profiler = MemoryProfiler(model, device)
stats = profiler.profile_forward(batch)
print(f"Peak memory: {stats.peak_mb:.2f} MB")

# Memory breakdown
breakdown = get_memory_breakdown(model, batch, device)
print(f"Parameters: {breakdown['parameters']:.2f} MB")
print(f"Activations: {breakdown['forward_activations']:.2f} MB")
```

### FLOPs Estimation

```python
from src.profiling import FLOPsEstimator, estimate_model_flops

# Quick estimate
estimate = estimate_model_flops(
    num_nodes=500,
    num_edges=2500,
    hidden_dim=256,
    num_layers=6,
    num_heads=8,
    ffn_dim=512,
)
print(f"Total: {estimate.total / 1e9:.2f} GFLOPs")
print(f"Attention: {estimate.breakdown['attention'] / 1e9:.2f} GFLOPs")

# From model
estimator = FLOPsEstimator(model)
estimate = estimator.estimate(num_nodes=500, num_edges=2500)
params = estimator.count_parameters()
```

### Scaling Experiments

```python
from src.profiling import ScalingExperiment, run_node_scaling

# Node scaling
def model_factory(**kwargs):
    return get_model(hidden_dim=256, **kwargs)

result = run_node_scaling(
    model_factory,
    num_nodes_list=[100, 500, 1000, 2000],
)

for i, n in enumerate(result.parameter_values):
    print(f"N={n}: {result.runtime_stats[i].mean:.2f} ms")

# Full experiment
experiment = ScalingExperiment(model_factory, device)
experiment.run_node_scaling()
experiment.run_layer_scaling()
experiment.print_summary()
```

### Using Scripts

```bash
# Basic profiling
python scripts/profile_model.py \
    --model graph_transformer \
    --profile runtime memory flops

# With dataset
python scripts/profile_model.py \
    --dataset peptides_func \
    --model graph_transformer \
    --profile runtime memory flops scaling

# With custom parameters
python scripts/profile_model.py \
    --model graph_transformer \
    --hidden_dim 512 \
    --num_layers 12 \
    --num_nodes 1000 \
    --profile runtime memory
```

## Runtime Profiling

### Functions

```python
# Benchmark any function
stats = benchmark_function(
    fn, *args,
    num_runs=100,
    warmup_runs=10,
    **kwargs
)

# Forward pass
stats = profile_forward(model, batch, num_runs=100)

# Backward pass
stats = profile_backward(model, batch, loss_fn=nn.MSELoss())

# Training step
stats = profile_training_step(model, batch, optimizer, loss_fn)
```

### RuntimeStats

```python
@dataclass
class RuntimeStats:
    mean: float      # Mean time (ms)
    std: float       # Standard deviation
    min: float       # Minimum time
    max: float       # Maximum time
    num_runs: int    # Number of measurements
```

### CUDA Synchronization

All measurements are automatically synchronized with CUDA:

```python
def _cuda_synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
```

## Memory Profiling

### Functions

```python
# Reset stats before measurement
reset_memory_stats()

# Get current peak
peak_mb = get_peak_memory(device)

# Profile function
stats = profile_memory_usage(fn, *args, device=device)

# Memory breakdown
breakdown = get_memory_breakdown(model, batch, device)
```

### MemoryStats

```python
@dataclass
class MemoryStats:
    peak_mb: float       # Peak GPU memory
    allocated_mb: float  # Currently allocated
    reserved_mb: float   # Reserved by allocator
    device: str          # Device string
```

### Attention Memory Estimation

```python
# Estimation without running model
mem_mb = estimate_attention_memory(
    num_nodes=1000,
    num_heads=8,
    hidden_dim=256,
    batch_size=1,
)
```

## FLOPs Estimation

### Assumptions

FLOPs estimates are based on the following assumptions:
- Matrix multiplication A(m×k) @ B(k×n): 2 × m × k × n FLOPs
- Element-wise ops: 1 FLOP per element
- Softmax: ~5N FLOPs per row
- LayerNorm: ~5N FLOPs
- Activation (GELU): ~4 FLOPs per element

### Components

```python
# Linear layer
flops = estimate_linear_flops(in_features, out_features, batch_size, seq_len)

# Attention
flops = estimate_attention_flops(num_nodes, hidden_dim, num_heads, sparse_ratio=1.0)

# MPNN
flops = estimate_mpnn_flops(num_nodes, num_edges, in_features, out_features, mpnn_type="gin")

# FFN
flops = estimate_ffn_flops(hidden_dim, ffn_dim, num_nodes)

# PE computation
flops = estimate_pe_flops(num_nodes, pe_dim, pe_type="lap")
```

### FLOPsEstimate

```python
@dataclass
class FLOPsEstimate:
    total: int                    # Total FLOPs
    breakdown: Dict[str, int]     # FLOPs per component
```

## Scaling Experiments

### Node Scaling

```python
result = run_node_scaling(
    model_factory,
    num_nodes_list=[100, 500, 1000, 2000, 5000],
    avg_degree=5,
    device=device,
)
```

### Layer Scaling

```python
result = run_layer_scaling(
    model_factory,
    num_layers_list=[2, 4, 6, 8, 12, 16],
    num_nodes=500,
    device=device,
)
```

### PE Scaling

```python
results = run_pe_scaling(
    model_factory,
    pe_configs=[
        {"name": "lap", "dim": 16},
        {"name": "rwse", "dim": 16},
        {"name": "none", "dim": 0},
    ],
)
```

### Synthetic Graphs

```python
graphs = generate_scaling_graphs(
    num_nodes_list=[100, 500, 1000],
    avg_degree=5,
    feature_dim=16,
    seed=42,
)
```

## Profiling Context

For correct profiling, use the context:

```python
with ProfilingContext(model, disable_dropout=True, eval_mode=True, seed=42):
    # Profiling here
    stats = profile_forward(model, batch)
```

The context:
- Disables dropout
- Sets model to eval mode
- Sets seed
- Restores state after exit

## Output Format

### JSON Output

```json
{
  "timestamp": "2024-01-01T12:00:00",
  "hardware": {
    "cuda_available": true,
    "gpu_name": "NVIDIA A100",
    "gpu_memory_gb": 40.0
  },
  "model_info": {
    "total_parameters": 1234567,
    "parameter_memory_mb": 4.71
  },
  "runtime": {
    "forward": {"mean": 10.5, "std": 1.2, "unit": "ms"},
    "backward": {"mean": 25.3, "std": 2.1, "unit": "ms"}
  },
  "memory": {
    "peak_mb": 512.0,
    "breakdown": {
      "parameters": 50.0,
      "activations": 400.0
    }
  },
  "flops": {
    "total_gflops": 12.5,
    "breakdown": {
      "attention": 8.0,
      "ffn": 3.5
    }
  }
}
```

## Determinism

Profiling is deterministic when:
- Fixed seed
- Dropout disabled
- Same batch sizes

```python
set_seed(42)
with ProfilingContext(model, seed=42):
    # Reproducible results
    ...
```

## PE Cost Analysis

### Precomputation Time

```python
from src.profiling.runtime import profile_pe_computation

stats = profile_pe_computation(pe_transform, data, num_runs=10)
print(f"PE precomputation: {stats.mean:.2f} ms (one-time)")
```

### Storage Size

```python
from src.profiling.memory import estimate_pe_storage

storage = estimate_pe_storage(
    num_nodes=1000,
    node_pe_dim=32,
    relative_pe_buckets=16,
    sparse_ratio=0.1,
)
print(f"Node PE: {storage['node_pe']:.2f} MB")
print(f"Relative PE (sparse): {storage['relative_pe_sparse']:.2f} MB")
```

## Best Practices

1. **Warmup**: Always use warmup runs for stable measurements
2. **Batch size**: Use the same batch size for comparisons
3. **CUDA sync**: All measurements are automatically synchronized
4. **Eval mode**: Use eval mode for forward-only profiling
5. **Determinism**: Use ProfilingContext for reproducibility
