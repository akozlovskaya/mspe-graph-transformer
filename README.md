# Multi-Scale Structural Positional Encodings for Graph Transformers

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A unified framework for multi-scale structural positional encodings (PEs) in Graph Transformers.
This repository accompanies a master's thesis on positional encodings for graph neural networks.

---

## Project Goal

Investigate how multi-scale positional encodings improve Graph Transformer performance, 
particularly for long-range dependencies in molecular and structural graphs.

**Key Research Questions:**
1. How do different PE types (spectral, diffusion, structural) affect model performance?
2. Can multi-scale PEs improve long-range dependency modeling?
3. What are the efficiency-accuracy trade-offs of various PE configurations?

---

## Features

### Multi-Scale Node Positional Encodings
- **LapPE** — Laplacian eigenvector encodings with sign-invariant processing
- **RWSE** — Random-walk structural encodings at multiple scales
- **HKS** — Heat kernel signatures with multi-scale diffusion times
- **Role** — Structural features (degree, clustering, PageRank)

### Relative Structural Encodings
- **SPD Buckets** — Shortest-path distance with learnable buckets
- **Diffusion PE** — Heat kernel pairwise similarity
- **Effective Resistance** — Low-rank approximation
- **Landmark SPD** — Scalable distance approximation

### Graph Transformer Architecture
- GPS-style hybrid local MPNN + global attention
- Residual/gated PE mixing
- Multi-head attention with structural biases
- Support for deep models (up to 16 layers)

### Comprehensive Pipeline
- Unified dataset interface (ZINC, QM9, LRGB, OGB, synthetic)
- PE precomputation and caching
- Long-range dependency evaluation
- Efficiency profiling
- Thesis-ready result generation

### Synthetic Graph Benchmarks
- **Controlled evaluation** of PEs under known graph structures
- **6 benchmark tasks**: Pairwise distance, structural role, diffusion source, etc.
- **9 graph generators**: ER, WS, BA, SBM, geometric, regular, grid, ring, tree
- **Cross-graph generalization** and size extrapolation experiments
- **Fully reproducible** with deterministic generation

---

## Installation

```bash
# Clone repository
git clone https://github.com/akozlovskaya/mspe-graph-transformer.git
cd mspe-graph-transformer

# Create environment (recommended)
conda create -n mspe python=3.11
conda activate mspe

# Install dependencies
pip install -r requirements.txt

# Install PyTorch Geometric (adjust for your CUDA version)
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install torch-geometric
```

---

## Dataset Setup

Before running experiments, you need to download datasets and optionally precompute positional encodings.

### Downloading Datasets

Download datasets using the download script:

```bash
# Download a specific dataset
python scripts/download_datasets.py dataset=zinc

# Download multiple datasets
python scripts/download_datasets.py dataset=zinc
python scripts/download_datasets.py dataset=qm9
python scripts/download_datasets.py dataset=peptides_func

# Download with custom root directory
python scripts/download_datasets.py dataset=peptides_func root=./data
```

**Note:** Datasets are automatically downloaded by PyTorch Geometric on first access, but using the download script allows you to:
- Pre-download datasets before experiments
- Verify dataset integrity
- Check dataset statistics

### Precomputing Positional Encodings (Optional but Recommended)

For faster training, you can precompute PE for datasets:

```bash
# Precompute PE with default configuration
python scripts/preprocess_pe.py dataset=zinc pe=default

# Precompute PE with custom configuration
python scripts/preprocess_pe.py dataset=zinc pe=mspe pe_cache_dir=./data/pe_cache

# For synthetic datasets
python scripts/preprocess_pe.py dataset=synthetic/pairwise_distance pe=light
```

Preprocessed data is saved to `{pe_cache_dir}/{dataset_name}/` (defaults to `{dataset.root}/pe_cache/{dataset_name}/`) and can be loaded faster during training. The training script will automatically use preprocessed PE if available (controlled by `use_preprocessed_pe: true` in config).

**Benefits of preprocessing:**
- Faster training startup (PE computed once, not on-the-fly)
- Consistent PE across experiments
- Ability to experiment with different PE configurations without recomputing

**Note:** If you skip preprocessing, PE will be computed on-the-fly during training, which is slower but works fine for small datasets.

---

## Repository Structure

```
mspe-graph-transformer/
├── configs/                          # Hydra configuration files
│   ├── config.yaml                   # Main config
│   ├── dataset/                      # Dataset configs
│   │   └── synthetic/                # Synthetic benchmark configs
│   ├── model/                        # Model configs
│   ├── pe/                           # PE configs
│   ├── train/                        # Training configs
│   └── experiments/                  # Predefined experiments
│       └── synthetic/                 # Synthetic experiment configs
├── data/                             # Data directory (auto-created)
├── src/
│   ├── dataset/                      # Dataset loaders and transforms
│   ├── pe/
│   │   ├── node/                     # Node-wise PEs (LapPE, RWSE, HKS)
│   │   └── relative/                 # Relative PEs (SPD, Diffusion)
│   ├── models/                       # Model architectures
│   ├── training/                     # Training loop and utilities
│   ├── evaluation/                   # Evaluation and long-range analysis
│   ├── experiments/                  # Experiment orchestration
│   ├── profiling/                    # Efficiency profiling
│   ├── results/                      # Result processing and visualization
│   └── utils/                        # Common utilities
├── scripts/                          # CLI scripts
│   ├── download_datasets.py           # Download datasets
│   ├── preprocess_pe.py              # Precompute positional encodings
│   ├── train.py                      # Training script
│   ├── evaluate.py                   # Evaluation script
│   ├── run_experiment.py             # Single experiment
│   ├── run_sweep.py                  # Sweep execution
│   ├── make_tables.py                # Generate tables
│   ├── make_plots.py                 # Generate figures
│   └── validate_thesis_pipeline.py   # Validation
├── thesis/                           # Thesis alignment files
│   ├── figures_map.yaml              # Figure to experiment mapping
│   └── tables_map.yaml               # Table to experiment mapping
├── tests/                            # Unit and integration tests
├── notebooks/                        # Jupyter notebooks
├── outputs/                          # Experiment outputs (auto-created)
└── results/                          # Generated tables and figures
```

---

## Quick Start

### 0. Download Datasets (First Time Only)

Before training, download the datasets you need:

```bash
# Download ZINC dataset
python scripts/download_datasets.py dataset=zinc

# Optionally precompute PE for faster training
python scripts/preprocess_pe.py dataset=zinc pe=default
```

### 1. Train a Model

```bash
# Basic training
python scripts/train.py dataset=zinc model=default pe=mspe

# With custom parameters
python scripts/train.py \
    dataset=peptides_func \
    model=default \
    model.num_layers=8 \
    pe.node.type=combined \
    training.epochs=200
```

### 2. Run Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint outputs/experiment/checkpoints/best.pt \
    --dataset zinc
```

### 3. Long-Range Evaluation

```bash
python scripts/evaluate_long_range.py \
    --checkpoint outputs/experiment/checkpoints/best.pt \
    --dataset peptides_func \
    --max_distance 20
```

### 4. Generate Results

```bash
# Generate thesis tables
python scripts/make_tables.py --tables all --output_format latex

# Generate thesis figures
python scripts/make_plots.py --plots all --output_dir results/figures
```

### 5. Synthetic Graph Benchmarks

```bash
# Pairwise distance classification (Task A)
python scripts/train.py dataset=synthetic/pairwise_distance pe=mspe

# Structural role classification (Task C)
python scripts/train.py dataset=synthetic/structural_role pe.node.type=lap

# PE capacity stress test
python scripts/run_sweep.py --sweep configs/experiments/synthetic/pe_capacity.yaml

# Cross-graph generalization
python scripts/run_experiment.py experiment=synthetic/cross_graph_generalization
```

---

## Reproducing Thesis Results

All thesis results can be reproduced using predefined experiment configurations.

**Prerequisites:** Before running experiments, download required datasets:

```bash
# Download all datasets used in thesis
python scripts/download_datasets.py dataset=zinc
python scripts/download_datasets.py dataset=qm9
python scripts/download_datasets.py dataset=peptides_func
python scripts/download_datasets.py dataset=peptides_struct

# Optionally precompute PE for faster experiments
python scripts/preprocess_pe.py dataset=zinc pe=mspe
python scripts/preprocess_pe.py dataset=peptides_func pe=mspe
```

### Main Performance Table (Table 4.1)

```bash
# Run all baseline and MSPE experiments
python scripts/run_sweep.py --sweep configs/experiments/pe_ablation.yaml

# Generate table
python scripts/make_tables.py --tables performance --output_format latex
```

### PE Ablation Study (Table 4.2)

```bash
python scripts/run_sweep.py --sweep configs/experiments/pe_ablation.yaml
python scripts/make_tables.py --tables ablation --ablation_key node_pe
```

### Long-Range Analysis (Figure 4.3)

```bash
python scripts/make_plots.py --plots performance_vs_distance --metric mae
```

### Reproducibility with Seeds

```bash
# Run same experiment with multiple seeds
python scripts/run_sweep.py --sweep configs/experiments/seed_sweep.yaml
```

### Full Thesis Pipeline Validation

```bash
python scripts/validate_thesis_pipeline.py
```

### Synthetic Benchmark Experiments

```bash
# Pairwise distance classification with different PEs
python scripts/run_sweep.py --sweep configs/experiments/synthetic/pairwise_distance.yaml

# Structural role classification
python scripts/run_sweep.py --sweep configs/experiments/synthetic/structural_role.yaml

# PE capacity stress test (varying PE dimensions)
python scripts/run_sweep.py --sweep configs/experiments/synthetic/pe_capacity.yaml

# Cross-graph generalization (train on ER, test on WS)
python scripts/run_experiment.py experiment=synthetic/cross_graph_generalization

# Size extrapolation (train on small, test on large)
python scripts/run_experiment.py experiment=synthetic/size_extrapolation
```

---

## Supported Datasets

### Real-World Datasets

| Dataset | Type | Task | Graphs | Nodes (avg) |
|---------|------|------|--------|-------------|
| ZINC | Molecular | Regression | 12K | 23 |
| QM9 | Molecular | Regression | 130K | 18 |
| Peptides-func | Molecular | Multi-label | 15K | 150 |
| Peptides-struct | Molecular | Regression | 15K | 150 |
| ogbg-molhiv | Molecular | Binary | 41K | 26 |
| ogbg-molpcba | Molecular | Multi-label | 438K | 26 |
| PascalVOC-SP | Vision | Multi-class | 11K | 479 |
| CIFAR10-SP | Vision | Multi-class | 60K | 118 |

### Synthetic Graph Benchmarks

Synthetic datasets for controlled PE evaluation under known graph structures:

| Graph Type | Description | Use Case |
|------------|-------------|----------|
| **Erdős–Rényi** | Random graphs with edge probability p | Baseline connectivity |
| **Watts–Strogatz** | Small-world networks | Long-range dependencies |
| **Barabási–Albert** | Scale-free networks | Degree distribution effects |
| **SBM** | Stochastic Block Model | Community structure |
| **Random Geometric** | Spatial graphs | Distance-based tasks |
| **Regular** | Regular graphs | Structural role tasks |
| **Grid** | 2D/3D grids | Spatial reasoning |
| **Ring** | Cycle graphs | Simple topology |
| **Tree** | Balanced trees | Hierarchical structure |

**Benchmark Tasks:**
- **Task A**: Pairwise Distance Classification — classify if distance ≥ threshold
- **Task B**: Distance Regression — predict exact shortest-path distance
- **Task C**: Structural Role Classification — classify nodes by structural role
- **Task D**: Local vs Global Signal — predict from local (degree) or global (distance) signals
- **Task E**: Diffusion Source Identification — identify diffusion source node
- **Task F**: PE Capacity Stress Test — test PE capacity with varying dimensionality

See [`src/dataset/synthetic/README.md`](src/dataset/synthetic/README.md) for detailed documentation.

---

## Positional Encoding Reference

### Node-wise PEs

| PE | Formula | Scales | Sign Invariant |
|----|---------|--------|----------------|
| LapPE | $\phi_k(i)$ eigenvectors | k eigenpairs | ✓ (SignNet/flip) |
| RWSE | $p_i^{(t)} = (A^t)_{ii}$ | t ∈ {1,2,4,8,...} | N/A |
| HKS | $\sum_k e^{-\lambda_k t} \phi_k(i)^2$ | t ∈ {0.1,1,10,...} | ✓ |

### Relative PEs

| PE | Formula | Complexity | Sparse |
|----|---------|------------|--------|
| SPD | $d_{sp}(i,j)$ bucketed | O(VE) | ✓ |
| Diffusion | $K_t(i,j) = \sum_k e^{-\lambda_k t} \phi_k(i)\phi_k(j)$ | O(Vk²) | ✓ |
| Resistance | $R(i,j) = L^+_{ii} + L^+_{jj} - 2L^+_{ij}$ | O(Vk²) | ✓ |

---

## Configuration System

The project uses Hydra for configuration management. Key config groups:

```yaml
# configs/config.yaml
defaults:
  - dataset: zinc
  - model: default
  - pe: mspe
  - train: default

seed: 42
deterministic: true
```

Override any parameter from CLI:

```bash
python scripts/train.py \
    model.hidden_dim=512 \
    pe.node.dim=64 \
    training.lr=3e-4
```

---

## Development

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/test_pe.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

### Code Quality

```bash
# Linting
flake8 src/ scripts/

# Type checking
mypy src/
```

---

## Citation

If you use this repository in your research, please cite:

```bibtex
@mastersthesis{kozlovskaya2025mspe,
  title={Multi-Scale Structural Positional Encodings for Graph Transformers},
  author={Kozlovskaya, Anastasia},
  year={2025},
  school={University}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [Graphormer](https://github.com/microsoft/Graphormer)
- [GPS](https://github.com/rampasek/GraphGPS)
- [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric)
- [OGB](https://ogb.stanford.edu/)
