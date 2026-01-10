# Dataset Documentation

This document provides a comprehensive description of all datasets available in the project, including their data structure, features, storage format, size, task types, and target representations.

## Table of Contents

1. [Molecular Datasets](#molecular-datasets)
   - [ZINC](#zinc)
   - [QM9](#qm9)
2. [LRGB Datasets](#lrgb-datasets)
   - [Peptides-func](#peptides-func)
   - [Peptides-struct](#peptides-struct)
   - [PascalVOC-SP](#pascalvoc-sp)
   - [CIFAR10-SP](#cifar10-sp)
3. [OGB Datasets](#ogb-datasets)
   - [ogbg-molhiv](#ogbg-molhiv)
   - [ogbg-molpcba](#ogbg-molpcba)
   - [PCQM4M](#pcqm4m)
   - [PCQM-Contact](#pcqm-contact)
4. [Synthetic Datasets](#synthetic-datasets)
   - [Graph Generators](#graph-generators)
   - [Benchmark Tasks](#benchmark-tasks)

---

## Molecular Datasets

### ZINC

**Description**: ZINC is a molecular dataset for regression tasks, containing molecular graphs with associated molecular properties.

**Data Source**: PyTorch Geometric's ZINC dataset (subset of the ZINC database)

**Task Type**: Graph-level regression

**Target**:
- **Format**: `torch.Tensor` of shape `[1]` (scalar value)
- **Type**: `torch.float32`
- **Meaning**: Molecular property value (typically a continuous scalar)
- **Range**: Depends on the specific property, typically normalized
- **Storage**: Stored in `data.y` attribute of each graph

**Node Features**:
- **Format**: `torch.Tensor` of shape `[num_nodes, num_features]`
- **Type**: `torch.float32`
- **Dimension**: 9 (one-hot encoding of atom types)
- **Content**: One-hot encoded atom types (C, N, O, F, P, S, Cl, Br, I)
- **Storage**: Stored in `data.x` attribute

**Edge Features**:
- **Format**: `torch.Tensor` of shape `[num_edges, edge_dim]` (optional)
- **Type**: `torch.float32`
- **Content**: Bond type encoding (single, double, triple, aromatic)
- **Storage**: Stored in `data.edge_attr` attribute (may be None)

**Graph Structure**:
- **Format**: `torch.Tensor` of shape `[2, num_edges]`
- **Type**: `torch.long`
- **Content**: Edge indices in COO format `[source_nodes, target_nodes]`
- **Storage**: Stored in `data.edge_index` attribute
- **Direction**: Undirected (edges appear in both directions)

**Dataset Size**:
- **Full dataset**: ~250,000 molecules
- **Subset (default)**: 10,000 train, 1,000 val, 1,000 test
- **Full dataset**: ~250,000 train, ~5,000 val, ~5,000 test

**Storage Format**:
- **Location**: `{root}/ZINC/`
- **Format**: PyTorch Geometric's native format (processed from raw SDF files)
- **Download**: Automatic on first use via PyTorch Geometric
- **Size on disk**: 
  - Subset: ~50-100 MB
  - Full: ~500 MB - 1 GB

**Splits**:
- **Type**: Predefined splits provided by PyTorch Geometric
- **Train**: First 10,000 (subset) or ~250,000 (full) molecules
- **Val**: Next 1,000 (subset) or ~5,000 (full) molecules
- **Test**: Last 1,000 (subset) or ~5,000 (full) molecules

**Typical Graph Statistics**:
- **Average nodes per graph**: 20-30
- **Average edges per graph**: 40-60
- **Node count range**: 5-50 nodes
- **Edge count range**: 10-100 edges

**Usage Example**:
```python
from src.dataset import get_dataset

dataset = get_dataset(
    name="zinc",
    root="./data",
    subset=True,  # Use subset (10k train, 1k val, 1k test)
    pe_config={
        "node": {"enabled": True, "type": "rwse", "dim": 32},
        "relative": {"enabled": True, "type": "spd", "num_buckets": 32}
    }
)

train_data, val_data, test_data = dataset.get_splits()
print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
print(f"Node features: {dataset.num_features}, Target dim: {dataset.num_classes}")
```

---

### QM9

**Description**: QM9 is a quantum-mechanical molecular dataset containing 19 different molecular properties computed using density functional theory (DFT).

**Data Source**: PyTorch Geometric's QM9 dataset (subset of GDB-9 database)

**Task Type**: Graph-level regression (multi-target, but one target selected at a time)

**Target**:
- **Format**: `torch.Tensor` of shape `[1]` (scalar value)
- **Type**: `torch.float32`
- **Meaning**: One of 19 quantum-mechanical properties (selected via `target_idx` parameter)
- **Available targets** (indices 0-18):
  - 0: mu (dipole moment)
  - 1: alpha (isotropic polarizability)
  - 2: HOMO (highest occupied molecular orbital energy)
  - 3: LUMO (lowest unoccupied molecular orbital energy)
  - 4: gap (HOMO-LUMO gap)
  - 5: R2 (electronic spatial extent)
  - 6: ZPVE (zero-point vibrational energy)
  - 7: U0 (internal energy at 0K)
  - 8: U (internal energy at 298.15K)
  - 9: H (enthalpy at 298.15K)
  - 10: G (free energy at 298.15K)
  - 11: Cv (heat capacity at 298.15K)
  - 12-18: Other properties
- **Range**: Property-dependent, typically normalized if `normalize_targets=True`
- **Normalization**: Mean and std computed from training set only
- **Storage**: Stored in `data.y` attribute

**Node Features**:
- **Format**: `torch.Tensor` of shape `[num_nodes, num_features]`
- **Type**: `torch.float32`
- **Dimension**: 11
- **Content**: 
  - One-hot encoding of atom type (C, N, O, F)
  - Additional atomic properties (formal charge, hybridization, etc.)
- **Storage**: Stored in `data.x` attribute

**Edge Features**:
- **Format**: `torch.Tensor` of shape `[num_edges, edge_dim]` (optional)
- **Type**: `torch.float32`
- **Content**: Bond type and distance information
- **Storage**: Stored in `data.edge_attr` attribute (may be None)

**Graph Structure**:
- **Format**: `torch.Tensor` of shape `[2, num_edges]`
- **Type**: `torch.long`
- **Content**: Edge indices in COO format
- **Storage**: Stored in `data.edge_index` attribute
- **Direction**: Undirected

**Dataset Size**:
- **Total**: ~133,000 molecules
- **Train**: 100,000 molecules
- **Val**: 10,000 molecules
- **Test**: ~23,000 molecules (remaining)

**Storage Format**:
- **Location**: `{root}/qm9/`
- **Format**: PyTorch Geometric's native format
- **Download**: Automatic on first use
- **Size on disk**: ~2-3 GB

**Splits**:
- **Type**: Fixed split (first 100k train, next 10k val, rest test)
- **Train**: Indices 0-99,999
- **Val**: Indices 100,000-109,999
- **Test**: Indices 110,000-end

**Typical Graph Statistics**:
- **Average nodes per graph**: 18
- **Average edges per graph**: 18-19
- **Node count range**: 3-29 nodes (all molecules have ≤29 atoms)
- **Edge count range**: 2-28 edges

**Usage Example**:
```python
from src.dataset import get_dataset

dataset = get_dataset(
    name="qm9",
    root="./data",
    target_idx=0,  # Predict dipole moment (mu)
    normalize_targets=True,  # Normalize targets using training set statistics
    pe_config={
        "node": {"enabled": True, "type": "lap", "dim": 32},
        "relative": {"enabled": True, "type": "spd", "num_buckets": 32}
    }
)

train_data, val_data, test_data = dataset.get_splits()
print(f"Target mean: {dataset.target_mean}, std: {dataset.target_std}")
```

---

## LRGB Datasets

### Peptides-func

**Description**: Peptides-func is a graph classification dataset from the LRGB benchmark, containing peptide graphs with functional property labels.

**Data Source**: LRGB (Long Range Graph Benchmark) dataset

**Task Type**: Graph-level multi-label classification

**Target**:
- **Format**: `torch.Tensor` of shape `[10]` (10 binary labels)
- **Type**: `torch.float32` or `torch.long`
- **Meaning**: 10 binary functional properties of peptides
- **Content**: Binary labels indicating presence/absence of each functional property
- **Storage**: Stored in `data.y` attribute

**Node Features**:
- **Format**: `torch.Tensor` of shape `[num_nodes, num_features]`
- **Type**: `torch.float32`
- **Dimension**: 9
- **Content**: Amino acid type encoding and structural features
- **Storage**: Stored in `data.x` attribute

**Edge Features**:
- **Format**: `torch.Tensor` of shape `[num_edges, edge_dim]` (optional)
- **Type**: `torch.float32`
- **Content**: Bond type and distance information
- **Storage**: Stored in `data.edge_attr` attribute

**Graph Structure**:
- **Format**: `torch.Tensor` of shape `[2, num_edges]`
- **Type**: `torch.long`
- **Content**: Edge indices in COO format
- **Storage**: Stored in `data.edge_index` attribute

**Dataset Size**:
- **Train**: ~15,000 graphs
- **Val**: ~2,000 graphs
- **Test**: ~2,000 graphs

**Storage Format**:
- **Location**: `{root}/Peptides-func/`
- **Format**: PyTorch Geometric's native format
- **Download**: Automatic on first use
- **Size on disk**: ~100-200 MB

**Splits**:
- **Type**: Predefined splits (stored in `data.split` attribute)
- **Train/Val/Test**: Automatically extracted from dataset

**Typical Graph Statistics**:
- **Average nodes per graph**: 150-200
- **Average edges per graph**: 300-400
- **Node count range**: 50-500 nodes
- **Edge count range**: 100-1000 edges

**Usage Example**:
```python
from src.dataset import get_dataset

dataset = get_dataset(
    name="peptides_func",
    root="./data",
    pe_config={
        "node": {"enabled": True, "type": "rwse", "dim": 32},
        "relative": {"enabled": True, "type": "spd", "num_buckets": 32}
    }
)

train_data, val_data, test_data = dataset.get_splits()
print(f"Num classes: {dataset.num_classes}")  # 10 binary tasks
```

---

### Peptides-struct

**Description**: Peptides-struct is a graph classification dataset from LRGB, containing peptide graphs with structural property labels.

**Data Source**: LRGB (Long Range Graph Benchmark) dataset

**Task Type**: Graph-level multi-class classification

**Target**:
- **Format**: `torch.Tensor` of shape `[1]` or `[11]` (depending on representation)
- **Type**: `torch.long` (class index) or `torch.float32` (one-hot)
- **Meaning**: Structural property class (11 classes)
- **Content**: Class index or one-hot encoding
- **Storage**: Stored in `data.y` attribute

**Node Features**:
- **Format**: `torch.Tensor` of shape `[num_nodes, num_features]`
- **Type**: `torch.float32`
- **Dimension**: 9
- **Content**: Amino acid type encoding and structural features
- **Storage**: Stored in `data.x` attribute

**Edge Features**:
- **Format**: `torch.Tensor` of shape `[num_edges, edge_dim]` (optional)
- **Type**: `torch.float32`
- **Content**: Bond type and distance information
- **Storage**: Stored in `data.edge_attr` attribute

**Graph Structure**:
- **Format**: `torch.Tensor` of shape `[2, num_edges]`
- **Type**: `torch.long`
- **Content**: Edge indices in COO format
- **Storage**: Stored in `data.edge_index` attribute

**Dataset Size**:
- **Train**: ~15,000 graphs
- **Val**: ~2,000 graphs
- **Test**: ~2,000 graphs

**Storage Format**:
- **Location**: `{root}/Peptides-struct/`
- **Format**: PyTorch Geometric's native format
- **Download**: Automatic on first use
- **Size on disk**: ~100-200 MB

**Splits**:
- **Type**: Predefined splits
- **Train/Val/Test**: Automatically extracted

**Typical Graph Statistics**:
- **Average nodes per graph**: 150-200
- **Average edges per graph**: 300-400
- **Node count range**: 50-500 nodes
- **Edge count range**: 100-1000 edges

---

### PascalVOC-SP

**Description**: PascalVOC-SP is a superpixel graph dataset from LRGB, created from PascalVOC images by converting superpixels to nodes.

**Data Source**: LRGB (Long Range Graph Benchmark) dataset

**Task Type**: Graph-level multi-class classification

**Target**:
- **Format**: `torch.Tensor` of shape `[1]` or `[21]` (depending on representation)
- **Type**: `torch.long` (class index) or `torch.float32` (one-hot)
- **Meaning**: PascalVOC object class (21 classes: 20 object classes + background)
- **Content**: Class index or one-hot encoding
- **Storage**: Stored in `data.y` attribute

**Node Features**:
- **Format**: `torch.Tensor` of shape `[num_nodes, num_features]`
- **Type**: `torch.float32`
- **Dimension**: 3 (RGB values) or higher (with additional features)
- **Content**: RGB color values of superpixels, possibly with additional spatial/statistical features
- **Storage**: Stored in `data.x` attribute

**Edge Features**:
- **Format**: `torch.Tensor` of shape `[num_edges, edge_dim]` (optional)
- **Type**: `torch.float32`
- **Content**: Spatial relationships between superpixels
- **Storage**: Stored in `data.edge_attr` attribute

**Graph Structure**:
- **Format**: `torch.Tensor` of shape `[2, num_edges]`
- **Type**: `torch.long`
- **Content**: Edge indices in COO format (superpixel adjacency)
- **Storage**: Stored in `data.edge_index` attribute

**Node Positions** (if available):
- **Format**: `torch.Tensor` of shape `[num_nodes, 2]`
- **Type**: `torch.float32`
- **Content**: (x, y) coordinates of superpixel centroids
- **Storage**: Stored in `data.pos` attribute

**Dataset Size**:
- **Train**: ~10,000 graphs
- **Val**: ~1,000 graphs
- **Test**: ~1,000 graphs

**Storage Format**:
- **Location**: `{root}/PascalVOC-SP/`
- **Format**: PyTorch Geometric's native format
- **Download**: Automatic on first use
- **Size on disk**: ~500 MB - 1 GB

**Splits**:
- **Type**: Predefined splits
- **Train/Val/Test**: Automatically extracted

**Typical Graph Statistics**:
- **Average nodes per graph**: 200-300 (superpixels)
- **Average edges per graph**: 500-800
- **Node count range**: 50-1000 nodes
- **Edge count range**: 100-2000 edges

---

### CIFAR10-SP

**Description**: CIFAR10-SP is a superpixel graph dataset from LRGB, created from CIFAR-10 images by converting superpixels to nodes.

**Data Source**: LRGB (Long Range Graph Benchmark) dataset

**Task Type**: Graph-level multi-class classification

**Target**:
- **Format**: `torch.Tensor` of shape `[1]` or `[10]` (depending on representation)
- **Type**: `torch.long` (class index) or `torch.float32` (one-hot)
- **Meaning**: CIFAR-10 class (10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Content**: Class index or one-hot encoding
- **Storage**: Stored in `data.y` attribute

**Node Features**:
- **Format**: `torch.Tensor` of shape `[num_nodes, num_features]`
- **Type**: `torch.float32`
- **Dimension**: 3 (RGB values) or higher
- **Content**: RGB color values of superpixels
- **Storage**: Stored in `data.x` attribute

**Edge Features**:
- **Format**: `torch.Tensor` of shape `[num_edges, edge_dim]` (optional)
- **Type**: `torch.float32`
- **Content**: Spatial relationships between superpixels
- **Storage**: Stored in `data.edge_attr` attribute

**Graph Structure**:
- **Format**: `torch.Tensor` of shape `[2, num_edges]`
- **Type**: `torch.long`
- **Content**: Edge indices in COO format
- **Storage**: Stored in `data.edge_index` attribute

**Node Positions** (if available):
- **Format**: `torch.Tensor` of shape `[num_nodes, 2]`
- **Type**: `torch.float32`
- **Content**: (x, y) coordinates of superpixel centroids
- **Storage**: Stored in `data.pos` attribute

**Dataset Size**:
- **Train**: ~50,000 graphs
- **Val**: ~5,000 graphs
- **Test**: ~5,000 graphs

**Storage Format**:
- **Location**: `{root}/CIFAR10-SP/`
- **Format**: PyTorch Geometric's native format
- **Download**: Automatic on first use
- **Size on disk**: ~1-2 GB

**Splits**:
- **Type**: Predefined splits
- **Train/Val/Test**: Automatically extracted

**Typical Graph Statistics**:
- **Average nodes per graph**: 100-200 (superpixels)
- **Average edges per graph**: 300-500
- **Node count range**: 20-500 nodes
- **Edge count range**: 50-1000 edges

---

## OGB Datasets

### ogbg-molhiv

**Description**: ogbg-molhiv is a molecular property prediction dataset from OGB, containing molecular graphs with binary HIV activity labels.

**Data Source**: Open Graph Benchmark (OGB)

**Task Type**: Graph-level binary classification

**Target**:
- **Format**: `torch.Tensor` of shape `[1]` (binary label)
- **Type**: `torch.long` or `torch.float32`
- **Meaning**: Binary label indicating HIV activity (0 = inactive, 1 = active)
- **Content**: Single binary value
- **Storage**: Stored in `data.y` attribute

**Node Features**:
- **Format**: `torch.Tensor` of shape `[num_nodes, num_features]`
- **Type**: `torch.float32`
- **Dimension**: 9 (one-hot encoding of atom types)
- **Content**: One-hot encoded atom types
- **Storage**: Stored in `data.x` attribute

**Edge Features**:
- **Format**: `torch.Tensor` of shape `[num_edges, edge_dim]` (optional)
- **Type**: `torch.float32`
- **Content**: Bond type encoding
- **Storage**: Stored in `data.edge_attr` attribute

**Graph Structure**:
- **Format**: `torch.Tensor` of shape `[2, num_edges]`
- **Type**: `torch.long`
- **Content**: Edge indices in COO format
- **Storage**: Stored in `data.edge_index` attribute

**Dataset Size**:
- **Train**: ~32,000 graphs
- **Val**: ~4,000 graphs
- **Test**: ~4,000 graphs

**Storage Format**:
- **Location**: `{root}/ogbg_molhiv/`
- **Format**: OGB's native format (processed by PyTorch Geometric)
- **Download**: Automatic on first use via OGB
- **Size on disk**: ~100-200 MB

**Splits**:
- **Type**: Predefined splits from OGB (scaffold split)
- **Train/Val/Test**: Automatically extracted via `get_idx_split()`

**Typical Graph Statistics**:
- **Average nodes per graph**: 25-30
- **Average edges per graph**: 50-60
- **Node count range**: 5-100 nodes
- **Edge count range**: 10-200 edges

**Usage Example**:
```python
from src.dataset import get_dataset

dataset = get_dataset(
    name="ogbg-molhiv",
    root="./data",
    pe_config={
        "node": {"enabled": True, "type": "rwse", "dim": 32},
        "relative": {"enabled": True, "type": "spd", "num_buckets": 32}
    }
)

train_data, val_data, test_data = dataset.get_splits()
print(f"Num classes: {dataset.num_classes}")  # 1 (binary classification)
```

---

### ogbg-molpcba

**Description**: ogbg-molpcba is a molecular property prediction dataset from OGB, containing molecular graphs with multiple binary property labels (multi-task classification).

**Data Source**: Open Graph Benchmark (OGB)

**Task Type**: Graph-level multi-label classification

**Target**:
- **Format**: `torch.Tensor` of shape `[128]` (128 binary labels)
- **Type**: `torch.float32` or `torch.long`
- **Meaning**: 128 binary labels indicating presence/absence of different molecular properties
- **Content**: Binary vector of length 128
- **Note**: Some labels may be NaN (missing), which should be masked during training
- **Storage**: Stored in `data.y` attribute

**Node Features**:
- **Format**: `torch.Tensor` of shape `[num_nodes, num_features]`
- **Type**: `torch.float32`
- **Dimension**: 9 (one-hot encoding of atom types)
- **Content**: One-hot encoded atom types
- **Storage**: Stored in `data.x` attribute

**Edge Features**:
- **Format**: `torch.Tensor` of shape `[num_edges, edge_dim]` (optional)
- **Type**: `torch.float32`
- **Content**: Bond type encoding
- **Storage**: Stored in `data.edge_attr` attribute

**Graph Structure**:
- **Format**: `torch.Tensor` of shape `[2, num_edges]`
- **Type**: `torch.long`
- **Content**: Edge indices in COO format
- **Storage**: Stored in `data.edge_index` attribute

**Dataset Size**:
- **Train**: ~350,000 graphs
- **Val**: ~40,000 graphs
- **Test**: ~40,000 graphs

**Storage Format**:
- **Location**: `{root}/ogbg_molpcba/`
- **Format**: OGB's native format
- **Download**: Automatic on first use via OGB
- **Size on disk**: ~2-3 GB

**Splits**:
- **Type**: Predefined splits from OGB (scaffold split)
- **Train/Val/Test**: Automatically extracted

**Typical Graph Statistics**:
- **Average nodes per graph**: 25-30
- **Average edges per graph**: 50-60
- **Node count range**: 5-100 nodes
- **Edge count range**: 10-200 edges

**Usage Example**:
```python
from src.dataset import get_dataset

dataset = get_dataset(
    name="ogbg-molpcba",
    root="./data",
    pe_config={
        "node": {"enabled": True, "type": "rwse", "dim": 32},
        "relative": {"enabled": True, "type": "spd", "num_buckets": 32}
    }
)

train_data, val_data, test_data = dataset.get_splits()
print(f"Num classes: {dataset.num_classes}")  # 128 (multi-label)
```

---

### PCQM4M

**Description**: PCQM4M is a large-scale molecular property prediction dataset from OGB, containing 4 million molecular graphs with quantum-mechanical properties.

**Data Source**: Open Graph Benchmark Large-Scale Challenge (OGB-LSC)

**Task Type**: Graph-level regression

**Target**:
- **Format**: `torch.Tensor` of shape `[1]` (scalar value)
- **Type**: `torch.float32`
- **Meaning**: Quantum-mechanical property (HOMO-LUMO gap or similar)
- **Content**: Continuous scalar value
- **Storage**: Stored in `data.y` attribute

**Node Features**:
- **Format**: `torch.Tensor` of shape `[num_nodes, num_features]`
- **Type**: `torch.float32`
- **Dimension**: 9 (one-hot encoding of atom types)
- **Content**: One-hot encoded atom types
- **Storage**: Stored in `data.x` attribute

**Edge Features**:
- **Format**: `torch.Tensor` of shape `[num_edges, edge_dim]` (optional)
- **Type**: `torch.float32`
- **Content**: Bond type encoding
- **Storage**: Stored in `data.edge_attr` attribute

**Graph Structure**:
- **Format**: `torch.Tensor` of shape `[2, num_edges]`
- **Type**: `torch.long`
- **Content**: Edge indices in COO format
- **Storage**: Stored in `data.edge_index` attribute

**Dataset Size**:
- **Train**: ~3,000,000 graphs
- **Val**: ~400,000 graphs
- **Test-dev**: ~400,000 graphs
- **Test-challenge**: ~400,000 graphs (for competition)

**Storage Format**:
- **Location**: `{root}/pcqm4m/`
- **Format**: OGB-LSC's native format
- **Download**: Automatic on first use via OGB-LSC
- **Size on disk**: ~50-100 GB (very large dataset)

**Splits**:
- **Type**: Predefined splits from OGB-LSC
- **Train/Val/Test-dev**: Automatically extracted via `get_idx_split()`

**Typical Graph Statistics**:
- **Average nodes per graph**: 20-30
- **Average edges per graph**: 40-60
- **Node count range**: 5-50 nodes
- **Edge count range**: 10-100 edges

**Usage Example**:
```python
from src.dataset import get_dataset

# Use subset for testing (first 10k samples)
dataset = get_dataset(
    name="pcqm4m",
    root="./data",
    subset=10000,  # Use only first 10k for testing
    pe_config={
        "node": {"enabled": True, "type": "rwse", "dim": 32},
        "relative": {"enabled": True, "type": "spd", "num_buckets": 32}
    }
)

train_data, val_data, test_data = dataset.get_splits()
```

---

### PCQM-Contact

**Description**: PCQM-Contact is an edge prediction dataset from OGB-LSC, where the task is to predict whether two atoms in a molecule are in contact (within a certain distance threshold).

**Data Source**: Open Graph Benchmark Large-Scale Challenge (OGB-LSC)

**Task Type**: Edge-level binary classification (link prediction)

**Target**:
- **Format**: `torch.Tensor` of shape `[num_edges]` or `[num_pairs]` (depending on representation)
- **Type**: `torch.float32` or `torch.long`
- **Meaning**: Binary labels indicating whether each edge/pair represents a contact
- **Content**: Binary values (0 = no contact, 1 = contact)
- **Storage**: Stored in `data.y` attribute or `data.edge_y` attribute

**Node Features**:
- **Format**: `torch.Tensor` of shape `[num_nodes, num_features]`
- **Type**: `torch.float32`
- **Dimension**: 9 (one-hot encoding of atom types)
- **Content**: One-hot encoded atom types
- **Storage**: Stored in `data.x` attribute

**Edge Features**:
- **Format**: `torch.Tensor` of shape `[num_edges, edge_dim]` (optional)
- **Type**: `torch.float32`
- **Content**: Bond type encoding
- **Storage**: Stored in `data.edge_attr` attribute

**Graph Structure**:
- **Format**: `torch.Tensor` of shape `[2, num_edges]`
- **Type**: `torch.long`
- **Content**: Edge indices in COO format
- **Storage**: Stored in `data.edge_index` attribute

**Dataset Size**:
- **Train**: ~3,000,000 graphs
- **Val**: ~400,000 graphs
- **Test**: ~400,000 graphs

**Storage Format**:
- **Location**: `{root}/pcqm_contact/`
- **Format**: OGB-LSC's native format
- **Download**: Automatic on first use via OGB-LSC
- **Size on disk**: ~50-100 GB

**Splits**:
- **Type**: Predefined splits from OGB-LSC
- **Train/Val/Test**: Automatically extracted

**Typical Graph Statistics**:
- **Average nodes per graph**: 20-30
- **Average edges per graph**: 40-60 (existing edges)
- **Node count range**: 5-50 nodes
- **Edge count range**: 10-100 edges

**Usage Example**:
```python
from src.dataset import get_dataset

dataset = get_dataset(
    name="pcqm-contact",
    root="./data",
    subset=10000,  # Use subset for testing
    pe_config={
        "node": {"enabled": True, "type": "rwse", "dim": 32},
        "relative": {"enabled": True, "type": "spd", "num_buckets": 32}
    }
)

train_data, val_data, test_data = dataset.get_splits()
```

---

## Synthetic Datasets

Synthetic datasets are generated on-the-fly using NetworkX and are designed for controlled evaluation of positional encodings under various graph structures and tasks.

### Graph Generators

All synthetic datasets support the following graph generation types:

#### Erdős–Rényi (`synthetic_erdos_renyi`)

**Description**: Random graphs where each edge exists with probability p.

**Parameters**:
- `n`: Number of nodes (default: 20)
- `p`: Edge probability (default: 0.3)

**Graph Properties**:
- **Average degree**: `n * p`
- **Expected edges**: `n * (n-1) / 2 * p`
- **Connectivity**: May be disconnected for small p
- **Structure**: Random, no inherent structure

**Storage**: Generated on-the-fly, not stored on disk

---

#### Watts–Strogatz (`synthetic_watts_strogatz`)

**Description**: Small-world graphs with high clustering and short path lengths.

**Parameters**:
- `n`: Number of nodes (default: 20)
- `k`: Each node connected to k nearest neighbors (default: 4)
- `p`: Rewiring probability (default: 0.3)

**Graph Properties**:
- **Average degree**: `k`
- **Structure**: Regular ring lattice with random rewiring
- **Properties**: High clustering coefficient, short average path length

**Storage**: Generated on-the-fly

---

#### Barabási–Albert (`synthetic_barabasi_albert`)

**Description**: Scale-free networks with power-law degree distribution.

**Parameters**:
- `n`: Number of nodes (default: 20)
- `m`: Number of edges to attach from new node (default: 2)

**Graph Properties**:
- **Average degree**: `2 * m` (approximately)
- **Structure**: Preferential attachment, hub nodes
- **Properties**: Power-law degree distribution, few highly connected hubs

**Storage**: Generated on-the-fly

---

#### Stochastic Block Model (`synthetic_sbm`)

**Description**: Graphs with community structure (blocks).

**Parameters**:
- `n`: Number of nodes (default: 50)
- `n_blocks`: Number of communities (default: 3)
- `block_sizes`: List of block sizes (default: equal-sized blocks)
- `p_in`: Within-block edge probability (default: 0.3)
- `p_out`: Between-block edge probability (default: 0.05)

**Graph Properties**:
- **Structure**: Clear community structure
- **Properties**: Dense within communities, sparse between communities
- **Node attributes**: Block membership stored in `data.block` (if available)

**Storage**: Generated on-the-fly

---

#### Random Geometric (`synthetic_random_geometric`)

**Description**: Spatial graphs where nodes are connected if within a certain distance.

**Parameters**:
- `n`: Number of nodes (default: 50)
- `radius`: Connection radius (default: 0.2)
- `dim`: Dimensionality of space (default: 2)

**Graph Properties**:
- **Structure**: Spatial, distance-based connections
- **Node positions**: Stored in `data.pos` attribute `[num_nodes, dim]`
- **Properties**: Reflects spatial proximity

**Storage**: Generated on-the-fly

---

#### Random Regular (`synthetic_random_regular`)

**Description**: Regular graphs where all nodes have the same degree.

**Parameters**:
- `n`: Number of nodes (default: 20)
- `d`: Degree of each node (default: 3)

**Graph Properties**:
- **Structure**: All nodes have degree d
- **Properties**: Highly symmetric, no degree variation

**Storage**: Generated on-the-fly

---

#### Grid (`synthetic_grid_2d`, `synthetic_grid_3d`)

**Description**: Regular grid graphs (2D or 3D).

**Parameters**:
- `m`, `n`: Grid dimensions (2D: default 10x10, 3D: default 5x5x5)
- `k`: Third dimension for 3D (default: 5)

**Graph Properties**:
- **Structure**: Regular grid lattice
- **Properties**: Highly structured, regular connectivity

**Storage**: Generated on-the-fly

---

#### Ring (`synthetic_ring`)

**Description**: Cycle graphs (ring topology).

**Parameters**:
- `n`: Number of nodes (default: 20)

**Graph Properties**:
- **Structure**: Single cycle, all nodes have degree 2
- **Properties**: Highly symmetric, regular

**Storage**: Generated on-the-fly

---

#### Tree (`synthetic_tree`)

**Description**: Balanced tree graphs.

**Parameters**:
- `r`: Branching factor (default: 3)
- `h`: Height of tree (default: 4)

**Graph Properties**:
- **Structure**: Tree structure (no cycles)
- **Properties**: Hierarchical, regular branching

**Storage**: Generated on-the-fly

---

### Benchmark Tasks

Synthetic datasets support various benchmark tasks for evaluating positional encodings:

#### Task A: Pairwise Distance Classification

**Description**: Classify whether the shortest-path distance between two nodes is >= threshold.

**Task Type**: Graph-level binary classification (or node-pair classification)

**Target**:
- **Format**: `torch.Tensor` of shape `[1]` (binary label)
- **Type**: `torch.long` (0 or 1)
- **Meaning**: 1 if distance >= threshold, 0 otherwise
- **Storage**: Stored in `data.y` attribute

**Additional Metadata**:
- `data.pair_sources`: Source node indices `[num_pairs]`
- `data.pair_targets`: Target node indices `[num_pairs]`
- `data.pair_labels`: Distance labels `[num_pairs]`

**Parameters**:
- `distance_threshold`: Distance threshold (default: 3)
- `num_pairs`: Number of node pairs per graph (default: 100)

**Usage**:
```python
dataset = get_dataset(
    name="synthetic_erdos_renyi",
    task="pairwise_distance_classification",
    task_params={"distance_threshold": 3, "num_pairs": 100},
    graph_params={"n": 50, "p": 0.3},
    num_graphs=1000,
    task_type="classification",
    num_classes=2,
    seed=42
)
```

---

#### Task B: Distance Regression

**Description**: Predict the exact shortest-path distance between node pairs.

**Task Type**: Graph-level regression (or node-pair regression)

**Target**:
- **Format**: `torch.Tensor` of shape `[1]` (distance value)
- **Type**: `torch.float32`
- **Meaning**: Exact shortest-path distance
- **Storage**: Stored in `data.y` attribute

**Additional Metadata**:
- `data.pair_sources`: Source node indices
- `data.pair_targets`: Target node indices
- `data.pair_labels`: Distance labels

**Parameters**:
- `num_pairs`: Number of node pairs per graph (default: 100)

**Usage**:
```python
dataset = get_dataset(
    name="synthetic_erdos_renyi",
    task="distance_regression",
    task_params={"num_pairs": 100},
    graph_params={"n": 50, "p": 0.3},
    num_graphs=1000,
    task_type="regression",
    num_classes=1,
    seed=42
)
```

---

#### Task C: Structural Role Classification

**Description**: Classify nodes by their structural role (e.g., block ID in SBM).

**Task Type**: Node-level multi-class classification

**Target**:
- **Format**: `torch.Tensor` of shape `[num_nodes]` (one label per node)
- **Type**: `torch.long`
- **Meaning**: Structural role/block ID for each node
- **Storage**: Stored in `data.y` attribute

**Parameters**:
- `use_node_features`: Set to `False` for structure-only task

**Usage**:
```python
dataset = get_dataset(
    name="synthetic_sbm",
    task="structural_role",
    graph_params={"n": 100, "n_blocks": 3, "p_in": 0.3, "p_out": 0.05},
    num_graphs=1000,
    task_type="classification",
    num_classes=3,  # Number of blocks
    use_node_features=False,  # Structure-only
    seed=42
)
```

---

#### Task D: Local vs Global Signal

**Description**: Predict based on local (degree) or global (distance) signals.

**Task Type**: Graph-level or node-level classification/regression

**Target**:
- **Format**: `torch.Tensor` of shape `[1]` or `[num_nodes]`
- **Type**: `torch.float32` or `torch.long`
- **Meaning**: Label based on local degree or global distance
- **Storage**: Stored in `data.y` attribute

**Parameters**:
- `use_local`: If True, use local signal (degree), else global (distance)
- `local_threshold`: Threshold for local signal (default: 3)

**Usage**:
```python
dataset = get_dataset(
    name="synthetic_watts_strogatz",
    task="local_vs_global",
    task_params={"use_local": True, "local_threshold": 3},
    graph_params={"n": 50, "k": 4, "p": 0.3},
    num_graphs=1000,
    task_type="classification",
    num_classes=2,
    seed=42
)
```

---

#### Task E: Diffusion Source Identification

**Description**: Identify the source node of a diffusion process.

**Task Type**: Node-level multi-class classification

**Target**:
- **Format**: `torch.Tensor` of shape `[num_nodes]` (one-hot or index)
- **Type**: `torch.long`
- **Meaning**: Source node ID (one node is the source, others are not)
- **Storage**: Stored in `data.y` attribute

**Parameters**:
- `diffusion_steps`: Number of diffusion steps (default: 5)
- `diffusion_rate`: Diffusion rate (default: 0.5)

**Usage**:
```python
dataset = get_dataset(
    name="synthetic_erdos_renyi",
    task="diffusion_source",
    task_params={"diffusion_steps": 5, "diffusion_rate": 0.5},
    graph_params={"n": 30, "p": 0.3},
    num_graphs=1000,
    task_type="classification",
    num_classes=30,  # Number of nodes (one class per node)
    seed=42
)
```

---

#### Task F: PE Capacity Stress Test

**Description**: Test PE capacity by varying dimensionality (same as Task A or C, but with PE dimension sweep).

**Task Type**: Same as base task (A or C)

**Target**: Same as base task

**Parameters**:
- `base_task`: Base task to use ("pairwise_distance_classification" or "structural_role")

**Usage**:
```python
dataset = get_dataset(
    name="synthetic_erdos_renyi",
    task="pe_capacity",
    task_params={"base_task": "pairwise_distance_classification"},
    graph_params={"n": 50, "p": 0.3},
    num_graphs=1000,
    task_type="classification",
    num_classes=2,
    seed=42
)
```

---

### Synthetic Dataset Storage

**Storage Format**:
- **Location**: Not stored on disk (generated on-the-fly)
- **Format**: In-memory PyTorch Geometric `Data` objects
- **Size**: Depends on `num_graphs` and graph size
- **Memory usage**: Approximately `num_graphs * avg_nodes * (feature_dim + pe_dim) * 4 bytes`

**Reproducibility**:
- **Seed**: Controlled via `seed` parameter
- **Determinism**: Same `seed` + `graph_params` + `num_graphs` → identical graphs
- **Graph generation**: Uses `np.random.seed(seed + graph_index)` for each graph

**Splits**:
- **Type**: Fixed split (80% train, 10% val, 10% test)
- **Train**: First 80% of generated graphs
- **Val**: Next 10% of generated graphs
- **Test**: Last 10% of generated graphs

**Node Features** (if `use_node_features=True`):
- **Format**: `torch.Tensor` of shape `[num_nodes, 1]`
- **Type**: `torch.float32`
- **Content**: Node degree (normalized or raw)
- **Storage**: Stored in `data.x` attribute

**Node Features** (if `use_node_features=False`):
- **Format**: `torch.Tensor` of shape `[num_nodes, 1]`
- **Type**: `torch.float32`
- **Content**: Constant value (1.0) - structure-only
- **Storage**: Stored in `data.x` attribute

---

## Common Data Structure

All datasets use PyTorch Geometric's `Data` class with the following common structure:

```python
Data(
    # Node features
    x=torch.Tensor,              # [num_nodes, num_features]
    
    # Graph structure
    edge_index=torch.Tensor,      # [2, num_edges] (COO format)
    edge_attr=torch.Tensor,       # [num_edges, edge_dim] (optional)
    
    # Positional encodings (added by transforms)
    node_pe=torch.Tensor,         # [num_nodes, node_pe_dim] (optional)
    edge_pe=torch.Tensor,         # [num_edges, relative_pe_dim] (optional)
    edge_pe_index=torch.Tensor,  # [2, num_pairs] for pairwise PE (optional)
    
    # Target
    y=torch.Tensor,               # [num_targets] or [1] or [num_nodes]
    
    # Additional attributes (dataset-specific)
    pos=torch.Tensor,             # [num_nodes, 2/3] (spatial positions, optional)
    split=str,                    # "train"/"val"/"test" (some datasets)
    block=torch.Tensor,           # [num_nodes] (block membership, SBM)
    pair_sources=torch.Tensor,    # [num_pairs] (pairwise tasks)
    pair_targets=torch.Tensor,    # [num_pairs] (pairwise tasks)
    pair_labels=torch.Tensor,     # [num_pairs] (pairwise tasks)
)
```

---

## Preprocessing and Caching

### PE Preprocessing

Positional encodings can be precomputed and cached to speed up training:

**Cache Location**: `{pe_cache_dir}/{dataset_name}/` or `{root}/pe_cache/{dataset_name}/`

**Cache Files**:
- `train.pt`: Preprocessed training graphs
- `val.pt`: Preprocessed validation graphs
- `test.pt`: Preprocessed test graphs
- `pe_config.yaml`: PE configuration used for preprocessing

**Usage**:
```bash
# Precompute PE
python scripts/preprocess_pe.py dataset=zinc pe=mspe

# Use precomputed PE (automatic if cache exists)
python scripts/train.py dataset=zinc pe=mspe use_preprocessed_pe=true
```

**Cache Format**: List of PyTorch Geometric `Data` objects saved via `torch.save()`

**Cache Size**: Similar to original dataset size, plus PE dimensions

---

## Dataset Statistics Summary

| Dataset | Type | Task | Nodes (avg) | Edges (avg) | Size | Target Dim |
|---------|------|------|-------------|-------------|------|------------|
| ZINC | Molecular | Regression | 20-30 | 40-60 | ~100 MB | 1 |
| QM9 | Molecular | Regression | 18 | 18-19 | ~2-3 GB | 1 |
| Peptides-func | LRGB | Multi-label | 150-200 | 300-400 | ~200 MB | 10 |
| Peptides-struct | LRGB | Multi-class | 150-200 | 300-400 | ~200 MB | 11 |
| PascalVOC-SP | LRGB | Multi-class | 200-300 | 500-800 | ~1 GB | 21 |
| CIFAR10-SP | LRGB | Multi-class | 100-200 | 300-500 | ~2 GB | 10 |
| ogbg-molhiv | OGB | Binary | 25-30 | 50-60 | ~200 MB | 1 |
| ogbg-molpcba | OGB | Multi-label | 25-30 | 50-60 | ~3 GB | 128 |
| PCQM4M | OGB-LSC | Regression | 20-30 | 40-60 | ~100 GB | 1 |
| PCQM-Contact | OGB-LSC | Edge pred | 20-30 | 40-60 | ~100 GB | 1 |
| Synthetic | Generated | Various | Configurable | Configurable | In-memory | Configurable |

---

## Notes

1. **Download**: Most datasets download automatically on first use. Ensure sufficient disk space.
2. **Memory**: Large datasets (PCQM4M, ogbg-molpcba) may require significant RAM.
3. **PE Computation**: PE computation can be time-consuming for large graphs. Use preprocessing for faster iteration.
4. **Reproducibility**: All datasets support seed-based reproducibility (synthetic datasets are fully deterministic).
5. **Normalization**: Some datasets (QM9) support target normalization. Check dataset-specific parameters.
6. **Missing Values**: Some datasets (ogbg-molpcba) may have missing labels (NaN), which should be masked during training.

