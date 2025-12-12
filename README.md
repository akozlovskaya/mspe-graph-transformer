# mspe-graph-transformer

A unified framework for multi-scale structural positional encodings (PEs) in Graph Transformers.
Includes node-wise spectral/diffusion encodings (LapPE, RWSE, HKS), relative pairwise biases (SPD, diffusion, effective resistance), a GraphGPS-style Transformer architecture, dataset loaders, PE precomputation tools, long-range evaluation, and reproducible training pipelines.

---

## 🚀 Features

### **Multi-Scale Node Positional Encodings**

* Laplacian Positional Encodings (LapPE) with sign-invariant processing
* Random-Walk Structural Encodings (RWSE)
* Heat Kernel Signatures (HKS)
* Role/structural features (degree, clustering, coreness)

### **Relative Structural Encodings**

* Shortest-path distance (SPD) buckets
* Diffusion / heat-kernel structural distances
* Approximate effective resistance (low-rank pseudoinverse)
* Landmark and truncated-SPD approximations
* Sparse attention bias integration

### **GraphGPS-Style Hybrid Transformer**

* Local MPNN + global attention
* Residual/gated PE mixing
* Scalable multi-head attention with structural biases

### **Dataset & Precomputation Pipeline**

Supports:
ZINC, QM9, LRGB Peptides, PCQM-Contact, OGB mol-series, superpixel vision graphs, synthetic long-range graphs, and more.
Includes PE precomputation, caching, and normalization.

### **Long-Range Evaluation Tools**

* Distance-binned metrics
* Occlusion-by-radius analysis
* Counterfactual edge interventions
* Stability tests for deep GTs (12–16 layers)

### **Efficiency Profiling**

* Low-rank spectral approximations
* VRAM/throughput evaluation
* Quality–compute trade-off curves

---

## 📦 Installation

```bash
git clone https://github.com/<your-username>/mspe-graph-transformer.git
cd mspe-graph-transformer
pip install -r requirements.txt
```

---

## 📁 Repository Structure

```
mspe-graph-transformer/
    configs/
    data/
    src/
        dataset/
        pe/
            node/
            relative/
        models/
            transformer/
            mpnn/
        training/
        evaluation/
        utils/
    scripts/
    notebooks/
    tests/
    README.md
```

---

## ▶️ Quick Start

### Train a model

```bash
python scripts/train.py dataset=zinc model=graphgps pe=mspe
```

### Evaluate

```bash
python scripts/evaluate.py checkpoint=path/to/ckpt
```

### Precompute PEs

```bash
python scripts/preprocess_pe.py dataset=peptides_struct pe=mspe
```

---

## 🧪 Supported Datasets

* **Molecular:** ZINC, QM9, OGB-MOLHIV, OGB-MOLPCBA
* **Long-range:** LRGB Peptides-func/struct, PCQM-Contact
* **Vision-superpixels:** PascalVOC-SP, CIFAR10-SP
* **Synthetic:** grids, rings, trees, small-world, random regular
* **Other:** code graphs, knowledge graphs (optional)

---

## 📚 Citation

If you use this repository in research, please cite the corresponding thesis or project.

---

## 📄 License

MIT License.

