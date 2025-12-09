
# G2G_Sparse_SNN

Spiking Neural Network implementations on Fashion-MNIST exploring different connectivity patterns inspired by biological neural circuits:

- **Dense**: fully-connected baseline for comparison  
- **Index**: G2GNet with index-based grouping (preserves spatial locality)  
- **Random**: G2GNet with random grouping (disrupts spatial structure)  
- **Mixer**: G2GNet with mixer-based grouping (alternates between spatial and feature mixing)

G2GNet is our proposed architecture that uses sparse, modular connectivity inspired by ensemble-to-ensemble communication observed in mouse visual cortex. The three grouping strategies (Index, Random, Mixer) represent different ways to organize neurons within each layer.

You can train these models normally, enable **Dynamic Sparse Training (DST)** to update sparse connectivity during training, or run **k-fold cross-validation** on the Fashion-MNIST training set.

---

## Installation

I'd recommend setting up a virtual environment first:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

### Core dependencies

```bash
pip install -r requirements.txt
```

Or alternatively:

```bash
pip install snntorch torchvision numpy
```

### GPU support

**NVIDIA (CUDA)**  
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**AMD GPU (Windows, DirectML)**  
```bash
pip install torch-directml
```

The code will try to use DirectML if available; otherwise CUDA, otherwise CPU.

---

## Training

Run the main training script:

```bash
python evaluation/train.py --model X [--p_inter P] [--epochs N] [--sparsity_mode M] [--cp CP] [--cg CG]
```

where **X ∈ {dense, index, random, mixer}**

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Model type: `dense`, `index`, `random`, `mixer` | required |
| `--p_inter` | Inter-group connection probability (only sparse models) | `0.15` |
| `--epochs` | Training epochs | `20` |
| `--sparsity_mode` | `static` = fixed sparsity, `dynamic` = enable DST | `static` |
| `--cp` | Pruning rule (C_P): `three` = SET + Random + Hebbian, `set` = magnitude pruning only | `three` |
| `--cg` | Growth rule (C_G): `hebb` = based on correlation (CH), `random` = random growth | `hebb` |

> **DST options (`--sparsity_mode`, `--cp`, `--cg`) are only active when using sparse models**, mainly implemented for the **mixer** configuration.

---

## Example Commands

### Standard training (without DST)

```bash
python evaluation/train.py --model index
python evaluation/train.py --model mixer --p_inter 0.20
python evaluation/train.py --model random --epochs 30 --p_inter 0.10
python evaluation/train.py --model dense --epochs 20
```

### Enable Dynamic Sparse Training (DST)

**Paper-style setup (recommended):**
```bash
python evaluation/train.py   --model mixer   --sparsity_mode dynamic   --cp three   --cg hebb   --epochs 10
```

**Magnitude-only pruning + Hebbian growth:**
```bash
python evaluation/train.py   --model mixer   --sparsity_mode dynamic   --cp set   --cg hebb   --epochs 10
```

**DST with random growth:**
```bash
python evaluation/train.py   --model mixer   --sparsity_mode dynamic   --cp set   --cg random   --epochs 10
```

---

## Cross-validation

```bash
python evaluation/crossval.py --model X [--p_inter P] [--epochs N] [--k_folds K]
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Model type | `dense` |
| `--p_inter` | Inter-group connectivity (sparse models only) | `0.15` |
| `--epochs` | Epochs per fold | `5` |
| `--k_folds` | Number of folds | `5` |

---

## Reproducibility

- `set_seed(42)` used across Python, NumPy, PyTorch  
- Deterministic CUDA & cuDNN  
- Custom seed possible via argument or code edit

---

## Dataset

Fashion-MNIST:

| Property | Value |
|----------|-------|
| Resolution | 28×28 |
| Channels | Grayscale |
| Classes | 10 |

---

## About G2GNet and DST

G2GNet is inspired by ensemble connectivity in mouse visual cortex. The DST mechanism dynamically reallocates sparse connections to improve performance:

- **Pruning ("C_P")** removes weak or unimportant weights  
- **Growth ("C_G")** allocates new connections based on neuron activity (Hebbian) or randomly  
- Connectivity adapts during training while remaining sparse

| Model | Sparse | Can use DST |
|-------|--------|-------------|
| Dense | ❌ | ❌ |
| Index | ✔ | static only |
| Random | ✔ | ❌ |
| Mixer | ✔ | ✔ **fully DST-ready** |

---

If you want to include biological motivation, CH similarity, or visual diagrams, let me know!

---