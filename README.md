# G2G_Sparse_SNN

Spiking Neural Network implementations on Fashion-MNIST with different connectivity patterns:

- **Dense**: standard fully-connected baseline
- **Index**: group-based sparse connectivity (G2G approach)
- **Random**: randomly sparse connections
- **Mixer**: mixer-based sparse connectivity with intra/inter-group structure

You can train these models normally or run k-fold cross-validation on the Fashion-MNIST training set.

---

## Installation

I'd recommend setting up a virtual environment first:
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

### Core dependencies

Install everything from requirements.txt:
```bash
pip install -r requirements.txt
```

Or just install the essentials manually:
```bash
pip install snntorch torchvision numpy
```

### GPU support

**NVIDIA (CUDA)**

Grab the right CUDA wheel from PyTorch's site, something like:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

(Pick cu121, cu128, etc. depending on what you have installed)

**AMD GPU (Windows, DirectML)**

If you're on Windows with an AMD GPU:
```bash
pip install torch-directml
```

The code will try to use DirectML if it's available, otherwise CUDA, otherwise CPU.

## Training

Run the main training script:
```bash
python evaluation/train.py --model X [--p_inter P] [--epochs N]
```

where X is one of: `dense`, `index`, `random`, `mixer`

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Model type: dense, index, random, mixer | required |
| `--p_inter` | Inter-group probability for sparse models | 0.15 |
| `--epochs` | Training epochs | 20 |

Note: `p_inter` only matters for sparse models (index, random, mixer) - it's ignored for dense.

### Examples

Train index-based sparse model:
```bash
python evaluation/train.py --model index
```

Train mixer with higher inter-group connectivity:
```bash
python evaluation/train.py --model mixer --p_inter 0.20
```

Train random sparse model with more epochs:
```bash
python evaluation/train.py --model random --epochs 30 --p_inter 0.10
```

Dense baseline:
```bash
python evaluation/train.py --model dense --epochs 20
```

## Cross-validation

The `evaluation/crossval.py` script runs k-fold CV on the training set. For each fold it:

- Builds a fresh model
- Trains on (k-1)/k of the data
- Validates on the remaining 1/k
- Records train/val accuracy

At the end you get mean ± std across all folds.

### Usage
```bash
python evaluation/crossval.py --model X [--p_inter P] [--epochs N] [--k_folds K]
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Model type | dense |
| `--p_inter` | Inter-group probability (sparse models only) | 0.15 |
| `--epochs` | Epochs per fold | 5 |
| `--k_folds` | Number of folds | 5 |

### Examples

5-fold CV for dense model:
```bash
python evaluation/crossval.py --model dense --epochs 5 --k_folds 5
```

5-fold CV for index model:
```bash
python evaluation/crossval.py --model index --p_inter 0.15 --epochs 5 --k_folds 5
```

Compare different inter-group probabilities:
```bash
python evaluation/crossval.py --model mixer --p_inter 0.20 --epochs 10 --k_folds 5
```

The output shows train/val accuracy for each fold, plus overall mean ± std. Useful for checking how sensitive each architecture is to the train/val split and how `p_inter` affects performance.

## Reproducibility

Both scripts use `set_seed(42)` to keep things reproducible across:
- Python random
- NumPy
- PyTorch (CPU & CUDA)
- cuDNN (deterministic mode)

If you want a different seed, just modify it in the code or add a `--seed` argument.

## Dataset

Everything runs on Fashion-MNIST:
- 28×28 grayscale images
- 10 classes (T-shirt, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, ankle boot)

We use the training set for both standard training and k-fold cross-validation.