import os
import sys
import warnings

# Add project root to sys.path (same pattern as train.py)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import torch
from torch import nn
from torch.optim import Adam

from models.dense_snn import DenseSNN
from models.mixer_snn import MixerSNN
from models.er_snn import ERSNN
from data.data_fashionmnist import get_fashion_loaders
from utils.encoding import rate_encode

warnings.filterwarnings(
    "ignore",
    message=".*aten::lerp.Scalar_out.*"
)

# Try to match the device selection logic from train.py (DirectML / CUDA / CPU)
try:
    import torch_directml
    HAS_DML = True
except ImportError:
    HAS_DML = False


def select_device() -> torch.device:
    """Select compute device following the same priority as train.py."""
    if HAS_DML:
        device = torch_directml.device()
        print(f"Using DirectML device: {device}")
        return device

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        print(f"Using GPU: {gpu_name} | CUDA version: {cuda_version}")
        return device

    device = torch.device("cpu")
    print("No GPU backend available — using CPU.")
    return device


# -------------------------------------------------------------------
# Hyperparameters – kept in sync with train.py
# -------------------------------------------------------------------

batch_size = 256
T = 50
input_dim = 28 * 28
hidden_dim = 1024          # same width as G2GNet sparse models
hidden_dim_dense = 447     # chosen to roughly match the parameter count
num_classes = 10
num_epochs = 20
lr = 1e-3
weight_decay = 1e-4

# Mixer (G2GNet) connectivity parameters
num_groups = 8
p_intra = 1.0
p_inter = 0.15

# ER random density; can be tuned if you want closer param matching
p_er_active = 0.20


def count_params(model: nn.Module) -> int:
    """Return the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters())


def train_one_epoch(model: nn.Module,
                    loader,
                    optimizer: Adam,
                    device: torch.device) -> float:
    """
    Single training epoch.

    This mirrors the logic used in train.py:
    - encode images into spike trains with rate encoding
    - accumulate spike counts at the output
    - standard cross-entropy loss on spike counts
    """
    model.train()
    total = 0
    correct = 0

    for images, labels in loader:
        labels = labels.to(device, non_blocking=True)

        # [T, B, input_dim]
        spikes = rate_encode(images, T).to(device)

        optimizer.zero_grad()
        logits = model(spikes)

        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total


@torch.no_grad()
def evaluate(model: nn.Module,
             loader,
             device: torch.device) -> float:
    """
    Evaluation loop, identical in spirit to the one in train.py.
    """
    model.eval()
    total = 0
    correct = 0

    for images, labels in loader:
        labels = labels.to(device, non_blocking=True)
        spikes = rate_encode(images, T).to(device)
        logits = model(spikes)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total


def build_table1_models():
    """
    Build the four SNN models that mirror the architectures in Table I:

      1) Fully-Connected v1  – dense MLP with roughly the same number
                               of parameters as the sparse G2GNet model.
      2) Fully-Connected v2  – dense MLP with the same width (1024 units).
      3) ER Random Graph     – unstructured Erdos–Rényi sparse baseline.
      4) G2GNet (Proposed)   – Mixer-style grouping (V1-inspired topology).
    """
    models_cfg = [
        {
            "id": "fc_v1",
            "label": "Fully-Connected v1 (same #params)",
            "builder": lambda: DenseSNN(input_dim, hidden_dim_dense, num_classes),
        },
        {
            "id": "fc_v2",
            "label": "Fully-Connected v2 (width=1024)",
            "builder": lambda: DenseSNN(input_dim, hidden_dim, num_classes),
        },
        {
            "id": "er",
            "label": "ER Random Graph",
            "builder": lambda: ERSNN(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                p_active=p_er_active,
            ),
        },
        {
            "id": "g2g_mixer",
            "label": "G2GNet (Proposed, Mixer)",
            "builder": lambda: MixerSNN(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                num_groups=num_groups,
                p_intra=p_intra,
                p_inter=p_inter,
            ),
        },
    ]
    return models_cfg


def main():
    device = select_device()

    # Same data pipeline as train.py: Fashion-MNIST with ToTensor()
    train_loader, test_loader = get_fashion_loaders(batch_size)

    models_cfg = build_table1_models()
    results = []

    for cfg in models_cfg:
        print("\n" + "=" * 70)
        print(f"Training model: {cfg['label']}  (id={cfg['id']})")
        print("=" * 70)

        model = cfg["builder"]().to(device)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        num_params = count_params(model)
        final_test_acc = None

        for epoch in range(1, num_epochs + 1):
            train_acc = train_one_epoch(model, train_loader, optimizer, device)
            test_acc = evaluate(model, test_loader, device)
            final_test_acc = test_acc

            print(
                f"[{cfg['id']}] Epoch {epoch:02d} | "
                f"train_acc={train_acc:.4f} | test_acc={test_acc:.4f}"
            )

        results.append(
            {
                "id": cfg["id"],
                "label": cfg["label"],
                "test_acc": final_test_acc,
                "params": num_params,
            }
        )

    # Final SNN Table I-style summary for Fashion-MNIST
    print("\n" + "#" * 80)
    print("SNN Table I-style results on Fashion-MNIST")
    print("#" * 80)
    header = f"{'Connectivity Pattern':35s} | {'Test Acc (%)':12s} | {'#Params':>10s}"
    print(header)
    print("-" * len(header))

    for r in results:
        acc_percent = 100.0 * r["test_acc"]
        line = f"{r['label']:35s} | {acc_percent:12.2f} | {r['params']:10d}"
        print(line)


if __name__ == "__main__":
    main()
