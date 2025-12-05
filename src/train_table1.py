# train_table1.py
#
# Reproduces a Table-I-style comparison on Fashion-MNIST
# using a CNN encoder + SNN classifier heads with different
# connectivity patterns (FC v1, FC v2, ER, G2G/Mixer).

import os
import sys
import warnings

# Add project root to sys.path (same pattern as train.py)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import torch
from torch import nn
from torch.optim import Adam

from models.dense_snn import DenseSNN
from models.mixer_snn import MixerSNN, MixerSparseLinear
from models.er_snn import ERSNN, ERSparseLinear
from models.cnn_encoder import PatchConvEncoder
from data.data_fashionmnist import get_fashion_loaders

warnings.filterwarnings(
    "ignore",
    message=".*aten::lerp.Scalar_out.*",
)

# ---------------------------------------------------------------------
# Device selection (same logic as in train.py)
# ---------------------------------------------------------------------
try:
    import torch_directml

    HAS_DML = True
except ImportError:
    HAS_DML = False


def select_device() -> torch.device:
    """Select compute device (DirectML, CUDA, or CPU)."""
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


# ---------------------------------------------------------------------
# Hyperparameters – aligned with the paper where possible
# ---------------------------------------------------------------------

batch_size = 256
T = 50  # number of time steps for the SNN
feature_dim = 512  # output of PatchConvEncoder
hidden_dim = 1024  # width of G2G / ER / FC v2
# FC v1: dense head with fewer units to roughly match the
# number of active parameters of the sparse models.
hidden_dim_dense = 512

num_classes = 10
num_epochs = 20
lr = 1e-3
weight_decay = 1e-4

# G2G / Mixer connectivity (fixed as in the paper)
num_groups = 8
p_intra = 1.0
p_inter = 0.15

# ER random graph: probability of keeping a connection.
# This can be tuned to better match the effective parameter count.
p_er_active = 0.18


# ---------------------------------------------------------------------
# CNN + SNN wrapper
# ---------------------------------------------------------------------
class CNNSNNWrapper(nn.Module):
    """
    Wraps a CNN feature extractor and an SNN classifier head.

    The CNN encoder processes the 28x28 Fashion-MNIST images and
    outputs a 512-dimensional feature vector. This vector is then
    replicated across T time steps and fed into the SNN head.
    """

    def __init__(self, head: nn.Module, T_steps: int = T):
        super().__init__()
        self.encoder = PatchConvEncoder(in_channels=1, out_channels=32)
        self.head = head
        self.T = T_steps

    def encode_to_sequence(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: [B, 1, 28, 28]
        returns: [T, B, feature_dim] – constant features over time.
        """
        # CNN features: [B, 512]
        feats = self.encoder(images)

        # Optional feature scaling (0–1 per batch) to keep inputs
        # in a reasonable range for the SNN.
        with torch.no_grad():
            # Avoid division by zero
            f_min = feats.min(dim=1, keepdim=True)[0]
            f_max = feats.max(dim=1, keepdim=True)[0]
            denom = (f_max - f_min).clamp(min=1e-6)
            feats_norm = (feats - f_min) / denom

        # Repeat across time steps: [T, B, 512]
        x_seq = feats_norm.unsqueeze(0).repeat(self.T, 1, 1)
        return x_seq

    def forward(self, images: torch.Tensor, return_hidden_spikes: bool = False):
        # images: [B, 1, 28, 28]
        x_seq = self.encode_to_sequence(images)
        if return_hidden_spikes:
            return self.head(x_seq, return_hidden_spikes=True)
        return self.head(x_seq)


# ---------------------------------------------------------------------
# Parameter counting (classifier head only, active edges for sparse)
# ---------------------------------------------------------------------
def count_head_params(head: nn.Module) -> int:
    """
    Count the number of parameters in the classifier head only.

    - For dense Linear layers: full weight + bias.
    - For ER / Mixer sparse layers: only active weights according to
      the 'mask' plus the bias vector.
    """
    total = 0
    for module in head.modules():
        # Skip the top-level module itself
        if module is head:
            continue

        # Mixer / G2G sparse layer
        if isinstance(module, MixerSparseLinear):
            mask = module.mask
            total += int(mask.sum().item())  # active weights
            if module.bias is not None:
                total += module.bias.numel()

        # ER sparse layer
        elif isinstance(module, ERSparseLinear):
            mask = module.mask
            total += int(mask.sum().item())
            if module.bias is not None:
                total += module.bias.numel()

        # Standard dense layer
        elif isinstance(module, nn.Linear):
            total += module.weight.numel()
            if module.bias is not None:
                total += module.bias.numel()

    return total


# ---------------------------------------------------------------------
# Training / evaluation loops (image -> CNN -> SNN)
# ---------------------------------------------------------------------
def train_one_epoch(
    model: CNNSNNWrapper,
    loader,
    optimizer: Adam,
    device: torch.device,
) -> float:
    """
    Single training epoch.

    Mirrors the logic of train.py:
    - Input images are converted to spike-like sequences via the CNN.
    - The SNN head integrates inputs over T time steps.
    - Cross-entropy loss is applied on the summed output spikes.
    """
    model.train()
    total = 0
    correct = 0

    criterion = nn.CrossEntropyLoss()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)  # [B, num_classes]
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total


@torch.no_grad()
def evaluate(
    model: CNNSNNWrapper,
    loader,
    device: torch.device,
) -> float:
    """Evaluation loop, same structure as train_one_epoch but without gradients."""
    model.eval()
    total = 0
    correct = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total


# ---------------------------------------------------------------------
# Build the four heads corresponding to Table I
# ---------------------------------------------------------------------
def build_heads():
    """
    Build the four SNN classifier heads that correspond to the
    architectures in Table I:

      1) Fully-Connected v1  – dense SNN with reduced width
                               (hidden_dim_dense) to match the number
                               of parameters of the sparse models.
      2) Fully-Connected v2  – dense SNN with width=1024.
      3) ER Random Graph     – ER sparse SNN with active edges matching
                               the G2G parameter budget (via p_er_active).
      4) G2GNet (Mixer)      – Mixer sparse SNN (V1-inspired topology).
    """
    heads_cfg = [
        {
            "id": "fc_v1",
            "label": "Fully-Connected v1 (same #params)",
            "builder": lambda: DenseSNN(feature_dim, hidden_dim_dense, num_classes),
        },
        {
            "id": "fc_v2",
            "label": "Fully-Connected v2 (width=1024)",
            "builder": lambda: DenseSNN(feature_dim, hidden_dim, num_classes),
        },
        {
            "id": "er",
            "label": "ER Random Graph",
            "builder": lambda: ERSNN(
                input_dim=feature_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                p_active=p_er_active,
            ),
        },
        {
            "id": "g2g_mixer",
            "label": "G2GNet (Proposed, Mixer)",
            "builder": lambda: MixerSNN(
                input_dim=feature_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                num_groups=num_groups,
                p_intra=p_intra,
                p_inter=p_inter,
            ),
        },
    ]
    return heads_cfg


# ---------------------------------------------------------------------
# Main: train all four models and print a Table-I-style summary
# ---------------------------------------------------------------------
def main():
    device = select_device()

    # Same data pipeline as in train.py
    train_loader, test_loader = get_fashion_loaders(batch_size)

    heads_cfg = build_heads()
    results = []

    for cfg in heads_cfg:
        print("\n" + "=" * 70)
        print(f"Training model: {cfg['label']}  (id={cfg['id']})")
        print("=" * 70)

        head = cfg["builder"]()
        model = CNNSNNWrapper(head=head, T_steps=T).to(device)

        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Count classifier parameters only (CNN encoder excluded)
        num_params_head = count_head_params(model.head)
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
                "params_head": num_params_head,
            }
        )

    # Final Table-I-style summary for Fashion-MNIST
    print("\n" + "#" * 80)
    print("Table I-style results on Fashion-MNIST (CNN + SNN)")
    print("#" * 80)
    header = (
        f"{'Connectivity Pattern':35s} | "
        f"{'F-MNIST Acc (%)':16s} | "
        f"{'#Params (head)':>15s}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        acc_percent = 100.0 * r["test_acc"]
        line = (
            f"{r['label']:35s} | "
            f"{acc_percent:16.2f} | "
            f"{r['params_head']:15d}"
        )
        print(line)


if __name__ == "__main__":
    main()
