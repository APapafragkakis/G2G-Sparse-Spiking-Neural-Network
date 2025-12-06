# src/train_table1_fmnist.py
#
# Table-style comparison on Fashion-MNIST
# CNN (PatchConvEncoder) + SNN heads (FC v1, FC v2, ER, G2G-Index/Mixer).

import os
import sys
import warnings

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.dense_snn import DenseSNN
from models.mixer_snn import MixerSNN, MixerSparseLinear
from models.er_snn import ERSNN, ERSparseLinear
from models.index_snn import IndexSNN, IndexSparseLinear
from models.cnn_encoder import PatchConvEncoder
from data.fashionmnist import get_fashion_loaders

warnings.filterwarnings("ignore", message=".*aten::lerp.Scalar_out.*")

# ---------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------
try:
    import torch_directml
    HAS_DML = True
except ImportError:
    HAS_DML = False


def select_device() -> torch.device:
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
# Hyperparameters for the SNN Table-1-style experiment
# ---------------------------------------------------------------------
batch_size = 256
T = 50                  # Number of timesteps for rate-based SNN input
feature_dim = 512        # Output dimension of PatchConvEncoder
hidden_dim = 1024        # Width of the main SNN head (matches G2GNet paper)
hidden_dim_dense = 469   # Narrow FC baseline with similar param budget
num_classes = 10
num_epochs = 30
lr = 1e-3
weight_decay = 1e-5

num_groups = 8
p_intra = 1.0
p_inter = 0.15
p_er_active = 0.25625    # ER sparsity chosen to match G2G param budget


# ---------------------------------------------------------------------
# CNN + SNN wrapper
# ---------------------------------------------------------------------
class CNNSNNWrapper(nn.Module):
    """
    Wraps a CNN feature extractor and an SNN classifier head.
    The CNN encoder is shared across all connectivity patterns.
    """

    def __init__(self, head: nn.Module, T_steps: int = T):
        super().__init__()
        self.encoder = PatchConvEncoder(in_channels=1, out_channels=32)
        self.head = head
        self.T = T_steps

    def encode_to_sequence(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to a feature vector and tile it along the time dimension.
        """
        # images: [B, 1, 28, 28]
        feats = self.encoder(images)  # [B, 512]

        with torch.no_grad():
            f_min = feats.min(dim=1, keepdim=True)[0]
            f_max = feats.max(dim=1, keepdim=True)[0]
            denom = (f_max - f_min).clamp(min=1e-6)
            feats_norm = (feats - f_min) / denom

        x_seq = feats_norm.unsqueeze(0).repeat(self.T, 1, 1)  # [T, B, 512]
        return x_seq

    def forward(self, images: torch.Tensor, return_hidden_spikes: bool = False):
        x_seq = self.encode_to_sequence(images)
        if return_hidden_spikes:
            return self.head(x_seq, return_hidden_spikes=True)
        return self.head(x_seq)


# ---------------------------------------------------------------------
# Parameter counting (head only)
# ---------------------------------------------------------------------
def count_head_params(head: nn.Module) -> int:
    """
    Count only the parameters of the SNN head.
    For sparse layers, only active connections are counted.
    """
    total = 0
    for module in head.modules():
        if module is head:
            continue

        if isinstance(module, (MixerSparseLinear, ERSparseLinear, IndexSparseLinear)):
            mask = module.mask
            total += int(mask.sum().item())
            if module.bias is not None:
                total += module.bias.numel()
        elif isinstance(module, nn.Linear):
            total += module.weight.numel()
            if module.bias is not None:
                total += module.bias.numel()

    return total


# ---------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------
def train_one_epoch(
    model: CNNSNNWrapper,
    loader,
    optimizer: Adam,
    device: torch.device,
) -> float:
    """
    Single training epoch. Returns training accuracy.
    """
    model.train()
    total = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
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
    """
    Evaluate model on a given data loader. Returns accuracy.
    """
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
# Heads (FC v1, FC v2, ER, G2G-Index, G2G-Mixer)
# ---------------------------------------------------------------------
def build_heads():
    """
    Define all connectivity patterns to be evaluated.
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
            "id": "g2g_index",
            "label": "G2GNet (Index grouping)",
            "builder": lambda: IndexSNN(
                input_dim=feature_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                num_groups=num_groups,
                p_intra=p_intra,
                p_inter=p_inter,
            ),
        },
        {
            "id": "g2g_mixer",
            "label": "G2GNet (Mixer grouping)",
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
# Main
# ---------------------------------------------------------------------
def main():
    device = select_device()
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
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

        num_params_head = count_head_params(model.head)
        best_test_acc = 0.0

        for epoch in range(1, num_epochs + 1):
            train_acc = train_one_epoch(model, train_loader, optimizer, device)
            test_acc = evaluate(model, test_loader, device)

            if test_acc > best_test_acc:
                best_test_acc = test_acc

            scheduler.step()

            if epoch % 10 == 0 or epoch == num_epochs:
                print(
                    f"[{cfg['id']}] Epoch {epoch:02d} | "
                    f"train_acc={train_acc:.4f} | test_acc={test_acc:.4f} | best={best_test_acc:.4f}"
                )

        results.append(
            {
                "id": cfg["id"],
                "label": cfg["label"],
                "test_acc": best_test_acc,
                "params_head": num_params_head,
            }
        )

    # -----------------------------------------------------------------
    # Aggregate results and build a Table-1-style summary
    # -----------------------------------------------------------------
    print("\n" + "#" * 80)
    print("Table-style results on Fashion-MNIST (CNN + SNN, best accuracy)")
    print("#" * 80)

    # Index results by id for easier access
    id_to_res = {r["id"]: r for r in results}

    # Compute best G2G accuracy across Index and Mixer
    g2g_ids = [rid for rid in ("g2g_index", "g2g_mixer") if rid in id_to_res]
    if not g2g_ids:
        raise RuntimeError("No G2G results found (g2g_index / g2g_mixer).")

    g2g_best_acc = max(id_to_res[rid]["test_acc"] for rid in g2g_ids)
    # Assume same param budget for both G2G variants
    g2g_params = id_to_res[g2g_ids[0]]["params_head"]

    header = (
        f"{'Connectivity Pattern':40s} | "
        f"{'Best Acc (%)':14s} | "
        f"{'#Params (head)':>15s}"
    )
    print(header)
    print("-" * len(header))

    def print_row(label: str, acc: float, params: int):
        acc_percent = 100.0 * acc
        line = (
            f"{label:40s} | "
            f"{acc_percent:14.2f} | "
            f"{params:15d}"
        )
        print(line)

    # Main comparison rows
    print_row(
        id_to_res["fc_v1"]["label"],
        id_to_res["fc_v1"]["test_acc"],
        id_to_res["fc_v1"]["params_head"],
    )
    print_row(
        id_to_res["fc_v2"]["label"],
        id_to_res["fc_v2"]["test_acc"],
        id_to_res["fc_v2"]["params_head"],
    )
    print_row(
        id_to_res["er"]["label"],
        id_to_res["er"]["test_acc"],
        id_to_res["er"]["params_head"],
    )
    print_row(
        "G2GNet (Proposed, best of Index/Mixer)",
        g2g_best_acc,
        g2g_params,
    )

    # Optional: also print individual G2G variants for inspection
    print("\nDetails for individual G2G variants:")
    for gid in g2g_ids:
        r = id_to_res[gid]
        print_row(r["label"], r["test_acc"], r["params_head"])

    # -----------------------------------------------------------------
    # Save raw results to disk
    # -----------------------------------------------------------------
    os.makedirs("table1_results", exist_ok=True)
    import json

    # Compact CSV-style summary (only main rows + aggregated G2G)
    with open("table1_results/table1_fmnist.txt", "w") as f:
        f.write("Connectivity Pattern,Accuracy,Params\n")
        f.write(
            f"{id_to_res['fc_v1']['label']},"
            f"{100*id_to_res['fc_v1']['test_acc']:.2f},"
            f"{id_to_res['fc_v1']['params_head']}\n"
        )
        f.write(
            f"{id_to_res['fc_v2']['label']},"
            f"{100*id_to_res['fc_v2']['test_acc']:.2f},"
            f"{id_to_res['fc_v2']['params_head']}\n"
        )
        f.write(
            f"{id_to_res['er']['label']},"
            f"{100*id_to_res['er']['test_acc']:.2f},"
            f"{id_to_res['er']['params_head']}\n"
        )
        f.write(
            "G2GNet (Proposed, best of Index/Mixer),"
            f"{100*g2g_best_acc:.2f},"
            f"{g2g_params}\n"
        )

    # Full JSON with all individual connectivity patterns
    with open("table1_results/table1_fmnist_full.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
