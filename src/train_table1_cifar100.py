# src/train_table1_cifar100.py
#
# Table-I-style comparison on CIFAR-100
# Paper-faithful patch encoder (16 patches) + SNN heads (FC v1, FC v2, ER, G2G).

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
from models.cnn_encoder import PatchEncoder

from data.cifar10_100 import get_cifar100_loaders

warnings.filterwarnings("ignore", message=".*aten::lerp.Scalar_out.*")

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
    print("No GPU backend available – using CPU.")
    return device


# ---------------------------------------------------------------------
# Hyperparameters 
# ---------------------------------------------------------------------
batch_size = 256
T = 70

feature_dim = 512
hidden_dim = 1024
hidden_dim_dense = 469
num_classes = 100
num_epochs = 30
lr = 1e-3
weight_decay = 1e-4

num_groups = 8
p_intra = 1.0
p_inter = 0.15
p_er_active = 0.25625


class CNNSNNWrapper(nn.Module):
    def __init__(self, head: nn.Module, T_steps: int = T):
        super().__init__()

        # 32x32 split into 4x4=16 patches of size 8x8, conv per patch -> [B, 32, 4, 4] -> flatten [B, 512]
        self.encoder = PaperPatchEncoder(in_channels=3, img_size=32, out_channels=32, grid_size=4)
        self.head = head
        self.T = T_steps

    def encode_to_sequence(self, images: torch.Tensor) -> torch.Tensor:
        # images: [B, 3, 32, 32]
        feats = self.encoder(images)  # [B, 512]

        with torch.no_grad():
            denom = feats.abs().max().clamp(min=1e-6)
            feats_norm = torch.clamp(feats / denom, -1, 1)

        x_seq = feats_norm.unsqueeze(0).repeat(self.T, 1, 1)  # [T, B, 512]
        return x_seq

    def forward(self, images: torch.Tensor, return_hidden_spikes: bool = False):
        x_seq = self.encode_to_sequence(images)
        if return_hidden_spikes:
            return self.head(x_seq, return_hidden_spikes=True)
        return self.head(x_seq)


def count_head_params(head: nn.Module) -> int:
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


def train_one_epoch(
    model: CNNSNNWrapper,
    loader,
    optimizer: Adam,
    device: torch.device,
) -> float:
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


def build_heads():
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


def main():
    device = select_device()
    train_loader, test_loader = get_cifar100_loaders(batch_size)

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

    print("\n")
    print("Table-style results on CIFAR-100 (Patch encoder + SNN, best accuracy)")

    id_to_res = {r["id"]: r for r in results}

    g2g_ids = [rid for rid in ("g2g_index", "g2g_mixer") if rid in id_to_res]
    if not g2g_ids:
        raise RuntimeError("No G2G results found (g2g_index / g2g_mixer).")

    g2g_best_acc = max(id_to_res[rid]["test_acc"] for rid in g2g_ids)
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
        print(f"{label:40s} | {acc_percent:14.2f} | {params:15d}")

    print_row(id_to_res["fc_v1"]["label"], id_to_res["fc_v1"]["test_acc"], id_to_res["fc_v1"]["params_head"])
    print_row(id_to_res["fc_v2"]["label"], id_to_res["fc_v2"]["test_acc"], id_to_res["fc_v2"]["params_head"])
    print_row(id_to_res["er"]["label"], id_to_res["er"]["test_acc"], id_to_res["er"]["params_head"])
    print_row("G2GNet (Proposed, best of Index/Mixer)", g2g_best_acc, g2g_params)

    print("\nDetails for individual G2G variants:")
    for gid in g2g_ids:
        r = id_to_res[gid]
        print_row(r["label"], r["test_acc"], r["params_head"])

    os.makedirs("table1_results", exist_ok=True)
    import json

    with open("table1_results/table1_cifar100.txt", "w") as f:
        f.write("Connectivity Pattern,Accuracy,Params\n")
        f.write(f"{id_to_res['fc_v1']['label']},{100*id_to_res['fc_v1']['test_acc']:.2f},{id_to_res['fc_v1']['params_head']}\n")
        f.write(f"{id_to_res['fc_v2']['label']},{100*id_to_res['fc_v2']['test_acc']:.2f},{id_to_res['fc_v2']['params_head']}\n")
        f.write(f"{id_to_res['er']['label']},{100*id_to_res['er']['test_acc']:.2f},{id_to_res['er']['params_head']}\n")
        f.write(f"G2GNet (Proposed, best of Index/Mixer),{100*g2g_best_acc:.2f},{g2g_params}\n")

    with open("table1_results/table1_cifar100_full.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
