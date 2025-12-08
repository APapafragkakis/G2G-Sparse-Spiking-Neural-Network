# src/train_table1_cifar10.py
#
# CNN (CIFARCNNEncoder) + SNN heads (FC v1, FC v2, ER, G2G/Mixer).

import os
import sys
import warnings

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import torch
from torch import nn
from torch.optim import Adam

from models.dense_snn import DenseSNN
from models.mixer_snn import MixerSNN, MixerSparseLinear
from models.er_snn import ERSNN, ERSparseLinear
from models.cnn_encoder import CIFARCNNEncoder
from data.cifar10_100 import get_cifar10_loaders

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
    print("No GPU backend available — using CPU.")
    return device


batch_size = 256
T = 50
feature_dim = 512
hidden_dim = 1024
hidden_dim_dense = 512
num_classes = 10
num_epochs = 20
lr = 1e-3
weight_decay = 1e-4

num_groups = 8
p_intra = 1.0
p_inter = 0.15
p_er_active = 0.18


class CNNSNNWrapper(nn.Module):
    def __init__(self, head: nn.Module, T_steps: int = T):
        super().__init__()
        self.encoder = CIFARCNNEncoder()
        self.head = head
        self.T = T_steps

    def encode_to_sequence(self, images: torch.Tensor) -> torch.Tensor:
        # images: [B, 3, 32, 32]
        feats = self.encoder(images)  # [B, 512]

        with torch.no_grad():
            f_min = feats.min(dim=1, keepdim=True)[0]
            f_max = feats.max(dim=1, keepdim=True)[0]
            denom = (f_max - f_min).clamp(min=1e-6)
            feats_norm = (feats - f_min) / denom

        x_seq = feats_norm.unsqueeze(0).repeat(self.T, 1, 1)
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

        if isinstance(module, MixerSparseLinear):
            mask = module.mask
            total += int(mask.sum().item())
            if module.bias is not None:
                total += module.bias.numel()
        elif isinstance(module, ERSparseLinear):
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


def main():
    device = select_device()
    train_loader, test_loader = get_cifar10_loaders(batch_size)

    heads_cfg = build_heads()
    results = []

    for cfg in heads_cfg:
        print("\n" + "=" * 70)
        print(f"Training model: {cfg['label']}  (id={cfg['id']})")
        print("=" * 70)

        head = cfg["builder"]()
        model = CNNSNNWrapper(head=head, T_steps=T).to(device)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

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

    print("\n" + "#" * 80)
    print("Table I-style results on CIFAR-10 (CNN + SNN)")
    print("#" * 80)
    header = (
        f"{'Connectivity Pattern':35s} | "
        f"{'CIFAR-10 Acc (%)':16s} | "
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

    os.makedirs("table1_results", exist_ok=True)

    import json

    with open("table1_results/table1_cifar10.txt", "w") as f:
        f.write("Connectivity Pattern,Accuracy,Params\n")
        for r in results:
            f.write(f"{r['label']},{100*r['test_acc']:.2f},{r['params_head']}\n")

    with open("table1_results/table1_cifar10.csv", "w") as f:
        f.write("id,label,accuracy,params\n")
        for r in results:
            f.write(f"{r['id']},{r['label']},{100*r['test_acc']:.2f},{r['params_head']}\n")

    with open("table1_results/table1_cifar10.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
