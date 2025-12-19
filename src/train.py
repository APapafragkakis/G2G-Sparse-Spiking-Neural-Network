import os
import sys
import argparse

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import torch
from torch import nn

from models.dense_snn import DenseSNN
from models.index_snn import IndexSNN
from models.random_snn import RandomSNN
from models.mixer_snn import MixerSNN, MixerSparseLinear

from data.fashionmnist import get_fashion_loaders
from data.cifar10_100 import get_cifar10_loaders, get_cifar100_loaders
from utils.encoding import rate_encode

import warnings
warnings.filterwarnings("ignore", message=".*aten::lerp.Scalar_out.*")

try:
    import torch_directml
    has_dml = True
except ImportError:
    has_dml = False


def select_device():
    if has_dml:
        device = torch_directml.device()
        print(f"Using DirectML device: {device}")
        return device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)} | CUDA {torch.version.cuda}")
        return device
    print("No GPU backend available — using CPU.")
    return torch.device("cpu")



batch_size = 256
T = 50
input_dim = 28 * 28
hidden_dim = 1024
hidden_dim_dense = 447
num_classes = 10
num_epochs = 20
lr = 1e-3

# Dynamic Sparse Training (DST) configuration
global_step = 0
UPDATE_INTERVAL = 1000
cp_mode = "set"
cg_mode = "hebb"

# Buffers for Hebbian pre/post activity
hebb_buffer = {"fc1": None, "fc2": None, "fc3": None}


def build_model(model_name: str, p_inter: float):
    if model_name == "dense":
        return DenseSNN(input_dim, hidden_dim_dense, num_classes)
    elif model_name == "index":
        return IndexSNN(input_dim, hidden_dim, num_classes, 8, 1.0, p_inter)
    elif model_name == "random":
        return RandomSNN(input_dim, hidden_dim, num_classes, 8, 1.0, p_inter)
    elif model_name == "mixer":
        return MixerSNN(input_dim, hidden_dim, num_classes, 8, 1.0, p_inter)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def update_hebb_buffer(input_spikes, activations):
    global hebb_buffer
    hebb_buffer["fc1"] = {"pre": input_spikes.cpu(), "post": activations["layer1"].cpu()}
    hebb_buffer["fc2"] = {"pre": activations["layer1"].cpu(), "post": activations["layer2"].cpu()}
    hebb_buffer["fc3"] = {"pre": activations["layer2"].cpu(), "post": activations["layer3"].cpu()}


def compute_ch_matrix(pre_batch, post_batch):
    T_steps, B, N_in = pre_batch.shape
    _, _, N_out = post_batch.shape

    pre_flat = pre_batch.reshape(T_steps * B, N_in).t()
    post_flat = post_batch.reshape(T_steps * B, N_out).t()

    eps = 1e-8
    pre_norm = pre_flat / (pre_flat.norm(dim=1, keepdim=True) + eps)
    post_norm = post_flat / (post_flat.norm(dim=1, keepdim=True) + eps)

    return post_norm @ pre_norm.t()


def _grow_connections(mask_cpu, ch_cpu, num_to_grow, mode):
    inactive = ~mask_cpu.bool()
    inactive_idx = inactive.nonzero(as_tuple=False)

    if inactive_idx.size(0) == 0 or num_to_grow <= 0:
        return mask_cpu, 0

    num_to_grow = min(num_to_grow, inactive_idx.size(0))

    if mode == "hebb":
        scores = ch_cpu[inactive]
        _, top_idx = torch.topk(scores, k=num_to_grow)
        grow_idx = inactive_idx[top_idx]
    else:
        perm = torch.randperm(inactive_idx.size(0))[:num_to_grow]
        grow_idx = inactive_idx[perm]

    mask_cpu[grow_idx[:, 0], grow_idx[:, 1]] = 1
    return mask_cpu, num_to_grow


def dst_update_layer_cp_cg_single(layer, layer_name, prune_frac, cp_mode_local, cg_mode_local):
    global hebb_buffer
    buf = hebb_buffer.get(layer_name)
    if buf is None:
        return

    pre_batch = buf["pre"]
    post_batch = buf["post"]

    weight = layer.weight.data
    mask = layer.mask
    device = weight.device

    mask_cpu = mask.cpu()
    w_cpu = weight.cpu()

    active = mask_cpu.bool()
    num_active = active.sum().item()
    if num_active == 0:
        return

    total_to_prune = int(prune_frac * num_active)
    if total_to_prune < 1:
        return

    ch_cpu = None
    if cp_mode_local == "hebb" or cg_mode_local == "hebb":
        ch_cpu = compute_ch_matrix(pre_batch, post_batch)

    if cp_mode_local == "set":
        active_weights = w_cpu[active].abs().view(-1)
        n_eff = min(total_to_prune, active_weights.numel())
        if n_eff < 1:
            return
        thresh, _ = torch.kthvalue(active_weights, n_eff)
        prune_mask = active & (w_cpu.abs() <= thresh)

    elif cp_mode_local == "random":
        active_idx = active.nonzero(as_tuple=False)
        n_eff = min(total_to_prune, active_idx.size(0))
        if n_eff < 1:
            return
        perm = torch.randperm(active_idx.size(0))[:n_eff]
        chosen = active_idx[perm]
        prune_mask = torch.zeros_like(mask_cpu, dtype=torch.bool)
        prune_mask[chosen[:, 0], chosen[:, 1]] = True

    else:  # Hebbian pruning
        if ch_cpu is None:
            return
        active_scores = ch_cpu[active].view(-1)
        n_eff = min(total_to_prune, active_scores.numel())
        if n_eff < 1:
            return
        thresh, _ = torch.kthvalue(active_scores, n_eff)
        prune_mask = active & (ch_cpu <= thresh)

    mask_cpu[prune_mask] = 0
    num_pruned = prune_mask.sum().item()

    if ch_cpu is None:
        ch_cpu = torch.zeros_like(mask_cpu, dtype=torch.float32)

    mask_cpu, _ = _grow_connections(mask_cpu, ch_cpu, num_pruned, cg_mode_local)
    mask.copy_(mask_cpu.to(device))
    weight[~mask.bool()] = 0.0


def dst_step(model, prune_frac=0.025):
    for name, module in model.named_modules():
        if isinstance(module, MixerSparseLinear):
            short = name.split(".")[-1]
            if short in hebb_buffer:
                dst_update_layer_cp_cg_single(module, short, prune_frac, cp_mode, cg_mode)
    print("[DST] step executed")


def train_one_epoch(model, loader, optimizer, device, epoch_idx, use_dst):
    global global_step
    model.train()

    total = 0
    correct = 0
    total_batches = len(loader)

    for batch_idx, (images, labels) in enumerate(loader):
        progress = (batch_idx + 1) / total_batches
        bar_len = 30
        filled = int(bar_len * progress)
        bar = "█" * filled + "░" * (bar_len - filled)
        percent = int(progress * 100)
        end_char = "\r" if (batch_idx + 1) < total_batches else "\n"

        print(
            f"[Epoch {epoch_idx}] [{bar}] {percent:3d}% ({batch_idx + 1}/{total_batches})",
            end=end_char,
            flush=True,
        )

        labels = labels.to(device, non_blocking=True)
        spikes = rate_encode(images, T).to(device)

        optimizer.zero_grad()

        if use_dst and isinstance(model, MixerSNN):
            spk_counts, acts = model(spikes, return_activations=True)
            update_hebb_buffer(spikes, acts)
        else:
            spk_counts = model(spikes)

        loss = nn.CrossEntropyLoss()(spk_counts, labels)
        loss.backward()
        optimizer.step()

        if use_dst and isinstance(model, MixerSNN):
            if global_step > 0 and global_step % UPDATE_INTERVAL == 0:
                dst_step(model)

        global_step += 1

        preds = spk_counts.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0

    for images, labels in loader:
        labels = labels.to(device, non_blocking=True)
        spikes = rate_encode(images, T).to(device)
        spk_counts = model(spikes)

        preds = spk_counts.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total


def parse_args():
    p = argparse.ArgumentParser(description="Train SNN models on multiple datasets.")
    p.add_argument("--dataset", type=str, default="fashionmnist",
                   choices=["fashionmnist", "cifar10", "cifar100"])
    p.add_argument("--model", type=str, default="dense",
                   choices=["dense", "index", "random", "mixer"])
    p.add_argument("--epochs", type=int, default=num_epochs)
    p.add_argument("--p_inter", type=float, default=0.15)
    p.add_argument("--sparsity_mode", type=str, default="static",
                   choices=["static", "dynamic"])
    p.add_argument("--cp", type=str, default="set",
                   choices=["set", "random", "hebb"])
    p.add_argument("--cg", type=str, default="hebb",
                   choices=["hebb", "random"])
    p.add_argument("--T", type=int, default=T)
    p.add_argument("--batch_size", type=int, default=batch_size)
    return p.parse_args()


def main():
    args = parse_args()
    device = select_device()

    global cp_mode, cg_mode, T, batch_size
    cp_mode = args.cp
    cg_mode = args.cg
    T = args.T
    batch_size = args.batch_size

    global input_dim, num_classes

    if args.dataset == "fashionmnist":
        input_dim = 28 * 28
        num_classes = 10
        train_loader, test_loader = get_fashion_loaders(batch_size)

    elif args.dataset == "cifar10":
        input_dim = 3 * 32 * 32
        num_classes = 10
        train_loader, test_loader = get_cifar10_loaders(batch_size)

    elif args.dataset == "cifar100":
        input_dim = 3 * 32 * 32
        num_classes = 100
        train_loader, test_loader = get_cifar100_loaders(batch_size)

    num_groups = 8 if args.model in ["index", "random", "mixer"] else "N/A"

    print(
        f"[CONFIG] dataset={args.dataset} | model={args.model} | "
        f"sparsity={args.sparsity_mode} | T={T} | epochs={args.epochs} | "
        f"batch={batch_size} | groups={num_groups} | p_inter={args.p_inter}"
    )

    model = build_model(args.model, args.p_inter).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    use_dst = (args.sparsity_mode == "dynamic")

    for epoch in range(1, args.epochs + 1):
        train_acc = train_one_epoch(model, train_loader, optimizer, device, epoch, use_dst)
        test_acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch:02d} | train_acc={train_acc:.4f} | test_acc={test_acc:.4f}")


if __name__ == "__main__":
    main()
