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


# hyperparameters (μερικά γίνονται override από CLI)
batch_size = 256
T = 50
input_dim = 28 * 28
hidden_dim = 1024
hidden_dim_dense = 447
num_classes = 10
num_epochs = 20
lr = 1e-3

# DST globals
global_step = 0
UPDATE_INTERVAL = 1000
cp_mode = "set"
cg_mode = "hebb"
hebb_buffer = {"fc1": None, "fc2": None, "fc3": None}


def build_model(model_name: str, p_inter: float):
    if model_name == "dense":
        return DenseSNN(input_dim, hidden_dim_dense, num_classes)
    elif model_name == "index":
        return IndexSNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_groups=8,
            p_intra=1.0,
            p_inter=p_inter,
        )
    elif model_name == "random":
        return RandomSNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_groups=8,
            p_intra=1.0,
            p_inter=p_inter,
        )
    elif model_name == "mixer":
        return MixerSNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_groups=8,
            p_intra=1.0,
            p_inter=p_inter,
        )
    else:
        raise ValueError(f"Unknown model type: {model_name}")


def update_hebb_buffer(input_spikes: torch.Tensor, activations: dict):
    global hebb_buffer
    hebb_buffer["fc1"] = {
        "pre": input_spikes.detach().cpu(),
        "post": activations["layer1"].detach().cpu(),
    }
    hebb_buffer["fc2"] = {
        "pre": activations["layer1"].detach().cpu(),
        "post": activations["layer2"].detach().cpu(),
    }
    hebb_buffer["fc3"] = {
        "pre": activations["layer2"].detach().cpu(),
        "post": activations["layer3"].detach().cpu(),
    }


def compute_ch_matrix(pre_batch: torch.Tensor, post_batch: torch.Tensor) -> torch.Tensor:
    T_steps, B, N_in = pre_batch.shape
    _, _, N_out = post_batch.shape
    pre_flat = pre_batch.reshape(T_steps * B, N_in).t()
    post_flat = post_batch.reshape(T_steps * B, N_out).t()
    eps = 1e-8
    pre_norm = pre_flat / (pre_flat.norm(dim=1, keepdim=True) + eps)
    post_norm = post_flat / (post_flat.norm(dim=1, keepdim=True) + eps)
    return post_norm @ pre_norm.t()


def _grow_connections(mask_cpu: torch.Tensor,
                      ch_cpu: torch.Tensor,
                      num_to_grow: int,
                      cg_mode_local: str):
    inactive = ~mask_cpu.bool()
    inactive_idx = inactive.nonzero(as_tuple=False)
    n_inactive = inactive_idx.size(0)
    if n_inactive == 0 or num_to_grow <= 0:
        return mask_cpu, 0
    num_to_grow = min(num_to_grow, n_inactive)

    if cg_mode_local == "hebb":
        scores = ch_cpu[inactive]
        _, top_idx = torch.topk(scores, k=num_to_grow, largest=True)
        grow_idx = inactive_idx[top_idx]
    else:
        perm = torch.randperm(n_inactive)[:num_to_grow]
        grow_idx = inactive_idx[perm]

    mask_cpu[grow_idx[:, 0], grow_idx[:, 1]] = 1
    return mask_cpu, num_to_grow


def dst_update_layer_cp_cg_single(layer: MixerSparseLinear,
                                  layer_name: str,
                                  prune_frac: float,
                                  cp_mode_local: str,
                                  cg_mode_local: str):
    global hebb_buffer
    buf = hebb_buffer.get(layer_name, None)
    if buf is None:
        return
    pre_batch = buf["pre"]
    post_batch = buf["post"]

    weight = layer.weight.data
    mask = layer.mask
    device = weight.device

    mask_cpu = mask.detach().cpu()
    w_cpu = weight.detach().cpu()

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

    # pruning
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
        rand_idx = active_idx[perm]
        prune_mask = torch.zeros_like(mask_cpu, dtype=torch.bool)
        prune_mask[rand_idx[:, 0], rand_idx[:, 1]] = True
    else:  # hebb
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
    if num_pruned <= 0:
        mask.copy_(mask_cpu.to(device))
        weight[~mask.bool()] = 0.0
        return

    # growth
    if ch_cpu is None:
        ch_cpu = torch.zeros_like(mask_cpu, dtype=torch.float32)

    mask_cpu, _ = _grow_connections(mask_cpu, ch_cpu, num_pruned, cg_mode_local)
    mask.copy_(mask_cpu.to(device))
    weight[~mask.bool()] = 0.0


def dst_step(model: nn.Module, prune_frac: float = 0.025):
    for name, module in model.named_modules():
        if not isinstance(module, MixerSparseLinear):
            continue
        short_name = name.split(".")[-1]
        if short_name not in hebb_buffer:
            continue
        dst_update_layer_cp_cg_single(
            module,
            short_name,
            prune_frac,
            cp_mode,
            cg_mode,
        )
    print("[DST] step executed")


def train_one_epoch(model, loader, optimizer, device, epoch_idx: int, use_dst: bool):
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
            f"[Epoch {epoch_idx}] [{bar}] {percent:3d}% "
            f"({batch_idx + 1}/{total_batches})",
            end=end_char,
            flush=True,
        )

        labels = labels.to(device, non_blocking=True)
        spikes = rate_encode(images, T).to(device)

        optimizer.zero_grad()

        if use_dst and isinstance(model, MixerSNN):
            spk_counts, activations = model(spikes, return_activations=True)
            update_hebb_buffer(spikes, activations)
        else:
            spk_counts = model(spikes)

        loss = nn.CrossEntropyLoss()(spk_counts, labels)
        loss.backward()
        optimizer.step()

        if use_dst and isinstance(model, MixerSNN):
            if global_step > 0 and global_step % UPDATE_INTERVAL == 0:
                dst_step(model, prune_frac=0.025)

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


@torch.no_grad()
def compute_firing_rates(model, loader, device):
    model.eval()
    total_samples = 0
    l1_sum = l2_sum = l3_sum = None

    for images, _ in loader:
        B = images.size(0)
        total_samples += B
        spikes = rate_encode(images, T).to(device)
        _, hidden_spikes = model(spikes, return_hidden_spikes=True)
        batch_l1 = hidden_spikes["layer1"].sum(dim=0)
        batch_l2 = hidden_spikes["layer2"].sum(dim=0)
        batch_l3 = hidden_spikes["layer3"].sum(dim=0)

        if l1_sum is None:
            l1_sum, l2_sum, l3_sum = batch_l1, batch_l2, batch_l3
        else:
            l1_sum += batch_l1
            l2_sum += batch_l2
            l3_sum += batch_l3

    denom = T * total_samples
    l1_rate = l1_sum / denom
    l2_rate = l2_sum / denom
    l3_rate = l3_sum / denom
    hidden_concat = torch.cat([l1_rate, l2_rate, l3_rate])

    return {
        "layer1_mean": l1_rate.mean().item(),
        "layer2_mean": l2_rate.mean().item(),
        "layer3_mean": l3_rate.mean().item(),
        "overall_hidden_mean": hidden_concat.mean().item(),
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="Train SNN on Fashion-MNIST with different connectivity patterns."
    )
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
    p.add_argument("--T", type=int, default=T,
                   help="Number of time steps for SNN simulation.")
    return p.parse_args()


def main():
    args = parse_args()
    device = select_device()

    global cp_mode, cg_mode, T
    cp_mode = args.cp
    cg_mode = args.cg
    T = args.T

    num_groups = 8 if args.model in ["index", "random", "mixer"] else "N/A"
    print(
        f"[CONFIG] model={args.model} | sparsity={args.sparsity_mode} | "
        f"T={T} | epochs={args.epochs} | batch={batch_size} | "
        f"groups={num_groups} | p_inter={args.p_inter}"
    )

    train_loader, test_loader = get_fashion_loaders(batch_size)
    model = build_model(args.model, p_inter=args.p_inter).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    use_dst = (args.sparsity_mode == "dynamic")

    for epoch in range(1, args.epochs + 1):
        train_acc = train_one_epoch(model, train_loader, optimizer, device, epoch, use_dst=use_dst)
        test_acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch:02d} | train_acc={train_acc:.4f} | test_acc={test_acc:.4f}")

    rates = compute_firing_rates(model, test_loader, device)
    print("Average firing rates (test set):")
    print(f"  Layer 1 mean rate:        {rates['layer1_mean']:.6f}")
    print(f"  Layer 2 mean rate:        {rates['layer2_mean']:.6f}")
    print(f"  Layer 3 mean rate:        {rates['layer3_mean']:.6f}")
    print(f"  Overall hidden mean rate: {rates['overall_hidden_mean']:.6f}")


if __name__ == "__main__":
    main()
