import os
import sys
import argparse

# Add project root to sys.path so "models", "data", etc. can be imported.
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
warnings.filterwarnings(
    "ignore",
    message=".*aten::lerp.Scalar_out.*"
)

# Device selection (CUDA / DirectML / CPU)
try:
    import torch_directml
    has_dml = True
except ImportError:
    has_dml = False


def select_device():
    """Select compute device (DirectML, CUDA, or CPU)."""
    if has_dml:
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


# Shared hyperparameters
batch_size = 256
T = 50                      # default T, overridable via --T
input_dim = 28 * 28
hidden_dim = 1024
hidden_dim_dense = 447      # Dense baseline
num_classes = 10
num_epochs = 20
lr = 1e-3

# Global state for Dynamic Sparse Training (DST)
global_step = 0
UPDATE_INTERVAL = 1000

# CP/CG rules
cp_mode = "set"
cg_mode = "hebb"

# Hebbian buffer (Mixer only)
hebb_buffer = {
    "fc1": None,
    "fc2": None,
    "fc3": None,
}


def build_model(model_name: str, p_inter: float):
    """Build and return the selected model."""
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


def update_hebb_buffer(input_spikes, activations):
    """Store pre/post activations for Mixer layers."""
    global hebb_buffer
    hebb_buffer["fc1"] = {"pre": input_spikes.cpu(), "post": activations["layer1"].cpu()}
    hebb_buffer["fc2"] = {"pre": activations["layer1"].cpu(), "post": activations["layer2"].cpu()}
    hebb_buffer["fc3"] = {"pre": activations["layer2"].cpu(), "post": activations["layer3"].cpu()}


def compute_ch_matrix(pre_batch, post_batch):
    """Compute cosine similarity CH(i,j)."""
    T_steps, B, N_in = pre_batch.shape
    _, _, N_out = post_batch.shape

    pre_flat = pre_batch.reshape(T_steps * B, N_in).transpose(0, 1)
    post_flat = post_batch.reshape(T_steps * B, N_out).transpose(0, 1)

    eps = 1e-8
    pre_norm = pre_flat / (pre_flat.norm(dim=1, keepdim=True) + eps)
    post_norm = post_flat / (post_flat.norm(dim=1, keepdim=True) + eps)

    return torch.matmul(post_norm, pre_norm.transpose(0, 1))


def _grow_connections(mask_cpu, ch_cpu, num_to_grow, cg_mode_local):
    inactive = ~mask_cpu.bool()
    inactive_idx = inactive.nonzero(as_tuple=False)
    n_inactive = inactive_idx.size(0)
    if n_inactive == 0:
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


def dst_update_layer_cp_cg_single(layer, layer_name, prune_frac, cp_mode_local, cg_mode_local):
    """Pruning + growth on Mixer layers."""
    global hebb_buffer
    buf = hebb_buffer[layer_name]
    pre, post = buf["pre"], buf["post"]

    weight = layer.weight.data
    mask = layer.mask
    device = weight.device

    mask_cpu = mask.cpu()
    w_cpu = weight.cpu()

    active = mask_cpu.bool()
    num_active = active.sum().item()
    total_to_prune = int(prune_frac * num_active)
    if total_to_prune < 1:
        return

    ch_cpu = compute_ch_matrix(pre, post) if (cp_mode_local == "hebb" or cg_mode_local == "hebb") else None

    # --- PRUNING ---
    if cp_mode_local == "set":
        active_weights = w_cpu[active].abs()
        thresh, _ = torch.kthvalue(active_weights, total_to_prune)
        prune_mask = active & (w_cpu.abs() <= thresh)
    elif cp_mode_local == "random":
        active_idx = active.nonzero(as_tuple=False)
        rand_idx = active_idx[torch.randperm(active_idx.size(0))[:total_to_prune]]
        prune_mask = torch.zeros_like(mask_cpu, dtype=torch.bool)
        prune_mask[rand_idx[:, 0], rand_idx[:, 1]] = True
    else:
        scores = ch_cpu[active]
        thresh, _ = torch.kthvalue(scores, total_to_prune)
        prune_mask = active & (ch_cpu <= thresh)

    mask_cpu[prune_mask] = 0
    num_pruned = prune_mask.sum().item()
    if num_pruned <= 0:
        mask.copy_(mask_cpu.to(device))
        weight[~mask.bool()] = 0
        return

    # --- GROWTH ---
    if ch_cpu is None:
        ch_cpu = torch.zeros_like(mask_cpu, dtype=torch.float32)

    mask_cpu, _ = _grow_connections(mask_cpu, ch_cpu, num_pruned, cg_mode_local)

    mask.copy_(mask_cpu.to(device))
    weight[~mask.bool()] = 0


def dst_step(model, prune_frac=0.025):
    for name, module in model.named_modules():
        if isinstance(module, MixerSparseLinear):
            short = name.split(".")[-1]
            dst_update_layer_cp_cg_single(module, short, prune_frac, cp_mode, cg_mode)
    print("[DST] step executed")


def train_one_epoch(model, loader, optimizer, device, epoch_idx, use_dst):
    """Train with progress %."""
    global global_step
    model.train()

    total = 0
    correct = 0

    total_batches = len(loader)
    next_print = 10  # every 10%

    for batch_idx, (images, labels) in enumerate(loader):
        # ---- PROGRESS PRINT ----
        progress = int(((batch_idx + 1) / total_batches) * 100)
        if progress >= next_print or batch_idx == 0 or batch_idx == total_batches - 1:
            print(f"[Epoch {epoch_idx}] Progress: {progress:3d}% ({batch_idx + 1}/{total_batches})")
            next_print += 10
        # ------------------------

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

        if use_dst and isinstance(model, MixerSNN) and global_step % UPDATE_INTERVAL == 0:
            dst_step(model)

        global_step += 1

        preds = spk_counts.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    """Eval accuracy."""
    model.eval()
    total = 0
    correct = 0
    for images, labels in loader:
        labels = labels.to(device, non_blocking=True)
        spikes = rate_encode(images, T).to(device)
        preds = model(spikes).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


@torch.no_grad()
def compute_firing_rates(model, loader, device):
    """Compute firing rates for all hidden layers."""
    model.eval()

    total_samples = 0
    l1 = l2 = l3 = None

    for images, _ in loader:
        B = images.size(0)
        total_samples += B

        spikes = rate_encode(images, T).to(device)
        _, hidden = model(spikes, return_hidden_spikes=True)

        bl1 = hidden["layer1"].sum(dim=0)
        bl2 = hidden["layer2"].sum(dim=0)
        bl3 = hidden["layer3"].sum(dim=0)

        if l1 is None:
            l1, l2, l3 = bl1, bl2, bl3
        else:
            l1 += bl1
            l2 += bl2
            l3 += bl3

    denom = T * total_samples
    r1 = l1 / denom
    r2 = l2 / denom
    r3 = l3 / denom

    return {
        "layer1_mean": r1.mean().item(),
        "layer2_mean": r2.mean().item(),
        "layer3_mean": r3.mean().item(),
        "overall_hidden_mean": torch.cat([r1, r2, r3]).mean().item()
    }


def parse_args():
    p = argparse.ArgumentParser(description="Train SNN on Fashion-MNIST")

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

    return p.parse_args()


def main():
    args = parse_args()
    device = select_device()

    global cp_mode, cg_mode, T
    cp_mode = args.cp
    cg_mode = args.cg
    T = args.T  # override global T

    print(f"Selected model: {args.model}")
    print(f"Sparsity mode: {args.sparsity_mode}")
    print(f"Time steps T: {T}")

    if args.sparsity_mode == "dynamic":
        print(f"C_P (prune): {cp_mode}")
        print(f"C_G (grow):  {cg_mode}")
    else:
        print("Static sparsity: no DST (C_P/C_G ignored)")

    # --- CONFIG SUMMARY LINE ---
    num_groups = 8 if args.model in ["index", "random", "mixer"] else "N/A"
    print(
        f"[CONFIG] model={args.model} | T={T} | epochs={args.epochs} | "
        f"batch={batch_size} | groups={num_groups} | p_inter={args.p_inter} | "
        f"sparsity={args.sparsity_mode}"
    )

    train_loader, test_loader = get_fashion_loaders(batch_size)
    model = build_model(args.model, p_inter=args.p_inter).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    use_dst = args.sparsity_mode == "dynamic"

    for epoch in range(1, args.epochs + 1):
        train_acc = train_one_epoch(model, train_loader, optimizer, device, epoch, use_dst)
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
