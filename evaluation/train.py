import os
import sys
import argparse

# Ensure project root is on sys.path so that "models", "data", etc. can be imported
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import torch
from torch import nn

from models.dense_snn import DenseSNN
from models.index_snn import IndexSNN
from models.random_snn import RandomSNN
from models.mixer_snn import MixerSNN, MixerSparseLinear
from data.data_fashionmnist import get_fashion_loaders
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


# Hyperparameters (shared across models)
batch_size = 256
T = 50
input_dim = 28 * 28
hidden_dim = 1024
# Dense baseline with approximately the same number of parameters
hidden_dim_dense = 447
num_classes = 10
num_epochs = 20
lr = 1e-3

# Global state for Dynamic Sparse Training (DST)
global_step = 0
UPDATE_INTERVAL = 100   # number of training steps between DST updates


def build_model(model_name: str, p_inter: float):
    """Construct and return the selected model."""
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

    raise ValueError(f"Unknown model type: {model_name}")


def dst_update_layer_magnitude_random(layer: MixerSparseLinear, prune_frac: float):
    """
    Apply a DST update to a single MixerSparseLinear layer.

    - Prune: remove weights with the smallest absolute value |w|
    - Grow: add the same number of new connections at random
    - Respect layer.mask_static: static connections are never modified
    """
    weight = layer.weight.data
    mask = layer.mask  # current binary mask (0/1)

    # Use static_mask if available; those connections are protected
    static_mask = getattr(layer, "mask_static", None)
    if static_mask is not None:
        static_mask = static_mask.bool()
        non_static = ~static_mask
    else:
        # If no static mask exists, treat all positions as non-static
        non_static = torch.ones_like(mask, dtype=torch.bool)

    # Only non-static positions are eligible for DST
    active = mask.bool() & non_static
    inactive = (~mask.bool()) & non_static

    num_active = active.sum().item()
    if num_active == 0:
        return

    num_prune = int(prune_frac * num_active)
    if num_prune < 1:
        return

    # Prune: smallest |w| among active & non-static positions
    active_weights = weight[active].abs()
    thresh, _ = torch.kthvalue(active_weights, num_prune)
    prune_mask = active & (weight.abs() <= thresh)
    mask[prune_mask] = 0

    # Grow: random new connections in inactive & non-static positions
    inactive = (~mask.bool()) & non_static
    num_inactive = inactive.sum().item()
    num_grow = min(num_prune, num_inactive)
    if num_grow < 1:
        return

    inactive_idx = inactive.nonzero(as_tuple=False)  # [N_inactive, 2]
    perm = torch.randperm(num_inactive, device=weight.device)[:num_grow]
    grow_idx = inactive_idx[perm]
    mask[grow_idx[:, 0], grow_idx[:, 1]] = 1

    # Clear weights of inactive connections for numerical cleanliness
    weight[~mask.bool()] = 0.0


def dst_step(model: nn.Module, prune_frac: float = 0.025):
    """
    Apply a DST update to all MixerSparseLinear layers in the model.
    """
    for module in model.modules():
        if isinstance(module, MixerSparseLinear):
            dst_update_layer_magnitude_random(module, prune_frac)
    print("[DST] step executed")


def train_one_epoch(model, loader, optimizer, device, epoch_idx: int, use_dst: bool):
    """
    Train the model for one epoch over the given DataLoader.
    Optionally apply DST on MixerSNN layers if use_dst is True.
    """
    global global_step
    model.train()
    total = 0
    correct = 0

    for batch_idx, (images, labels) in enumerate(loader):
        labels = labels.to(device, non_blocking=True)

        # Rate encoding from images to spike trains
        spikes = rate_encode(images, T).to(device)   # [T, B, 784]

        optimizer.zero_grad()
        spk_counts = model(spikes)                  # [B, num_classes]
        loss = nn.CrossEntropyLoss()(spk_counts, labels)
        loss.backward()
        optimizer.step()

        # Dynamic Sparse Training step (only in dynamic mode and for MixerSNN)
        if use_dst and isinstance(model, MixerSNN):
            if global_step > 0 and global_step % UPDATE_INTERVAL == 0:
                dst_step(model, prune_frac=0.025)

        global_step += 1

        preds = spk_counts.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    return acc


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate the model on a dataset and return accuracy."""
    model.eval()
    total = 0
    correct = 0

    for batch_idx, (images, labels) in enumerate(loader):
        labels = labels.to(device, non_blocking=True)
        spikes = rate_encode(images, T).to(device)
        spk_counts = model(spikes)

        preds = spk_counts.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    return acc


@torch.no_grad()
def compute_firing_rates(model, loader, device):
    """
    Compute mean firing rates per hidden layer and overall (hidden layers only).
    """
    model.eval()

    total_samples = 0
    l1_sum = None
    l2_sum = None
    l3_sum = None

    for images, _ in loader:
        B = images.size(0)
        total_samples += B

        spikes = rate_encode(images, T).to(device)
        spk_out_sum, hidden_spikes = model(spikes, return_hidden_spikes=True)

        # hidden_spikes["layerX"]: [B, hidden_dim] (sum of spikes over time)
        batch_l1 = hidden_spikes["layer1"].sum(dim=0)  # [hidden_dim]
        batch_l2 = hidden_spikes["layer2"].sum(dim=0)
        batch_l3 = hidden_spikes["layer3"].sum(dim=0)

        if l1_sum is None:
            l1_sum = batch_l1
            l2_sum = batch_l2
            l3_sum = batch_l3
        else:
            l1_sum += batch_l1
            l2_sum += batch_l2
            l3_sum += batch_l3

    denom = T * total_samples
    l1_rate_per_neuron = l1_sum / denom
    l2_rate_per_neuron = l2_sum / denom
    l3_rate_per_neuron = l3_sum / denom

    hidden_concat = torch.cat(
        [l1_rate_per_neuron, l2_rate_per_neuron, l3_rate_per_neuron]
    )

    rates = {
        "layer1_mean": l1_rate_per_neuron.mean().item(),
        "layer2_mean": l2_rate_per_neuron.mean().item(),
        "layer3_mean": l3_rate_per_neuron.mean().item(),
        "overall_hidden_mean": hidden_concat.mean().item(),
    }

    return rates


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train SNN on Fashion-MNIST with different connectivity patterns."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="dense",
        choices=["dense", "index", "random", "mixer"],
        help="Model type.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=num_epochs,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--p_inter",
        type=float,
        default=0.15,
        help="Inter-group connection probability p' for sparse models "
             "(Index, Random, Mixer). Ignored for the dense model.",
    )

    parser.add_argument(
        "--sparsity_mode",
        type=str,
        default="static",
        choices=["static", "dynamic"],
        help="Sparsity mode for sparse models: "
             "'static' = only initial structured sparsity, "
             "'dynamic' = apply Dynamic Sparse Training (DST) on non-static connections.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    device = select_device()

    print(f"Selected model: {args.model}")
    print(f"Sparsity mode: {args.sparsity_mode}")

    train_loader, test_loader = get_fashion_loaders(batch_size)
    model = build_model(args.model, p_inter=args.p_inter).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    use_dst = (args.sparsity_mode == "dynamic")

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, train_loader, optimizer, device, epoch, use_dst=use_dst)
        test_acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch:02d} | test_acc={test_acc:.4f}")

    rates = compute_firing_rates(model, test_loader, device)
    print("Average firing rates (test set):")
    print(f"  Layer 1 mean rate:        {rates['layer1_mean']:.6f}")
    print(f"  Layer 2 mean rate:        {rates['layer2_mean']:.6f}")
    print(f"  Layer 3 mean rate:        {rates['layer3_mean']:.6f}")
    print(f"  Overall hidden mean rate: {rates['overall_hidden_mean']:.6f}")


if __name__ == "__main__":
    main()
