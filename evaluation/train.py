import os
import sys
import argparse

# Allow imports from project root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import torch
from torch import nn

from models.dense_snn import DenseSNN
from models.index_snn import IndexSNN
from models.random_snn import RandomSNN
from models.mixer_snn import MixerSNN
from data.data_fashionmnist import get_fashion_loaders
from utils.encoding import rate_encode

import warnings
warnings.filterwarnings(
    "ignore",
    message=".*aten::lerp.Scalar_out.*"
)


# Try DirectML for AMD or CUDA for NVIDIA GPUs
try:
    import torch_directml
    has_dml = True
except ImportError:
    has_dml = False


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif has_dml:
        return torch_directml.device()
    else:
        raise RuntimeError("No GPU backend available")


# Hyperparameters (shared across models)
batch_size = 256
T = 50
input_dim = 28 * 28
hidden_dim = 1024
# Dense equal-parameter baseline (≈ same number of params as sparse with 1024 units)
hidden_dim_dense = 447
num_classes = 10
num_epochs = 20   
lr = 1e-3


def build_model(model_name: str, p_inter: float):
    """Return the model specified by its name."""
    if model_name == "dense":
        # Fully-connected baseline (no p_inter used)
        return DenseSNN(input_dim, hidden_dim_dense, num_classes)
    elif model_name == "index":
        # Index-based sparse SNN
        return IndexSNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_groups=8,
            p_intra=1.0,
            p_inter=p_inter,
        )
    elif model_name == "random":
        # Random-group sparse SNN
        return RandomSNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_groups=8,
            p_intra=1.0,
            p_inter=p_inter,
        )
    elif model_name == "mixer":
        # Mixer-group sparse SNN
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


def train_one_epoch(model, loader, optimizer, device, epoch_idx: int):
    model.train()
    total = 0
    correct = 0

    for batch_idx, (images, labels) in enumerate(loader):
        # Move labels to device (images stay on CPU for encoding)
        labels = labels.to(device, non_blocking=True)

        # Convert images to spike trains over T time steps
        spikes = rate_encode(images, T).to(device)   # [T, B, 784]

        # Forward + backward pass
        optimizer.zero_grad()
        spk_counts = model(spikes)        # [B, num_classes]
        loss = nn.CrossEntropyLoss()(spk_counts, labels)
        loss.backward()
        optimizer.step()

        # Predictions based on spike counts
        preds = spk_counts.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    return acc


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0

    for batch_idx, (images, labels) in enumerate(loader):
        # Move labels to device
        labels = labels.to(device, non_blocking=True)

        # Convert images to spike trains over T time steps
        spikes = rate_encode(images, T).to(device)
        spk_counts = model(spikes)

        # Compute accuracy
        preds = spk_counts.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    return acc


@torch.no_grad()
def compute_firing_rates(model, loader, device):
    """Compute mean firing rates per hidden layer and overall (hidden layers only)."""
    model.eval()

    total_samples = 0

    l1_sum = None
    l2_sum = None
    l3_sum = None

    for images, _ in loader:
        B = images.size(0)
        total_samples += B

        # Encode on CPU, then move spikes to device
        spikes = rate_encode(images, T).to(device)

        # Forward pass with hidden spikes
        spk_out_sum, hidden_spikes = model(spikes, return_hidden_spikes=True)

        # hidden_spikes["layerX"]: [B, hidden_dim]  (sum of spikes over time)
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

    # Normalize by total time steps * total samples
    denom = T * total_samples
    l1_rate_per_neuron = l1_sum / denom
    l2_rate_per_neuron = l2_sum / denom
    l3_rate_per_neuron = l3_sum / denom

    # Overall means (hidden only)
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

    # Select which model to train
    parser.add_argument(
        "--model",
        type=str,
        default="dense",
        choices=["dense", "index", "random", "mixer"],
        help="Select model type.",
    )

    # Set number of training epochs
    parser.add_argument(
        "--epochs",
        type=int,
        default=num_epochs,
        help="Number of training epochs.",
    )

    # Inter-group probability p' for sparse models
    parser.add_argument(
        "--p_inter",
        type=float,
        default=0.15,
        help="Inter-group connection probability p' for sparse models (Index, Random, Mixer). "
             "Ignored for the dense model.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    device = select_device()

    print(f"Selected model: {args.model}")

    # DataLoaders
    train_loader, test_loader = get_fashion_loaders(batch_size)

    # Build model
    model = build_model(args.model, p_inter=args.p_inter).to(device)

    # Optimizer: Adam
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    # Training loop
    for epoch in range(1, args.epochs + 1):
        # Train for one epoch 
        train_one_epoch(model, train_loader, optimizer, device, epoch)

        # Evaluate on the test set
        test_acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch:02d} | test_acc={test_acc:.4f}")

    # ---- Firing rate analysis after training ----
    rates = compute_firing_rates(model, test_loader, device)
    print("Average firing rates (test set):")
    print(f"  Layer 1 mean rate:        {rates['layer1_mean']:.6f}")
    print(f"  Layer 2 mean rate:        {rates['layer2_mean']:.6f}")
    print(f"  Layer 3 mean rate:        {rates['layer3_mean']:.6f}")
    print(f"  Overall hidden mean rate: {rates['overall_hidden_mean']:.6f}")


if __name__ == "__main__":
    main()
