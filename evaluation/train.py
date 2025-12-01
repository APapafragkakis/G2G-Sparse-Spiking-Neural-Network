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


# Shared hyperparameters
batch_size = 256
T = 50
input_dim = 28 * 28
hidden_dim = 1024
# Dense baseline with roughly the same number of parameters
hidden_dim_dense = 447
num_classes = 10
num_epochs = 20
lr = 1e-3

# Global state for Dynamic Sparse Training (DST)
global_step = 0
# Number of training steps between DST updates
UPDATE_INTERVAL = 1000

# Modes for pruning (C_P) and growth (C_G)
# cp_mode: "set", "random", "hebb", "three"
#   "set"    -> magnitude-based pruning
#   "random" -> random pruning
#   "hebb"   -> Hebbian (CH-based) pruning
#   "three"  -> SET + Random + Hebbian in sequence (hybrid)
# cg_mode: "hebb", "random"
cp_mode = "set"
cg_mode = "hebb"

# Hebbian buffer:
# for each sparse layer we store the latest batch of
# pre- and post-synaptic activations
hebb_buffer = {
    "fc1": None,  # pre: input spikes,  post: layer1 spikes
    "fc2": None,  # pre: layer1 spikes, post: layer2 spikes
    "fc3": None,  # pre: layer2 spikes, post: layer3 spikes
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

    raise ValueError(f"Unknown model type: {model_name}")


def update_hebb_buffer(input_spikes: torch.Tensor, activations: dict):
    """
    Store the latest batch of pre/post activations for each MixerSparseLinear layer.

    input_spikes: [T, B, input_dim]
    activations:  dict with keys 'layer1', 'layer2', 'layer3',
                  each tensor has shape [T, B, H]
    """
    global hebb_buffer

    # Keep these on CPU to reduce pressure on GPU memory
    pre_fc1 = input_spikes.detach().cpu()         # [T, B, 784]
    post_fc1 = activations["layer1"].detach().cpu()

    pre_fc2 = activations["layer1"].detach().cpu()
    post_fc2 = activations["layer2"].detach().cpu()

    pre_fc3 = activations["layer2"].detach().cpu()
    post_fc3 = activations["layer3"].detach().cpu()

    hebb_buffer["fc1"] = {"pre": pre_fc1, "post": post_fc1}
    hebb_buffer["fc2"] = {"pre": pre_fc2, "post": post_fc2}
    hebb_buffer["fc3"] = {"pre": pre_fc3, "post": post_fc3}


def compute_ch_matrix(pre_batch: torch.Tensor, post_batch: torch.Tensor) -> torch.Tensor:
    """
    Compute CH(i,j) = cosine similarity between pre and post neuron activation vectors.

    pre_batch:  [T, B, N_in]
    post_batch: [T, B, N_out]

    Returns:
        ch: [N_out, N_in] cosine similarity matrix.
    """
    T_steps, B, N_in = pre_batch.shape
    _, _, N_out = post_batch.shape

    # Flatten time and batch into a single axis
    pre_flat = pre_batch.reshape(T_steps * B, N_in)     # [TB, N_in]
    post_flat = post_batch.reshape(T_steps * B, N_out)  # [TB, N_out]

    # Each neuron gets a vector of length TB
    pre_vecs = pre_flat.transpose(0, 1)    # [N_in,  TB]
    post_vecs = post_flat.transpose(0, 1)  # [N_out, TB]

    eps = 1e-8
    pre_norm = pre_vecs / (pre_vecs.norm(dim=1, keepdim=True) + eps)
    post_norm = post_vecs / (post_vecs.norm(dim=1, keepdim=True) + eps)

    # Cosine similarity: post_norm @ pre_norm^T
    ch = torch.matmul(post_norm, pre_norm.transpose(0, 1))  # [N_out, N_in]
    return ch


def _grow_connections(
    mask_cpu: torch.Tensor,
    ch_cpu: torch.Tensor,
    num_to_grow: int,
    cg_mode_local: str,
):
    """
    Growth step.

    mask_cpu:     layer mask on CPU
    ch_cpu:       [out_features, in_features] cosine similarity matrix
    num_to_grow:  how many new edges to activate
    cg_mode_local: "hebb" or "random"
    """
    inactive_cpu = ~mask_cpu.bool()
    num_inactive = inactive_cpu.sum().item()
    if num_inactive == 0 or num_to_grow <= 0:
        return mask_cpu, 0

    num_to_grow = min(num_to_grow, num_inactive)
    inactive_idx = inactive_cpu.nonzero(as_tuple=False)  # [N_inactive, 2]

    if cg_mode_local == "hebb":
        # Hebbian growth: pick edges with highest CH among inactive
        inactive_scores = ch_cpu[inactive_cpu]  # 1D tensor
        n_eff = min(num_to_grow, inactive_scores.numel())
        if n_eff < 1:
            return mask_cpu, 0
        _, top_idx = torch.topk(inactive_scores, k=n_eff, largest=True)
        grow_idx = inactive_idx[top_idx]
    else:
        # Random growth
        n_eff = num_to_grow
        perm = torch.randperm(inactive_idx.size(0))[:n_eff]
        grow_idx = inactive_idx[perm]

    mask_cpu[grow_idx[:, 0], grow_idx[:, 1]] = 1
    return mask_cpu, n_eff


def dst_update_layer_three_prune_hebb_growth(
    layer: MixerSparseLinear,
    layer_name: str,
    prune_frac: float,
    cg_mode_local: str,
):
    """
    Hybrid DST update for a MixerSparseLinear layer.

    Pruning (three criteria in sequence on active edges):
        1) SET    (magnitude-based)
        2) Random
        3) Hebbian (smallest CH(i,j))

    Growth:
        Hebbian (CH-based) or Random on inactive edges.
    """
    global hebb_buffer

    buf = hebb_buffer.get(layer_name, None)
    if buf is None or "pre" not in buf or "post" not in buf:
        return

    pre_batch = buf["pre"]   # [T, B, N_in]
    post_batch = buf["post"] # [T, B, N_out]

    if pre_batch.shape[-1] != layer.in_features or post_batch.shape[-1] != layer.out_features:
        return

    # Cosine similarity on CPU
    ch_cpu = compute_ch_matrix(pre_batch, post_batch)  # [out_features, in_features]

    weight = layer.weight.data
    mask = layer.mask
    device = weight.device

    mask_cpu = mask.detach().cpu()
    w_cpu = weight.detach().cpu()

    active_cpu = mask_cpu.bool()
    num_active = active_cpu.sum().item()
    if num_active == 0:
        return

    total_to_prune = int(prune_frac * num_active)
    if total_to_prune < 1:
        return

    # Split pruning budget into three parts
    base = total_to_prune // 3
    n_set = base
    n_rand = base
    n_hebb = total_to_prune - n_set - n_rand

    total_pruned = 0

    # 1) SET (magnitude-based pruning)
    active_cpu = mask_cpu.bool()
    if n_set > 0 and active_cpu.sum().item() > 0:
        active_weights = w_cpu[active_cpu].abs()
        n_set_eff = min(n_set, active_weights.numel())
        if n_set_eff > 0:
            thresh, _ = torch.kthvalue(active_weights, n_set_eff)
            prune_mask_set = active_cpu & (w_cpu.abs() <= thresh)
            num_pruned_set = prune_mask_set.sum().item()
            mask_cpu[prune_mask_set] = 0
            total_pruned += num_pruned_set

    # 2) Random pruning
    active_cpu = mask_cpu.bool()
    if n_rand > 0 and active_cpu.sum().item() > 0:
        active_idx = active_cpu.nonzero(as_tuple=False)  # [N_active, 2]
        n_rand_eff = min(n_rand, active_idx.size(0))
        if n_rand_eff > 0:
            perm = torch.randperm(active_idx.size(0))[:n_rand_eff]
            rand_idx = active_idx[perm]
            mask_cpu[rand_idx[:, 0], rand_idx[:, 1]] = 0
            total_pruned += n_rand_eff

    # 3) Hebbian pruning (lowest CH among active)
    active_cpu = mask_cpu.bool()
    if n_hebb > 0 and active_cpu.sum().item() > 0:
        active_scores = ch_cpu[active_cpu]
        n_hebb_eff = min(n_hebb, active_scores.numel())
        if n_hebb_eff > 0:
            thresh_hebb, _ = torch.kthvalue(active_scores, n_hebb_eff)
            prune_mask_hebb = active_cpu & (ch_cpu <= thresh_hebb)
            num_pruned_hebb = prune_mask_hebb.sum().item()
            mask_cpu[prune_mask_hebb] = 0
            total_pruned += num_pruned_hebb

    # Growth step
    if total_pruned <= 0:
        mask.copy_(mask_cpu.to(device))
        weight[~mask.bool()] = 0.0
        return

    mask_cpu, _ = _grow_connections(
        mask_cpu,
        ch_cpu,
        total_pruned,
        cg_mode_local,
    )

    mask.copy_(mask_cpu.to(device))
    weight[~mask.bool()] = 0.0


def dst_update_layer_cp_cg_single(
    layer: MixerSparseLinear,
    layer_name: str,
    prune_frac: float,
    cp_mode_local: str,
    cg_mode_local: str,
):
    """
    DST update for a MixerSparseLinear layer with a single pruning
    rule (cp_mode_local) and a single growth rule (cg_mode_local).

    cp_mode_local: "set", "random", or "hebb"
    cg_mode_local: "random" or "hebb"
    """
    global hebb_buffer

    buf = hebb_buffer.get(layer_name, None)
    if buf is None or "pre" not in buf or "post" not in buf:
        return

    pre_batch = buf["pre"]
    post_batch = buf["post"]

    if pre_batch.shape[-1] != layer.in_features or post_batch.shape[-1] != layer.out_features:
        return

    weight = layer.weight.data
    mask = layer.mask
    device = weight.device

    mask_cpu = mask.detach().cpu()
    w_cpu = weight.detach().cpu()

    active_cpu = mask_cpu.bool()
    num_active = active_cpu.sum().item()
    if num_active == 0:
        return

    total_to_prune = int(prune_frac * num_active)
    if total_to_prune < 1:
        return

    # Compute CH only if Hebbian is used in pruning or growth
    ch_cpu = None
    if cp_mode_local == "hebb" or cg_mode_local == "hebb":
        ch_cpu = compute_ch_matrix(pre_batch, post_batch)

    # --- Pruning (C_P) ---
    num_pruned = 0

    if cp_mode_local == "set":
        # Magnitude-based pruning
        active_weights = w_cpu[active_cpu].abs()
        n_eff = min(total_to_prune, active_weights.numel())
        if n_eff > 0:
            thresh, _ = torch.kthvalue(active_weights, n_eff)
            prune_mask = active_cpu & (w_cpu.abs() <= thresh)
            num_pruned = prune_mask.sum().item()
            mask_cpu[prune_mask] = 0

    elif cp_mode_local == "random":
        active_idx = active_cpu.nonzero(as_tuple=False)
        n_eff = min(total_to_prune, active_idx.size(0))
        if n_eff > 0:
            perm = torch.randperm(active_idx.size(0))[:n_eff]
            rand_idx = active_idx[perm]
            mask_cpu[rand_idx[:, 0], rand_idx[:, 1]] = 0
            num_pruned = n_eff

    elif cp_mode_local == "hebb":
        if ch_cpu is None:
            return
        active_scores = ch_cpu[active_cpu]
        n_eff = min(total_to_prune, active_scores.numel())
        if n_eff > 0:
            # Remove edges with lowest CH
            thresh, _ = torch.kthvalue(active_scores, n_eff)
            prune_mask = active_cpu & (ch_cpu <= thresh)
            num_pruned = prune_mask.sum().item()
            mask_cpu[prune_mask] = 0

    if num_pruned <= 0:
        mask.copy_(mask_cpu.to(device))
        weight[~mask.bool()] = 0.0
        return

    # --- Growth (C_G) ---
    if cg_mode_local not in ["random", "hebb"]:
        cg_mode_local = "random"

    if ch_cpu is None:
        # Dummy CH if we only need random growth
        ch_cpu = torch.zeros_like(mask_cpu, dtype=torch.float32)

    mask_cpu, _ = _grow_connections(
        mask_cpu,
        ch_cpu,
        num_pruned,
        cg_mode_local,
    )

    mask.copy_(mask_cpu.to(device))
    weight[~mask.bool()] = 0.0


def dst_step(model: nn.Module, prune_frac: float = 0.025):
    """
    Apply one DST update step to all MixerSparseLinear layers.

    Modes:
        cp_mode in {"set", "random", "hebb"} -> single CP/CG rule.
        cp_mode == "three"                  -> SET + Random + Hebbian hybrid.
    """
    for name, module in model.named_modules():
        if not isinstance(module, MixerSparseLinear):
            continue

        short_name = name.split(".")[-1]
        if short_name not in hebb_buffer:
            continue

        if cp_mode in ["set", "random", "hebb"]:
            dst_update_layer_cp_cg_single(
                module,
                short_name,
                prune_frac,
                cp_mode,
                cg_mode,
            )
        elif cp_mode == "three":
            dst_update_layer_three_prune_hebb_growth(
                module,
                short_name,
                prune_frac,
                cg_mode,
            )
        else:
            # Fallback: treat as single-rule SET pruning
            dst_update_layer_cp_cg_single(
                module,
                short_name,
                prune_frac,
                "set",
                cg_mode,
            )

    print("[DST] step executed")


def train_one_epoch(model, loader, optimizer, device, epoch_idx: int, use_dst: bool):
    """
    Train the model for one epoch.

    In dynamic mode with MixerSNN:
        - request activations from the model
        - update Hebbian buffers
        - apply DST every UPDATE_INTERVAL steps
    """
    global global_step
    model.train()
    total = 0
    correct = 0

    for batch_idx, (images, labels) in enumerate(loader):
        labels = labels.to(device, non_blocking=True)

        # Encode images into spike trains
        spikes = rate_encode(images, T).to(device)   # [T, B, 784]

        optimizer.zero_grad()

        if use_dst and isinstance(model, MixerSNN):
            # Ask the model to return per-layer activations for Hebbian updates
            spk_counts, activations = model(spikes, return_activations=True)
            update_hebb_buffer(spikes, activations)
        else:
            spk_counts = model(spikes)  # [B, num_classes]

        loss = nn.CrossEntropyLoss()(spk_counts, labels)
        loss.backward()
        optimizer.step()

        # Dynamic Sparse Training step (MixerSNN only)
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
    """Evaluate the model and return accuracy."""
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
    Compute mean firing rates per hidden layer and overall
    (only hidden layers are considered).
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
        help=(
            "Inter-group connection probability p' for sparse models "
            "(Index, Random, Mixer). Ignored for the dense model."
        ),
    )

    parser.add_argument(
        "--sparsity_mode",
        type=str,
        default="static",
        choices=["static", "dynamic"],
        help=(
            "Sparsity mode for sparse models: "
            "'static' = only initial structured sparsity, "
            "'dynamic' = apply Dynamic Sparse Training (DST)."
        ),
    )

    parser.add_argument(
        "--cp",
        type=str,
        default="set",
        choices=["set", "random", "hebb", "three"],
        help=(
            "Pruning criterion C_P: "
            "'set'    = magnitude-based pruning, "
            "'random' = random pruning, "
            "'hebb'   = Hebbian pruning (CH-based), "
            "'three'  = SET + Random + Hebbian in sequence."
        ),
    )

    parser.add_argument(
        "--cg",
        type=str,
        default="hebb",
        choices=["hebb", "random"],
        help=(
            "Growth criterion C_G: "
            "'hebb'   = Hebbian (CH-based) growth on inactive edges, "
            "'random' = random growth on inactive edges."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()
    device = select_device()

    global cp_mode, cg_mode
    cp_mode = args.cp
    cg_mode = args.cg

    print(f"Selected model: {args.model}")
    print(f"Sparsity mode: {args.sparsity_mode}")
    print(f"C_P (prune): {cp_mode}")
    print(f"C_G (grow):  {cg_mode}")

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
