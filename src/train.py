import os
import sys
import argparse
from datetime import datetime

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
from utils.encoding import encode_input
import warnings

warnings.filterwarnings("ignore", message=".*aten::lerp.Scalar_out.*")

# Global log file handle
log_file = None

def log_print(*args, **kwargs):
    """Print to both console and log file"""
    # Print to console
    print(*args, **kwargs)
    # Write to log file if it exists
    if log_file is not None:
        print(*args, **kwargs, file=log_file, flush=True)

try:
    import torch_directml
    has_dml = True
except ImportError:
    has_dml = False


def select_device():
    if has_dml:
        device = torch_directml.device()
        log_print(f"Using DirectML device: {device}")
        return device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        log_print(f"Using GPU: {torch.cuda.get_device_name(0)} | CUDA {torch.version.cuda}")
        return device
    log_print("No GPU backend available — using CPU.")
    return torch.device("cpu")


# Default training parameters (can be overridden via CLI)
batch_size = 256
T = 50
input_dim = 28 * 28
hidden_dim = 1024
hidden_dim_dense = 447
num_classes = 10
num_epochs = 20
lr = 1e-3

# Input encoding configuration (set from CLI)
enc_mode = "current"  # "current" (analog/current injection) or "rate" (Bernoulli rate code)
enc_scale = 1.0
enc_bias = 0.0

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
    log_print("[DST] step executed")


def _make_input_sequence(images, device):
    return encode_input(
        images,
        T,
        mode=enc_mode,
        scale=enc_scale,
        bias=enc_bias,
    ).to(device)


def train_one_epoch(model, loader, optimizer, device, epoch_idx, use_dst):
    global global_step
    model.train()
    total = 0
    correct = 0
    total_batches = len(loader)
    for batch_idx, (images, labels) in enumerate(loader):
        # Progress bar - ONLY to console (not to log file)
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
        x_seq = _make_input_sequence(images, device)
        optimizer.zero_grad()
        if use_dst and isinstance(model, MixerSNN):
            spk_counts, acts = model(x_seq, return_activations=True)
            update_hebb_buffer(x_seq, acts)
        else:
            spk_counts = model(x_seq)
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
        x_seq = _make_input_sequence(images, device)
        spk_counts = model(x_seq)
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
        x_seq = _make_input_sequence(images, device)
        _, hidden_spikes = model(x_seq, return_hidden_spikes=True)
        b1 = hidden_spikes["layer1"].sum(dim=0)
        b2 = hidden_spikes["layer2"].sum(dim=0)
        b3 = hidden_spikes["layer3"].sum(dim=0)
        if l1_sum is None:
            l1_sum, l2_sum, l3_sum = b1, b2, b3
        else:
            l1_sum += b1
            l2_sum += b2
            l3_sum += b3
    denom = T * total_samples
    r1 = l1_sum / denom
    r2 = l2_sum / denom
    r3 = l3_sum / denom
    allr = torch.cat([r1, r2, r3])
    return {
        "layer1_mean": r1.mean().item(),
        "layer2_mean": r2.mean().item(),
        "layer3_mean": r3.mean().item(),
        "overall_hidden_mean": allr.mean().item(),
    }


@torch.no_grad()
def compute_group_synchrony(model, loader, device, num_groups: int):
    if not isinstance(model, IndexSNN):
        raise TypeError("compute_group_synchrony is implemented only for IndexSNN")
    model.eval()
    stats = {
        "layer1_var_sum": 0.0,
        "layer2_var_sum": 0.0,
        "layer3_var_sum": 0.0,
        "layer1_groups": 0,
        "layer2_groups": 0,
        "layer3_groups": 0,
    }
    for images, _ in loader:
        x_seq = _make_input_sequence(images, device)
        _, hidden_time = model(x_seq, return_time_series=True)
        for layer_name in ["layer1", "layer2", "layer3"]:
            spk_time = hidden_time[layer_name]
            T_steps, _, H = spk_time.shape
            group_size = H // num_groups
            t_idx = torch.arange(T_steps, device=spk_time.device, dtype=torch.float32)
            for g in range(num_groups):
                start = g * group_size
                end = (g + 1) * group_size
                group_spikes = spk_time[:, :, start:end]
                s_t = group_spikes.sum(dim=(1, 2))
                if s_t.sum().item() == 0:
                    continue
                w = s_t / s_t.sum()
                mu = (w * t_idx).sum()
                var_g = (((t_idx - mu) ** 2) * w).sum().item()
                key_v = f"{layer_name}_var_sum"
                key_c = f"{layer_name}_groups"
                stats[key_v] += var_g
                stats[key_c] += 1
    sync = {}
    for layer_name in ["layer1", "layer2", "layer3"]:
        key_v = f"{layer_name}_var_sum"
        key_c = f"{layer_name}_groups"
        if stats[key_c] > 0:
            sync[f"{layer_name}_mean_temporal_var"] = stats[key_v] / stats[key_c]
        else:
            sync[f"{layer_name}_mean_temporal_var"] = float("nan")
    return sync


@torch.no_grad()
def compute_cross_group_coherence(model, loader, device):
    if not isinstance(model, IndexSNN):
        raise TypeError("Requires IndexSNN")
    model.eval()
    for images, _ in loader:
        B = images.size(0)
        x_seq = _make_input_sequence(images, device)
        _, hidden_time = model(x_seq, return_time_series=True)
        results = {}
        for layer_name, fc_layer, source_name in [
            ("layer2", model.fc2, "layer1"),
            ("layer3", model.fc3, "layer2"),
        ]:
            mask = fc_layer.mask
            source_spikes = hidden_time[source_name]
            T_steps, _, N_in = source_spikes.shape
            N_out = mask.size(0)
            group_size_in = N_in // model.fc1.num_groups
            num_groups = model.fc1.num_groups
            coherences = []
            cross_group_ratios = []
            firing_rates = []
            sampled_neurons = torch.arange(N_out, device=device)
            for neuron_idx in sampled_neurons:
                neuron_idx = neuron_idx.item()
                conn_mask = mask[neuron_idx].bool()
                num_conn = conn_mask.sum().item()
                if num_conn < 2:
                    continue
                connected_indices = torch.where(conn_mask)[0]
                connected_groups = connected_indices // group_size_in
                unique_groups = torch.unique(connected_groups)
                if len(unique_groups) < 2:
                    continue
                neuron_group = neuron_idx // (N_out // num_groups)
                cross_group = (connected_groups != neuron_group).sum().item()
                cross_ratio = cross_group / num_conn
                group_activity = torch.zeros(T_steps, len(unique_groups), device=device)
                for g_idx, group_id in enumerate(unique_groups):
                    group_neurons = connected_indices[connected_groups == group_id]
                    group_spikes = source_spikes[:, :, group_neurons].sum(dim=(1, 2))
                    group_activity[:, g_idx] = (group_spikes > 0).float()
                ga = group_activity - group_activity.mean(dim=0, keepdim=True)
                cov = (ga.T @ ga) / T_steps
                var = torch.diag(cov)
                denom = torch.sqrt(var.unsqueeze(0) * var.unsqueeze(1) + 1e-8)
                corr_mat = cov / denom
                triu_mask = torch.triu(torch.ones_like(corr_mat, dtype=torch.bool), diagonal=1)
                if triu_mask.sum() > 0:
                    coherence = corr_mat[triu_mask].mean().item()
                    coherences.append(coherence)
                    cross_group_ratios.append(cross_ratio)
                    fr = hidden_time[layer_name][:, :, neuron_idx].sum().item() / (T_steps * B)
                    firing_rates.append(fr)
            if len(coherences) > 10:
                import numpy as np
                coh = np.array(coherences)
                cgr = np.array(cross_group_ratios)
                fr = np.array(firing_rates)
                corr_cg_coh = np.corrcoef(cgr, coh)[0, 1]
                corr_coh_fr = np.corrcoef(coh, fr)[0, 1]
                results[layer_name] = {
                    "avg_cross_group_ratio": cgr.mean(),
                    "avg_coherence": coh.mean(),
                    "avg_firing_rate": fr.mean(),
                    "corr_crossgroup_coherence": corr_cg_coh,
                    "corr_coherence_firingrate": corr_coh_fr,
                }
        break
    return results


@torch.no_grad()
def compute_within_group_correlation(model, loader, device):
    if not isinstance(model, IndexSNN):
        raise TypeError("Requires IndexSNN")
    import numpy as np
    model.eval()
    for images, _ in loader:
        B = images.size(0)
        x_seq = _make_input_sequence(images, device)
        _, hidden_time = model(x_seq, return_time_series=True)
        results = {}
        num_groups = model.fc1.num_groups
        for layer_name, fc_layer in [
            ("layer1", model.fc1),
            ("layer2", model.fc2),
            ("layer3", model.fc3),
        ]:
            spk_ts = hidden_time[layer_name]
            T_steps, _, H = spk_ts.shape
            group_size = H // num_groups
            group_correlations = []
            group_entropies = []
            for g in range(num_groups):
                start_idx = g * group_size
                end_idx = (g + 1) * group_size
                group_spikes = spk_ts[:, :, start_idx:end_idx]
                group_flat = group_spikes.reshape(T_steps * B, group_size)
                if group_flat.sum() < 10:
                    continue
                sample_size = min(30, group_size)
                sampled_idx = torch.randperm(group_size, device=device)[:sample_size]
                group_sample = group_flat[:, sampled_idx]
                corrs = []
                for i in range(sample_size):
                    for j in range(i + 1, sample_size):
                        n1 = group_sample[:, i].float()
                        n2 = group_sample[:, j].float()
                        if n1.std() > 1e-8 and n2.std() > 1e-8:
                            n1_c = n1 - n1.mean()
                            n2_c = n2 - n2.mean()
                            corr = (n1_c * n2_c).mean() / (n1.std() * n2.std() + 1e-8)
                            corrs.append(corr.item())
                if len(corrs) > 0:
                    group_correlations.append(np.mean(corrs))
                if layer_name == "layer2":
                    input_group_counts = torch.zeros(num_groups, device=device)
                    input_group_size = fc_layer.in_features // num_groups
                    for local_idx in range(group_size):
                        global_idx = start_idx + local_idx
                        conn_mask = fc_layer.mask[global_idx].bool()
                        connected_idx = torch.where(conn_mask)[0]
                        connected_groups = connected_idx // input_group_size
                        for ig in range(num_groups):
                            input_group_counts[ig] += (connected_groups == ig).sum()
                    probs = input_group_counts / (input_group_counts.sum() + 1e-8)
                    probs = probs[probs > 0]
                    if len(probs) > 0:
                        entropy = -(probs * torch.log2(probs + 1e-8)).sum().item()
                        group_entropies.append(entropy)
            if len(group_correlations) > 0:
                results[layer_name] = {
                    "mean_correlation": np.mean(group_correlations),
                    "std_correlation": np.std(group_correlations),
                    "min_correlation": np.min(group_correlations),
                    "max_correlation": np.max(group_correlations),
                }
            if len(group_entropies) > 0 and layer_name == "layer2":
                results[layer_name]["mean_input_entropy"] = np.mean(group_entropies)
                results[layer_name]["std_input_entropy"] = np.std(group_entropies)
        break
    return results


def parse_args():
    p = argparse.ArgumentParser(description="Train SNN models on multiple datasets.")
    p.add_argument("--dataset", type=str, default="fashionmnist", choices=["fashionmnist", "cifar10", "cifar100"])
    p.add_argument("--model", type=str, default="dense", choices=["dense", "index", "random", "mixer"])
    p.add_argument("--epochs", type=int, default=num_epochs)
    p.add_argument("--p_inter", type=float, default=0.15)
    p.add_argument("--sparsity_mode", type=str, default="static", choices=["static", "dynamic"])
    p.add_argument("--cp", type=str, default="set", choices=["set", "random", "hebb"])
    p.add_argument("--cg", type=str, default="hebb", choices=["hebb", "random"])
    p.add_argument("--T", type=int, default=T)
    p.add_argument("--batch_size", type=int, default=batch_size)
    p.add_argument("--enc", type=str, default="current", choices=["current", "rate"])
    p.add_argument("--enc_scale", type=float, default=1.0)
    p.add_argument("--enc_bias", type=float, default=0.0)
    p.add_argument("--log_dir", type=str, default="logs", help="Directory to save log files")
    return p.parse_args()


def main():
    global log_file
    args = parse_args()
    
    # Create log directory if it doesn't exist
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Create log file with timestamp and configuration
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{args.dataset}_{args.model}_{args.sparsity_mode}_T{args.T}_{timestamp}.log"
    log_path = os.path.join(args.log_dir, log_filename)
    log_file = open(log_path, 'w')
    
    log_print(f"Log file: {log_path}")
    log_print("=" * 70)
    
    device = select_device()
    global cp_mode, cg_mode, T, batch_size
    cp_mode = args.cp
    cg_mode = args.cg
    T = args.T
    batch_size = args.batch_size
    global enc_mode, enc_scale, enc_bias
    enc_mode = args.enc
    enc_scale = float(args.enc_scale)
    enc_bias = float(args.enc_bias)
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
    log_print(
        f"[CONFIG] dataset={args.dataset} | model={args.model} | "
        f"sparsity={args.sparsity_mode} | T={T} | epochs={args.epochs} | "
        f"batch={batch_size} | groups={num_groups} | p_inter={args.p_inter} | "
        f"enc={enc_mode} | enc_scale={enc_scale} | enc_bias={enc_bias}"
    )
    log_print("=" * 70)
    
    model = build_model(args.model, args.p_inter).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    use_dst = (args.sparsity_mode == "dynamic")
    
    for epoch in range(1, args.epochs + 1):
        train_acc = train_one_epoch(model, train_loader, optimizer, device, epoch, use_dst)
        test_acc = evaluate(model, test_loader, device)
        log_print(f"Epoch {epoch:02d} | train_acc={train_acc:.4f} | test_acc={test_acc:.4f}")
    
    log_print("\n" + "=" * 70)
    log_print("FINAL METRICS")
    log_print("=" * 70)
    
    rates = compute_firing_rates(model, test_loader, device)
    log_print("\nAverage firing rates:")
    log_print(f" L1: {rates['layer1_mean']:.6f}")
    log_print(f" L2: {rates['layer2_mean']:.6f}")
    log_print(f" L3: {rates['layer3_mean']:.6f}")
    log_print(f" Overall: {rates['overall_hidden_mean']:.6f}")
    
    if isinstance(model, IndexSNN):
        ng = model.fc1.num_groups
        sync = compute_group_synchrony(model, test_loader, device, ng)
        log_print("\nGroup-wise temporal synchrony (mean variance per group):")
        log_print(f" Layer 1 mean temporal var: {sync['layer1_mean_temporal_var']:.6f}")
        log_print(f" Layer 2 mean temporal var: {sync['layer2_mean_temporal_var']:.6f}")
        log_print(f" Layer 3 mean temporal var: {sync['layer3_mean_temporal_var']:.6f}")
        
        corr_data = compute_cross_group_coherence(model, test_loader, device)
        if corr_data:
            log_print("\nCross-group temporal coherence analysis:")
            for layer_name in ["layer2", "layer3"]:
                if layer_name in corr_data:
                    d = corr_data[layer_name]
                    log_print(f"\n {layer_name.capitalize()}:")
                    log_print(f"  Avg cross-group ratio: {d['avg_cross_group_ratio']:.4f}")
                    log_print(f"  Avg temporal coherence: {d['avg_coherence']:.4f}")
                    log_print(f"  Avg firing rate: {d['avg_firing_rate']:.6f}")
                    log_print(f"  Correlation (cross-group % vs coherence): r={d['corr_crossgroup_coherence']:.3f}")
                    log_print(f"  Correlation (coherence vs firing): r={d['corr_coherence_firingrate']:.3f}")
        
        import numpy as np
        log_print("\n" + "=" * 70)
        log_print("Within-Group Spike Correlation Analysis")
        log_print("=" * 70)
        wg_results = compute_within_group_correlation(model, test_loader, device)
        for layer_name in ["layer1", "layer2", "layer3"]:
            if layer_name in wg_results:
                d = wg_results[layer_name]
                log_print(f"\n{layer_name.upper()}:")
                log_print(f" Mean within-group correlation: {d['mean_correlation']:.4f} ± {d['std_correlation']:.4f}")
                log_print(f" Range: [{d['min_correlation']:.4f}, {d['max_correlation']:.4f}]")
                if "mean_input_entropy" in d:
                    log_print(f" Input feature diversity (entropy): {d['mean_input_entropy']:.4f} ± {d['std_input_entropy']:.4f}")
                    log_print(f" (max entropy = {np.log2(model.fc1.num_groups):.2f} bits)")
    
    log_print("\n" + "=" * 70)
    log_print(f"Training completed. Results saved to: {log_path}")
    log_print("=" * 70)
    
    # Close log file
    if log_file is not None:
        log_file.close()


if __name__ == "__main__":
    main()