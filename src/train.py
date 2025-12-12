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


# defaults (override μέσω CLI)
batch_size = 256
T = 50
input_dim = 28 * 28
hidden_dim = 1024
hidden_dim_dense = 447
num_classes = 10
num_epochs = 20
lr = 1e-3

# DST
global_step = 0
UPDATE_INTERVAL = 1000
cp_mode = "set"
cg_mode = "hebb"

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
        chosen = active_idx[perm]
        prune_mask = torch.zeros_like(mask_cpu, dtype=torch.bool)
        prune_mask[chosen[:, 0], chosen[:, 1]] = True
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
    total_batches = len(loader)
    total = 0
    correct = 0

    for batch_idx, (images, labels) in enumerate(loader):
        # progress bar
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
    """
    Μετράει πόσο 'μαζεμένα' στο χρόνο είναι τα spikes ανά group (IndexSNN)
    χρησιμοποιώντας temporal variance.

    Μικρότερη variance => πιο συγχρονισμένα spikes.
    """
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
        B = images.size(0)
        spikes = rate_encode(images, T).to(device)  # [T, B, 784]

        # εδώ θέλουμε πλήρες time-series από το IndexSNN
        _, hidden_time = model(spikes, return_time_series=True)
        # hidden_time["layerX"]: [T, B, H]

        for layer_name in ["layer1", "layer2", "layer3"]:
            spk_time = hidden_time[layer_name]  # [T, B, H]
            T_steps, _, H = spk_time.shape
            group_size = H // num_groups

            t_idx = torch.arange(T_steps, device=spk_time.device, dtype=torch.float32)

            for g in range(num_groups):
                start = g * group_size
                end = (g + 1) * group_size

                group_spikes = spk_time[:, :, start:end]  # [T, B, group_size]
                s_t = group_spikes.sum(dim=(1, 2))       # [T]

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
    """
    Measures temporal coherence of inputs from different groups using pairwise correlation.
    
    Key idea: When a neuron receives inputs from multiple groups processing different features,
    those groups spike asynchronously. This metric measures how temporally aligned the spike
    patterns from different groups are.
    
    Coherence metric: For each neuron, compute pairwise temporal correlations between the
    spike activity of its connected groups. High coherence = groups spike in synchrony.
    Low coherence = groups spike at different, uncorrelated times.
    
    Expected results:
    - Low p_inter: neurons connect mainly to same group -> N/A or high coherence
    - High p_inter: neurons connect to many groups -> low coherence -> lower firing rates
    """
    if not isinstance(model, IndexSNN):
        raise TypeError("Requires IndexSNN")
    
    model.eval()
    
    for images, _ in loader:
        B = images.size(0)
        spikes_input = rate_encode(images, T).to(device)
        _, hidden_time = model(spikes_input, return_time_series=True)
        
        results = {}
        
        for layer_name, fc_layer, source_name in [
            ("layer2", model.fc2, "layer1"),
            ("layer3", model.fc3, "layer2")
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
                
                if len(unique_groups) >= 2:
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


def parse_args():
    p = argparse.ArgumentParser(description="Train SNN on Fashion-MNIST.")
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

    num_groups = 8 if args.model in ["index", "random", "mixer"] else "N/A"

    print(
        f"[CONFIG] model={args.model} | sparsity={args.sparsity_mode} | "
        f"T={T} | epochs={args.epochs} | batch={batch_size} | "
        f"groups={num_groups} | p_inter={args.p_inter}"
    )

    train_loader, test_loader = get_fashion_loaders(batch_size)
    model = build_model(args.model, args.p_inter).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    use_dst = (args.sparsity_mode == "dynamic")

    for epoch in range(1, args.epochs + 1):
        train_acc = train_one_epoch(model, train_loader, optimizer, device, epoch, use_dst)
        test_acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch:02d} | train_acc={train_acc:.4f} | test_acc={test_acc:.4f}")

    # firing rates
    rates = compute_firing_rates(model, test_loader, device)
    print("Average firing rates:")
    print(f"  L1: {rates['layer1_mean']:.6f}")
    print(f"  L2: {rates['layer2_mean']:.6f}")
    print(f"  L3: {rates['layer3_mean']:.6f}")
    print(f"  Overall: {rates['overall_hidden_mean']:.6f}")

    # synchrony metric μόνο για IndexSNN
    if isinstance(model, IndexSNN):
        ng = model.fc1.num_groups
        sync = compute_group_synchrony(model, test_loader, device, ng)
        print("Group-wise temporal synchrony (mean variance per group):")
        print(f"  Layer 1 mean temporal var: {sync['layer1_mean_temporal_var']:.6f}")
        print(f"  Layer 2 mean temporal var: {sync['layer2_mean_temporal_var']:.6f}")
        print(f"  Layer 3 mean temporal var: {sync['layer3_mean_temporal_var']:.6f}")
        print("  (Lower variance => more synchronized spikes within groups)")
        
        corr_data = compute_cross_group_coherence(model, test_loader, device)
        if corr_data:
            print("\nCross-group temporal coherence analysis:")
            for layer_name in ["layer2", "layer3"]:
                if layer_name in corr_data:
                    d = corr_data[layer_name]
                    print(f"\n  {layer_name.capitalize()}:")
                    print(f"    Avg cross-group ratio: {d['avg_cross_group_ratio']:.4f}")
                    print(f"    Avg temporal coherence: {d['avg_coherence']:.4f}")
                    print(f"    Avg firing rate: {d['avg_firing_rate']:.6f}")
                    print(f"    Correlation (cross-group % vs coherence): r={d['corr_crossgroup_coherence']:.3f}")
                    print(f"    Correlation (coherence vs firing rate): r={d['corr_coherence_firingrate']:.3f}")
            print("\n  Interpretation:")
            print("    Negative r (cross-group vs coherence): more cross-group -> less synchronized inputs")
            print("    Positive r (coherence vs firing): more synchronized inputs -> higher firing rates")


if __name__ == "__main__":
    main()