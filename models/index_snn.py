import torch
from torch import nn
import torch.nn.functional as F

import snntorch as snn
from snntorch import surrogate


class IndexSparseLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_groups: int,
        p_intra: float = 1.0,
        p_inter: float = 0.15,
        bias: bool = True,
    ):
        super().__init__()

        if in_features % num_groups != 0 or out_features % num_groups != 0:
            raise ValueError(
                f"in_features ({in_features}) and out_features ({out_features}) "
                f"must both be divisible by num_groups ({num_groups})."
            )

        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.p_intra = float(p_intra)
        self.p_inter = float(p_inter)

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.bias = None

        mask = self._build_mask()
        self.register_buffer("mask", mask)

        self.reset_parameters()

    def _build_mask(self) -> torch.Tensor:
        in_group_size = self.in_features // self.num_groups
        out_group_size = self.out_features // self.num_groups

        in_idx = torch.arange(self.in_features)
        out_idx = torch.arange(self.out_features)

        in_group = in_idx // in_group_size
        out_group = out_idx // out_group_size

        same_group = (out_group[:, None] == in_group[None, :])

        probs = torch.full((self.out_features, self.in_features), self.p_inter)
        probs[same_group] = self.p_intra

        mask = torch.bernoulli(probs)
        return mask

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

        with torch.no_grad():
            full_fan_in = float(self.in_features)
            for out_idx in range(self.out_features):
                row_mask = self.mask[out_idx]
                fan_in_active = row_mask.sum().item()
                if fan_in_active > 0 and fan_in_active < full_fan_in:
                    scale = (full_fan_in / fan_in_active) ** 0.5
                    self.weight[out_idx, row_mask.bool()] *= scale

        if self.bias is not None:
            fan_in_eff = self.mask.sum(dim=1).float().mean().item()
            if fan_in_eff <= 0:
                fan_in_eff = self.in_features
            bound = 1.0 / fan_in_eff**0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        effective_weight = self.weight * self.mask
        return F.linear(input, effective_weight, self.bias)


class IndexSNN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_classes,
        num_groups,
        p_intra=1.0,
        p_inter=0.15,
    ):
        super().__init__()

        # Fully-connected layers with index-based sparse connectivity
        self.fc1 = IndexSparseLinear(
            in_features=input_dim,
            out_features=hidden_dim,
            num_groups=num_groups,
            p_intra=p_intra,
            p_inter=p_inter,
        )
        self.fc2 = IndexSparseLinear(
            in_features=hidden_dim,
            out_features=hidden_dim,
            num_groups=num_groups,
            p_intra=p_intra,
            p_inter=p_inter,
        )
        self.fc3 = IndexSparseLinear(
            in_features=hidden_dim,
            out_features=hidden_dim,
            num_groups=num_groups,
            p_intra=p_intra,
            p_inter=p_inter,
        )
        self.fc_out = nn.Linear(hidden_dim, num_classes)

        # LIF neuron parameters
        beta = 0.95
        spike_grad = surrogate.fast_sigmoid(slope=25)

        # Spiking (LIF) layers
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif_out = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(
        self,
        x_seq,
        return_hidden_spikes: bool = False,
        return_time_series: bool = False,
        return_activations: bool = False,
    ):
        # x_seq: [T, B, input_dim]
        T_steps, B, _ = x_seq.shape
        device = x_seq.device

        # Membrane potentials
        mem1 = torch.zeros(B, self.fc1.out_features, device=device)
        mem2 = torch.zeros(B, self.fc2.out_features, device=device)
        mem3 = torch.zeros(B, self.fc3.out_features, device=device)
        mem_out = torch.zeros(B, self.fc_out.out_features, device=device)

        # Output spike counts
        spk_out_sum = torch.zeros(B, self.fc_out.out_features, device=device)

        # Optional: sums of hidden spikes over time
        if return_hidden_spikes:
            spk1_sum = torch.zeros(B, self.fc1.out_features, device=device)
            spk2_sum = torch.zeros(B, self.fc2.out_features, device=device)
            spk3_sum = torch.zeros(B, self.fc3.out_features, device=device)

        # Full time series of hidden spikes [T, B, H]
        if return_time_series:
            spk1_ts = torch.zeros(T_steps, B, self.fc1.out_features, device=device)
            spk2_ts = torch.zeros(T_steps, B, self.fc2.out_features, device=device)
            spk3_ts = torch.zeros(T_steps, B, self.fc3.out_features, device=device)

        # Per-timestep activations for DST
        if return_activations:
            activations = {
                "layer1": [],
                "layer2": [],
                "layer3": [],
            }

        for t in range(T_steps):
            x_t = x_seq[t]  # [B, input_dim]

            # Layer 1
            cur1 = self.fc1(x_t)
            spk1, mem1 = self.lif1(cur1, mem1)

            # Layer 2
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            # Layer 3
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            # Output
            cur_out = self.fc_out(spk3)
            spk_out, mem_out = self.lif_out(cur_out, mem_out)

            spk_out_sum += spk_out

            if return_hidden_spikes:
                spk1_sum += spk1
                spk2_sum += spk2
                spk3_sum += spk3

            if return_time_series:
                spk1_ts[t] = spk1
                spk2_ts[t] = spk2
                spk3_ts[t] = spk3

            if return_activations:
                activations["layer1"].append(spk1.detach().cpu())
                activations["layer2"].append(spk2.detach().cpu())
                activations["layer3"].append(spk3.detach().cpu())

        if return_activations:
            for k in activations:
                activations[k] = torch.stack(activations[k], dim=0)

        # 1) time-series mode
        if return_time_series:
            hidden_time = {
                "layer1": spk1_ts,  # [T, B, H]
                "layer2": spk2_ts,
                "layer3": spk3_ts,
            }
            return spk_out_sum, hidden_time

        # 2) summed-hidden mode
        if return_hidden_spikes:
            hidden_spikes = {
                "layer1": spk1_sum,  # [B, H]
                "layer2": spk2_sum,
                "layer3": spk3_sum,
            }
            if return_activations:
                return spk_out_sum, hidden_spikes, activations
            return spk_out_sum, hidden_spikes

        if return_activations:
            return spk_out_sum, activations

        return spk_out_sum