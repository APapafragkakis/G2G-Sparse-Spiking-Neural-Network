import torch
from torch import nn
import torch.nn.functional as F

import snntorch as snn
from snntorch import surrogate


class RandomGroupSparseLinear(nn.Module):
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

        base_in_group = torch.arange(self.in_features) // in_group_size
        base_out_group = torch.arange(self.out_features) // out_group_size

        perm_in = torch.randperm(self.in_features)
        perm_out = torch.randperm(self.out_features)

        in_group = torch.empty_like(base_in_group)
        out_group = torch.empty_like(base_out_group)
        in_group[perm_in] = base_in_group
        out_group[perm_out] = base_out_group

        self.register_buffer("in_group", in_group)
        self.register_buffer("out_group", out_group)

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


class RandomSNN(nn.Module):
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

        self.fc1 = RandomGroupSparseLinear(input_dim, hidden_dim, num_groups, p_intra, p_inter)
        self.fc2 = RandomGroupSparseLinear(hidden_dim, hidden_dim, num_groups, p_intra, p_inter)
        self.fc3 = RandomGroupSparseLinear(hidden_dim, hidden_dim, num_groups, p_intra, p_inter)
        self.fc_out = nn.Linear(hidden_dim, num_classes)

        beta = 0.9
        threshold = 1.0
        spike_grad = surrogate.atan()

        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,
                             learn_beta=True, learn_threshold=True, threshold=threshold)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,
                             learn_beta=True, learn_threshold=True, threshold=threshold)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,
                             learn_beta=True, learn_threshold=True, threshold=threshold)
        self.lif_out = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,
                                learn_beta=True, learn_threshold=True, threshold=threshold)

    def forward(self, x_seq, return_hidden_spikes: bool = False, 
                return_spk_rec: bool = False, return_activations: bool = False):
        T_steps, B, _ = x_seq.shape

        self.lif1.reset_mem()
        self.lif2.reset_mem()
        self.lif3.reset_mem()
        self.lif_out.reset_mem()

        spk_out_rec = [] if return_spk_rec else None
        spk_out_sum = torch.zeros(B, self.fc_out.out_features, device=x_seq.device)

        if return_hidden_spikes:
            spk1_sum = torch.zeros(B, self.fc1.out_features, device=x_seq.device)
            spk2_sum = torch.zeros(B, self.fc2.out_features, device=x_seq.device)
            spk3_sum = torch.zeros(B, self.fc3.out_features, device=x_seq.device)

        if return_activations:
            activations = {"layer1": [], "layer2": [], "layer3": []}

        for t in range(T_steps):
            x_t = x_seq[t]

            cur1 = self.fc1(x_t)
            spk1 = self.lif1(cur1)

            cur2 = self.fc2(spk1)
            spk2 = self.lif2(cur2)

            cur3 = self.fc3(spk2)
            spk3 = self.lif3(cur3)

            cur_out = self.fc_out(spk3)
            spk_out = self.lif_out(cur_out)

            if return_spk_rec:
                spk_out_rec.append(spk_out)

            spk_out_sum += spk_out

            if return_hidden_spikes:
                spk1_sum += spk1
                spk2_sum += spk2
                spk3_sum += spk3

            if return_activations:
                activations["layer1"].append(spk1.detach())
                activations["layer2"].append(spk2.detach())
                activations["layer3"].append(spk3.detach())

        if return_spk_rec:
            spk_out_rec = torch.stack(spk_out_rec, dim=0)

        if return_activations:
            for k in activations:
                activations[k] = torch.stack(activations[k], dim=0).cpu()

        hidden_spikes = None
        if return_hidden_spikes:
            hidden_spikes = {"layer1": spk1_sum, "layer2": spk2_sum, "layer3": spk3_sum}

        # Unified returns
        if return_spk_rec and return_hidden_spikes and return_activations:
            return spk_out_rec, spk_out_sum, hidden_spikes, activations

        if return_spk_rec and return_hidden_spikes:
            return spk_out_rec, spk_out_sum, hidden_spikes

        if return_spk_rec and return_activations:
            return spk_out_rec, spk_out_sum, activations

        if return_hidden_spikes and return_activations:
            return spk_out_sum, hidden_spikes, activations

        if return_hidden_spikes:
            return spk_out_sum, hidden_spikes

        if return_spk_rec:
            return spk_out_rec, spk_out_sum

        if return_activations:
            return spk_out_sum, activations

        return spk_out_sum