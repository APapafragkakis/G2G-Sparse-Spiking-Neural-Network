import torch
from torch import nn
import torch.nn.functional as F

import snntorch as snn
from snntorch import surrogate


class ERSparseLinear(nn.Module):
    """
    Unstructured Erdos–Rényi sparse linear layer.

    Each connection is independently active with probability p_active.
    This gives a parameter-matched random sparse baseline without any
    group structure.
    """

    def __init__(self, in_features: int, out_features: int, p_active: float, bias: bool = True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.p_active = float(p_active)

        # Trainable dense weight matrix; sparsity is enforced by a fixed binary mask
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

        # Binary Erdos–Rényi mask, sampled once at initialization
        mask = torch.bernoulli(torch.full((out_features, in_features), self.p_active))
        self.register_buffer("mask", mask)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Standard Kaiming init as if the layer was dense
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

        # Row-wise rescaling so that active weights see a similar fan-in
        with torch.no_grad():
            full_fan_in = float(self.in_features)
            for out_idx in range(self.out_features):
                row_mask = self.mask[out_idx]
                fan_in_active = row_mask.sum().item()
                if 0 < fan_in_active < full_fan_in:
                    scale = (full_fan_in / fan_in_active) ** 0.5
                    self.weight[out_idx, row_mask.bool()] *= scale

        # Bias init based on effective fan-in
        if self.bias is not None:
            fan_in_eff = self.mask.sum(dim=1).float().mean().item()
            if fan_in_eff <= 0:
                fan_in_eff = self.in_features
            bound = 1.0 / fan_in_eff**0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:      [batch_size, in_features]
        output: [batch_size, out_features]
        """
        effective_weight = self.weight * self.mask
        return F.linear(x, effective_weight, self.bias)


class ERSNN(nn.Module):
    """
    Three-layer SNN with ERSparseLinear layers and LIF neurons.

    This mirrors the structure and parameters of the existing DenseSNN /
    IndexSNN / MixerSNN models, but uses unstructured ER sparsity instead
    of group-based connectivity.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, p_active: float):
        super().__init__()

        # Sparse fully-connected layers with ER connectivity
        self.fc1 = ERSparseLinear(input_dim, hidden_dim, p_active=p_active)
        self.fc2 = ERSparseLinear(hidden_dim, hidden_dim, p_active=p_active)
        self.fc3 = ERSparseLinear(hidden_dim, hidden_dim, p_active=p_active)
        self.fc_out = nn.Linear(hidden_dim, num_classes)

        # LIF neuron parameters (kept identical to the other SNN models)
        beta = 0.95
        spike_grad = surrogate.fast_sigmoid(slope=25)

        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif_out = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x_seq: torch.Tensor, return_hidden_spikes: bool = False):
        """
        x_seq: [T, B, input_dim]

        This follows the same temporal interface as the other SNN models:
        we integrate over T time steps and return spike counts at the output.
        Optionally, hidden layer spike counts can also be returned for
        firing-rate analysis.
        """
        T_steps, B, _ = x_seq.shape

        # Membrane potentials for each layer
        mem1 = torch.zeros(B, self.fc1.out_features, device=x_seq.device)
        mem2 = torch.zeros(B, self.fc2.out_features, device=x_seq.device)
        mem3 = torch.zeros(B, self.fc3.out_features, device=x_seq.device)
        mem_out = torch.zeros(B, self.fc_out.out_features, device=x_seq.device)

        # Output spike counts accumulated over time
        spk_out_sum = torch.zeros(B, self.fc_out.out_features, device=x_seq.device)

        # Optional hidden spike accumulation for firing rate statistics
        if return_hidden_spikes:
            spk1_sum = torch.zeros(B, self.fc1.out_features, device=x_seq.device)
            spk2_sum = torch.zeros(B, self.fc2.out_features, device=x_seq.device)
            spk3_sum = torch.zeros(B, self.fc3.out_features, device=x_seq.device)

        for t in range(T_steps):
            x_t = x_seq[t]

            # Layer 1
            cur1 = self.fc1(x_t)
            spk1, mem1 = self.lif1(cur1, mem1)

            # Layer 2
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            # Layer 3
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            # Output layer
            cur_out = self.fc_out(spk3)
            spk_out, mem_out = self.lif_out(cur_out, mem_out)

            spk_out_sum += spk_out

            if return_hidden_spikes:
                spk1_sum += spk1
                spk2_sum += spk2
                spk3_sum += spk3

        if return_hidden_spikes:
            hidden_spikes = {
                "layer1": spk1_sum,
                "layer2": spk2_sum,
                "layer3": spk3_sum,
            }
            return spk_out_sum, hidden_spikes

        return spk_out_sum
