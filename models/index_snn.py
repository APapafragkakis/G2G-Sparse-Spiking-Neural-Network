import torch
from torch import nn
import torch.nn.functional as F

import snntorch as snn
from snntorch import surrogate

# Define a custom linear layer with index-based sparse connectivity
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

        # Check that feature dimensions can be evenly split into groups
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

        # Full weight matrix; sparsity is enforced by a fixed binary mask
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.bias = None

        # Build the fixed binary connectivity mask
        mask = self._build_mask()
        # Register as buffer so it moves with .to(device) but is not trainable
        self.register_buffer("mask", mask)

        self.reset_parameters()

    def _build_mask(self) -> torch.Tensor:
        """Construct the binary connectivity mask based on index-based groups."""
        in_group_size = self.in_features // self.num_groups
        out_group_size = self.out_features // self.num_groups

        in_idx = torch.arange(self.in_features)    # [in_features]
        out_idx = torch.arange(self.out_features)  # [out_features]

        # Group assignment based on integer division of the index
        in_group = in_idx // in_group_size         # [in_features]
        out_group = out_idx // out_group_size      # [out_features]

        # same_group[o, i] is True if input i and output o belong to the same group
        same_group = (out_group[:, None] == in_group[None, :])  # [out_features, in_features]

        # Start with inter-group probability everywhere
        probs = torch.full((self.out_features, self.in_features), self.p_inter)
        # Overwrite same-group entries with the intra-group probability
        probs[same_group] = self.p_intra

        # Sample a binary mask according to the defined probabilities
        mask = torch.bernoulli(probs)
        return mask

    def reset_parameters(self) -> None:
        """Initialize weights and biases taking sparsity into account."""
        # Standard Kaiming init as if the layer were dense
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

        # Rescale active weights row-wise based on actual fan-in
        with torch.no_grad():
            full_fan_in = float(self.in_features)
            for out_idx in range(self.out_features):
                row_mask = self.mask[out_idx]  # [in_features]
                fan_in_active = row_mask.sum().item()
                if fan_in_active > 0 and fan_in_active < full_fan_in:
                    scale = (full_fan_in / fan_in_active) ** 0.5
                    self.weight[out_idx, row_mask.bool()] *= scale

        # Bias init based on effective fan-in
        if self.bias is not None:
            fan_in_eff = self.mask.sum(dim=1).float().mean().item()
            if fan_in_eff <= 0:
                fan_in_eff = self.in_features
            bound = 1.0 / fan_in_eff**0.5
            nn.init.uniform_(self.bias, -bound, bound)


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        input: [batch_size, in_features]
        output: [batch_size, out_features]
        """
        # Apply the fixed sparse mask to the trainable weights
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

    def forward(self, x_seq, return_hidden_spikes: bool = False):
        # x_seq: [T, B, input_dim]
        T_steps, B, _ = x_seq.shape

        # Initialise membrane potentials for all layers
        mem1 = torch.zeros(B, self.fc1.out_features, device=x_seq.device)
        mem2 = torch.zeros(B, self.fc2.out_features, device=x_seq.device)
        mem3 = torch.zeros(B, self.fc3.out_features, device=x_seq.device)
        mem_out = torch.zeros(B, self.fc_out.out_features, device=x_seq.device)

        # Accumulate spikes over time (rate coding at the output)
        spk_out_sum = torch.zeros(B, self.fc_out.out_features, device=x_seq.device)

        # Optionally accumulate spikes for hidden layers
        if return_hidden_spikes:
            spk1_sum = torch.zeros(B, self.fc1.out_features, device=x_seq.device)
            spk2_sum = torch.zeros(B, self.fc2.out_features, device=x_seq.device)
            spk3_sum = torch.zeros(B, self.fc3.out_features, device=x_seq.device)

        for t in range(T_steps):
            x_t = x_seq[t]  # [B, input_dim] at time step t

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

            # Sum spikes across time steps
            spk_out_sum += spk_out

            if return_hidden_spikes:
                spk1_sum += spk1
                spk2_sum += spk2
                spk3_sum += spk3

        # Shape: [B, num_classes] (spike counts per class)
        if return_hidden_spikes:
            hidden_spikes = {
                "layer1": spk1_sum,  # [B, hidden_dim]
                "layer2": spk2_sum,
                "layer3": spk3_sum,
            }
            return spk_out_sum, hidden_spikes

        return spk_out_sum
