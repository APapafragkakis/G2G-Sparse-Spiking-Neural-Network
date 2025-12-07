import torch
from torch import nn

import snntorch as snn
from snntorch import surrogate


class DenseSNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()

        # Fully-connected layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, num_classes)

        # LIF neuron parameters
        beta = 0.90
        spike_grad = surrogate.fast_sigmoid(slope=50)

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

        # Accumulate spikes for hidden layers
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
