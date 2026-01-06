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

        # LIF neuron parameters - matching CIFAR_SNN
        beta = 0.9
        threshold = 1.0
        spike_grad = surrogate.atan()

        # Spiking (LIF) layers - with learnable params
        self.lif1 = snn.Leaky(
            beta=beta, 
            spike_grad=spike_grad,
            init_hidden=True,
            learn_beta=True,
            learn_threshold=True,
            threshold=threshold
        )
        self.lif2 = snn.Leaky(
            beta=beta, 
            spike_grad=spike_grad,
            init_hidden=True,
            learn_beta=True,
            learn_threshold=True,
            threshold=threshold
        )
        self.lif3 = snn.Leaky(
            beta=beta, 
            spike_grad=spike_grad,
            init_hidden=True,
            learn_beta=True,
            learn_threshold=True,
            threshold=threshold
        )
        self.lif_out = snn.Leaky(
            beta=beta, 
            spike_grad=spike_grad,
            init_hidden=True,
            learn_beta=True,
            learn_threshold=True,
            threshold=threshold
        )

    def forward(self, x_seq, return_hidden_spikes: bool = False, return_spk_rec: bool = False):
        # x_seq: [T, B, input_dim]
        T_steps, B, _ = x_seq.shape

        # Reset membrane potentials
        self.lif1.reset_mem()
        self.lif2.reset_mem()
        self.lif3.reset_mem()
        self.lif_out.reset_mem()

        # Lists to store time-series spikes (only if requested)
        spk_out_rec = [] if return_spk_rec else None
        
        # Accumulate spikes over time
        spk_out_sum = torch.zeros(B, self.fc_out.out_features, device=x_seq.device)

        # Accumulate spikes for hidden layers
        if return_hidden_spikes:
            spk1_sum = torch.zeros(B, self.fc1.out_features, device=x_seq.device)
            spk2_sum = torch.zeros(B, self.fc2.out_features, device=x_seq.device)
            spk3_sum = torch.zeros(B, self.fc3.out_features, device=x_seq.device)

        for t in range(T_steps):
            x_t = x_seq[t]

            # Layer 1
            cur1 = self.fc1(x_t)
            spk1 = self.lif1(cur1)

            # Layer 2
            cur2 = self.fc2(spk1)
            spk2 = self.lif2(cur2)

            # Layer 3
            cur3 = self.fc3(spk2)
            spk3 = self.lif3(cur3)

            # Output layer
            cur_out = self.fc_out(spk3)
            spk_out = self.lif_out(cur_out)

            # Store time-series spike (if requested)
            if return_spk_rec:
                spk_out_rec.append(spk_out)
            
            # Sum spikes across time steps
            spk_out_sum += spk_out

            if return_hidden_spikes:
                spk1_sum += spk1
                spk2_sum += spk2
                spk3_sum += spk3

        # Stack time-series spikes if requested: [T, B, num_classes]
        if return_spk_rec:
            spk_out_rec = torch.stack(spk_out_rec, dim=0)

        # Return based on flags
        if return_hidden_spikes and return_spk_rec:
            hidden_spikes = {
                "layer1": spk1_sum,
                "layer2": spk2_sum, 
                "layer3": spk3_sum,
            }
            return spk_out_rec, spk_out_sum, hidden_spikes
        
        if return_hidden_spikes:
            hidden_spikes = {
                "layer1": spk1_sum,
                "layer2": spk2_sum, 
                "layer3": spk3_sum,
            }
            return spk_out_sum, hidden_spikes
        
        if return_spk_rec:
            return spk_out_rec, spk_out_sum
        
        # Default: only sums (backwards compatible)
        return spk_out_sum