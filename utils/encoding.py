import torch

def rate_encode(images, T):
    B = images.size(0)
    flat = images.view(B, -1)
    flat_repeat = flat.unsqueeze(0).repeat(T, 1, 1)
    spikes = torch.bernoulli(flat_repeat)
    return spikes
