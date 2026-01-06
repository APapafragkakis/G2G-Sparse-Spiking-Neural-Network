import torch

from models.index_snn import IndexSparseLinear
from models.mixer_snn import MixerSparseLinear
from models.random_snn import RandomGroupSparseLinear


def _intra_mask_index(layer: IndexSparseLinear) -> torch.Tensor:
    dev = layer.mask.device
    in_group_size = layer.in_features // layer.num_groups
    out_group_size = layer.out_features // layer.num_groups

    in_idx = torch.arange(layer.in_features, device=dev)
    out_idx = torch.arange(layer.out_features, device=dev)

    in_group = in_idx // in_group_size
    out_group = out_idx // out_group_size

    return (out_group[:, None] == in_group[None, :])


def _intra_mask_mixer(layer: MixerSparseLinear) -> torch.Tensor:
    dev = layer.mask.device
    in_group_size = layer.in_features // layer.num_groups
    out_group_size = layer.out_features // layer.num_groups

    if not layer.interleaved:
        in_group = torch.arange(layer.in_features, device=dev) // in_group_size
        out_group = torch.arange(layer.out_features, device=dev) // out_group_size
    else:
        in_group = torch.arange(layer.in_features, device=dev) % layer.num_groups
        out_group = torch.arange(layer.out_features, device=dev) % layer.num_groups

    return (out_group[:, None] == in_group[None, :])


def _intra_mask_random(layer: RandomGroupSparseLinear) -> torch.Tensor:
    dev = layer.mask.device
    if not hasattr(layer, "in_group") or not hasattr(layer, "out_group"):
        raise RuntimeError(
            "RandomGroupSparseLinear missing in_group/out_group. "
            "Re-train Random after applying the patch."
        )
    return (layer.out_group.to(dev)[:, None] == layer.in_group.to(dev)[None, :])


def apply_intra_only_masks(model) -> dict:
    """
    Temporarily sets each sparse layer mask to (mask & intra_mask).
    Returns backups so you can restore the exact original masks.
    """
    backups = {}
    for name, module in model.named_modules():
        if not hasattr(module, "mask"):
            continue

        if isinstance(module, IndexSparseLinear):
            intra = _intra_mask_index(module)
        elif isinstance(module, MixerSparseLinear):
            intra = _intra_mask_mixer(module)
        elif isinstance(module, RandomGroupSparseLinear):
            intra = _intra_mask_random(module)
        else:
            continue

        backups[name] = module.mask.detach().clone()
        module.mask.data = (module.mask.bool() & intra.bool()).to(module.mask.dtype)

    return backups


def restore_masks(model, backups: dict) -> None:
    for name, module in model.named_modules():
        if name in backups:
            module.mask.data = backups[name].to(module.mask.dtype)