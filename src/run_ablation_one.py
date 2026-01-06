import os
import torch
import train  

from train import select_device, build_model, evaluate, parse_args, get_checkpoint_path
from intra_ablation import apply_intra_only_masks, restore_masks


def load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device)


def get_test_loader(args):
    """Get test loader based on dataset"""
    if args.dataset == "fashionmnist":
        from data.fashionmnist import get_fashion_loaders
        _, test_loader = get_fashion_loaders(args.batch_size)
    elif args.dataset == "cifar10":
        from data.cifar10_100 import get_cifar10_loaders
        _, test_loader = get_cifar10_loaders(args.batch_size)
    else:
        from data.cifar10_100 import get_cifar100_loaders
        _, test_loader = get_cifar100_loaders(args.batch_size)
    return test_loader


def setup_dataset_globals(args):
    """Make train.py globals match the dataset, same as train.main()"""
    if args.dataset == "fashionmnist":
        train.input_dim = 28 * 28
        train.num_classes = 10
    elif args.dataset == "cifar10":
        train.input_dim = 3 * 32 * 32
        train.num_classes = 10
    elif args.dataset == "cifar100":
        train.input_dim = 3 * 32 * 32
        train.num_classes = 100
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


def main():
    args = parse_args()
    device = select_device()

    setup_dataset_globals(args)
    
    model = build_model(args.model, args.p_inter).to(device)

    # Use train.py's checkpoint naming logic
    ckpt_path = get_checkpoint_path(args)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = load_checkpoint(model, ckpt_path, device)
    test_loader = get_test_loader(args)

    # Evaluate normal
    acc_normal = evaluate(model, test_loader, device)

    # Evaluate intra-only
    backups = apply_intra_only_masks(model)
    acc_intra_only = evaluate(model, test_loader, device)
    restore_masks(model, backups)

    # Print results
    print("=" * 60)
    print(f"Checkpoint: {ckpt_path}")
    print(f"Normal acc:     {acc_normal:.4f}")
    print(f"Intra-only acc: {acc_intra_only:.4f}")
    print(f"Delta:          {acc_normal - acc_intra_only:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()