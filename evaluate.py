"""
LightenDiffusion — Evaluation Script

Usage:
    python evaluate.py                                           # Default evaluation
    python evaluate.py --resume ckpt/stage2/model_best.pth.tar   # Use best model
    python evaluate.py --gt_dir /path/to/gt_images               # Compute metrics against GT
    python evaluate.py --no_metrics                               # Skip metric computation
"""

import argparse
import os
import yaml
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import models
import datasets
import utils
from models import DenoisingDiffusion, DiffusiveRestoration


def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Latent-Retinex Diffusion Models — Evaluation')
    parser.add_argument("--config", default='unsupervised.yml', type=str,
                        help="Path to the config file")
    parser.add_argument('--mode', type=str, default='evaluation', help='training or evaluation')
    parser.add_argument('--resume', default='ckpt/stage2/stage2_weight.pth.tar', type=str,
                        help='Path for the diffusion model checkpoint to load for evaluation')
    parser.add_argument("--image_folder", default='results/', type=str,
                        help="Location to save restored images")
    parser.add_argument('--gt_dir', default=None, type=str,
                        help="Directory with ground-truth images for computing metrics")
    parser.add_argument('--no_metrics', action='store_true',
                        help="Skip metric computation (just save images)")
    parser.add_argument('--seed', default=230, type=int,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return args, new_config


def dict2namespace(config):
    """Recursively converts a nested dict to argparse.Namespace."""
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()

    # Setup device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: {}".format(device))
    config.device = device

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        print('Note: Currently supports evaluations (restoration) when run only on a single GPU!')

    # Override metrics setting from CLI
    if args.no_metrics and hasattr(config, 'evaluation'):
        config.evaluation.compute_metrics = False

    print("=> using dataset '{}'".format(config.data.val_dataset))
    DATASET = datasets.__dict__[config.data.type](config)
    _, val_loader = DATASET.get_loaders()

    # Create model
    print("=> creating denoising-diffusion model")
    diffusion = DenoisingDiffusion(args, config)
    model = DiffusiveRestoration(diffusion, args, config)

    print(f"=> starting restoration ({len(val_loader)} images)")
    model.restore(val_loader, gt_dir=args.gt_dir)


if __name__ == '__main__':
    main()
