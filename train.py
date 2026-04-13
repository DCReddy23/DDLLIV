"""
LightenDiffusion — Training Script

Usage:
    python train.py                                    # Train with default config
    python train.py --config unsupervised.yml          # Specify config
    python train.py --resume ckpt/stage2/model_latest.pth.tar  # Resume training
    python train.py --seed 42                          # Set random seed
"""

import argparse
import os
import random
import yaml
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import numpy as np
import models
import datasets
import utils
from utils.logging import setup_logger, log_config
from models import DenoisingDiffusion


def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Latent-Retinex Diffusion Models')
    parser.add_argument("--config", default='unsupervised.yml', type=str,
                        help="Path to the config file")
    parser.add_argument('--mode', type=str, default='training', help='training or evaluation')
    parser.add_argument('--resume', default='', type=str,
                        help='Path for checkpoint to load and resume')
    parser.add_argument("--image_folder", default='results/', type=str,
                        help="Location to save restored validation image patches")
    parser.add_argument('--seed', default=230, type=int, metavar='N',
                        help='Seed for initializing training (default: 230)')
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


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def main():
    args, config = parse_args_and_config()

    # Setup device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: {}".format(device))
    config.device = device

    # Set random seed for reproducibility
    set_seed(args.seed)
    print(f"Random seed: {args.seed}")

    # Setup logger
    log_dir = getattr(config.data, 'ckpt_dir', 'ckpt/stage2')
    os.makedirs(log_dir, exist_ok=True)
    log = setup_logger(log_dir)

    # Log full configuration
    log_config(config, log)

    # Data loading
    log.info("=> using dataset '{}'".format(config.data.train_dataset))
    DATASET = datasets.__dict__[config.data.type](config)

    # Create model
    log.info("=> creating denoising-diffusion model...")
    diffusion = DenoisingDiffusion(args, config)

    # Log model size
    total_params = sum(p.numel() for p in diffusion.model.parameters())
    log.info(f"Total model parameters: {total_params:,}")

    # Train
    diffusion.train(DATASET)


if __name__ == "__main__":
    main()
