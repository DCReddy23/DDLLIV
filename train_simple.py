import argparse
import os
import random

import numpy as np
import torch
import yaml

import datasets
from models.simple_ddm import SimpleDDM, SimpleDDMTrainer


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            value = dict2namespace(value)
        setattr(namespace, key, value)
    return namespace


def parse_args_and_config():
    parser = argparse.ArgumentParser(description="Simple DDLLIV training")
    parser.add_argument("--config", default="unsupervised.yml", type=str)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--seed", default=230, type=int)
    args = parser.parse_args()

    with open(os.path.join("configs", args.config), "r") as f:
        config = dict2namespace(yaml.safe_load(f))
    return args, config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args, config = parse_args_and_config()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = datasets.__dict__[config.data.type](config)
    train_loader, val_loader = dataset.get_loaders()

    model = SimpleDDM(
        in_channels=3,
        out_channels=3,
        base_channels=getattr(config.model, "ch", 32),
    )
    trainer = SimpleDDMTrainer(
        model=model,
        device=device,
        lr=getattr(config.optim, "lr", 1e-4),
        weight_decay=getattr(config.optim, "weight_decay", 0.0),
        step_size=getattr(config.training, "lr_step_size", 20),
        gamma=getattr(config.training, "lr_gamma", 0.5),
        l1_weight=getattr(config.training, "l1_weight", 1.0),
        l2_weight=getattr(config.training, "l2_weight", 1.0),
    )

    ckpt_dir = getattr(config.data, "ckpt_dir", "ckpt/simple")
    os.makedirs(ckpt_dir, exist_ok=True)
    latest_path = os.path.join(ckpt_dir, "simple_model_latest.pth.tar")
    best_path = os.path.join(ckpt_dir, "simple_model_best.pth.tar")

    start_epoch = 0
    best_psnr = float("-inf")
    if args.resume and os.path.isfile(args.resume):
        start_epoch, best_psnr = trainer.load_checkpoint(args.resume)
        start_epoch += 1
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    n_epochs = getattr(config.training, "n_epochs", 100)
    for epoch in range(start_epoch, n_epochs):
        train_stats = trainer.train_one_epoch(train_loader)
        val_stats = trainer.validate(val_loader)
        trainer.step_scheduler()

        lr = trainer.current_lr()
        print(
            f"Epoch {epoch + 1}/{n_epochs} | "
            f"train_loss: {train_stats['total']:.5f} (l1: {train_stats['l1']:.5f}, l2: {train_stats['l2']:.5f}) | "
            f"val_loss: {val_stats['total']:.5f} | "
            f"PSNR: {val_stats['psnr']:.3f} dB | SSIM: {val_stats['ssim']:.4f} | "
            f"lr: {lr:.2e}"
        )

        trainer.save_checkpoint(latest_path, epoch, best_psnr)
        if val_stats["psnr"] > best_psnr:
            best_psnr = val_stats["psnr"]
            trainer.save_checkpoint(best_path, epoch, best_psnr)
            print(f"Saved new best model (PSNR: {best_psnr:.3f} dB)")

    print("Simple training complete.")


if __name__ == "__main__":
    main()
