import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import yaml

import datasets
import utils
from models.simple_ddm import SimpleDDM, SimpleDDMTrainer
from utils.metrics import MetricTracker

DEFAULT_SIMPLE_CKPT_DIR = "ckpt/simple"
# Keep 64-padding to match original inference path alignment behavior.
PADDING_MULTIPLE = 64


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            value = dict2namespace(value)
        setattr(namespace, key, value)
    return namespace


def parse_args_and_config():
    parser = argparse.ArgumentParser(description="Simple DDLLIV evaluation")
    parser.add_argument("--config", default="unsupervised.yml", type=str)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--image_folder", default="results_simple", type=str)
    args = parser.parse_args()

    with open(os.path.join("configs", args.config), "r") as f:
        config = dict2namespace(yaml.safe_load(f))
    return args, config


def main():
    args, config = parse_args_and_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = datasets.__dict__[config.data.type](config)
    _, val_loader = dataset.get_loaders()

    model = SimpleDDM(
        in_channels=3,
        out_channels=3,
        base_channels=getattr(config.model, "ch", 32),
    )
    trainer = SimpleDDMTrainer(model=model, device=device)

    resume_path = args.resume or os.path.join(
        getattr(config.data, "ckpt_dir", DEFAULT_SIMPLE_CKPT_DIR),
        "simple_model_best.pth.tar",
    )
    if not os.path.isfile(resume_path):
        raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
    trainer.load_checkpoint(resume_path)

    output_dir = os.path.join(args.image_folder, config.data.val_dataset)
    os.makedirs(output_dir, exist_ok=True)
    tracker = MetricTracker(use_lpips=False)

    with torch.no_grad():
        for i, (batch, img_ids) in enumerate(val_loader):
            low = batch[:, :3].to(device)
            high = batch[:, 3:].to(device)

            h, w = low.shape[-2:]
            h_pad = int(PADDING_MULTIPLE * np.ceil(h / float(PADDING_MULTIPLE)))
            w_pad = int(PADDING_MULTIPLE * np.ceil(w / float(PADDING_MULTIPLE)))
            low_padded = F.pad(low, (0, w_pad - w, 0, h_pad - h), "reflect")

            pred = trainer.predict(low_padded)[:, :, :h, :w]
            pred_clamped = pred.clamp(0.0, 1.0)
            high_clamped = high.clamp(0.0, 1.0)

            utils.save_image(pred_clamped, os.path.join(output_dir, img_ids[0]))
            psnr, ssim, _ = tracker.update(pred_clamped, high_clamped, image_name=img_ids[0])
            print(f"[{i + 1}/{len(val_loader)}] {img_ids[0]} | PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}")

    tracker.print_summary()
    tracker.save(os.path.join(output_dir, "metrics.json"))


if __name__ == "__main__":
    main()
