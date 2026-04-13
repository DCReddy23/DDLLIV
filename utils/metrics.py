"""
Evaluation metrics for image restoration quality assessment.

Provides:
    - compute_psnr: Peak Signal-to-Noise Ratio
    - compute_ssim: Structural Similarity Index
    - compute_lpips: Learned Perceptual Image Patch Similarity (optional)
    - MetricTracker: Running average tracker for all metrics
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import os


def compute_psnr(pred, target, max_val=1.0):
    """Compute PSNR between predicted and target images.

    Args:
        pred: Predicted image tensor [B, C, H, W] in [0, 1].
        target: Target image tensor [B, C, H, W] in [0, 1].
        max_val: Maximum pixel value. Default: 1.0.

    Returns:
        PSNR value in dB (float). Higher is better.
    """
    mse = F.mse_loss(pred, target, reduction='mean')
    if mse == 0:
        return float('inf')
    return (10.0 * torch.log10(max_val ** 2 / mse)).item()


def compute_ssim(pred, target, window_size=11, channels=3):
    """Compute SSIM between predicted and target images.

    Uses a sliding Gaussian window approach matching Wang et al. (2004).

    Args:
        pred: Predicted image tensor [B, C, H, W] in [0, 1].
        target: Target image tensor [B, C, H, W] in [0, 1].
        window_size: Gaussian kernel size. Default: 11.
        channels: Number of channels. Default: 3.

    Returns:
        SSIM value (float) in [0, 1]. Higher is better.
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Create Gaussian window
    coords = torch.arange(window_size, dtype=torch.float32, device=pred.device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    g /= g.sum()
    window_2d = g.unsqueeze(1) @ g.unsqueeze(0)
    window = window_2d.unsqueeze(0).unsqueeze(0).expand(channels, 1, window_size, window_size).contiguous()

    pad = window_size // 2

    mu1 = F.conv2d(pred, window, padding=pad, groups=channels)
    mu2 = F.conv2d(target, window, padding=pad, groups=channels)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=pad, groups=channels) - mu12

    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().item()


def compute_lpips(pred, target, lpips_fn=None):
    """Compute LPIPS between predicted and target images.

    Args:
        pred: Predicted image tensor [B, C, H, W] in [0, 1].
        target: Target image tensor [B, C, H, W] in [0, 1].
        lpips_fn: Pre-initialized LPIPS function. If None, returns 0.0.

    Returns:
        LPIPS value (float). Lower is better.
    """
    if lpips_fn is None:
        return 0.0
    # LPIPS expects inputs in [-1, 1]
    pred_scaled = pred * 2.0 - 1.0
    target_scaled = target * 2.0 - 1.0
    with torch.no_grad():
        score = lpips_fn(pred_scaled, target_scaled)
    return score.mean().item()


class MetricTracker:
    """Tracks running averages and per-image metrics for evaluation.

    Usage:
        tracker = MetricTracker()
        for pred, gt in data:
            tracker.update(pred, gt, image_name="img_001.png")
        tracker.print_summary()
        tracker.save("results/metrics.json")
    """

    def __init__(self, use_lpips=False):
        self.psnr_values = []
        self.ssim_values = []
        self.lpips_values = []
        self.image_names = []
        self.lpips_fn = None

        if use_lpips:
            try:
                import lpips
                self.lpips_fn = lpips.LPIPS(net='alex').eval()
                print("LPIPS metric initialized (AlexNet backbone)")
            except ImportError:
                print("Warning: lpips package not installed. Skipping LPIPS metric.")
                print("Install with: pip install lpips")

    def update(self, pred, target, image_name=""):
        """Compute and store metrics for one image pair.

        Args:
            pred: Predicted image tensor [1, C, H, W] or [C, H, W] in [0, 1].
            target: Target image tensor, same shape as pred, in [0, 1].
            image_name: Optional identifier for this image.
        """
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
        if target.dim() == 3:
            target = target.unsqueeze(0)

        pred = pred.clamp(0.0, 1.0)
        target = target.clamp(0.0, 1.0)

        psnr = compute_psnr(pred, target)
        ssim = compute_ssim(pred, target, channels=pred.shape[1])

        self.psnr_values.append(psnr)
        self.ssim_values.append(ssim)
        self.image_names.append(image_name)

        if self.lpips_fn is not None:
            if self.lpips_fn.parameters().__next__().device != pred.device:
                self.lpips_fn = self.lpips_fn.to(pred.device)
            lpips_val = compute_lpips(pred, target, self.lpips_fn)
            self.lpips_values.append(lpips_val)
        else:
            self.lpips_values.append(0.0)

        return psnr, ssim, self.lpips_values[-1]

    def get_summary(self):
        """Return dict with mean and std of all metrics."""
        summary = {
            'psnr_mean': float(np.mean(self.psnr_values)),
            'psnr_std': float(np.std(self.psnr_values)),
            'ssim_mean': float(np.mean(self.ssim_values)),
            'ssim_std': float(np.std(self.ssim_values)),
            'num_images': len(self.psnr_values),
        }
        if self.lpips_fn is not None:
            summary['lpips_mean'] = float(np.mean(self.lpips_values))
            summary['lpips_std'] = float(np.std(self.lpips_values))
        return summary

    def print_summary(self):
        """Print formatted metrics summary."""
        s = self.get_summary()
        print("\n" + "=" * 60)
        print("  EVALUATION METRICS SUMMARY")
        print("=" * 60)
        print(f"  Images evaluated : {s['num_images']}")
        print(f"  PSNR             : {s['psnr_mean']:.4f} ± {s['psnr_std']:.4f} dB")
        print(f"  SSIM             : {s['ssim_mean']:.4f} ± {s['ssim_std']:.4f}")
        if 'lpips_mean' in s:
            print(f"  LPIPS            : {s['lpips_mean']:.4f} ± {s['lpips_std']:.4f}")
        print("=" * 60 + "\n")

    def save(self, filepath):
        """Save per-image and summary metrics to JSON."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        per_image = []
        for i, name in enumerate(self.image_names):
            entry = {
                'image': name,
                'psnr': self.psnr_values[i],
                'ssim': self.ssim_values[i],
            }
            if self.lpips_fn is not None:
                entry['lpips'] = self.lpips_values[i]
            per_image.append(entry)

        output = {
            'summary': self.get_summary(),
            'per_image': per_image,
        }

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Metrics saved to {filepath}")
