"""
Diffusive restoration module for inference.

Handles:
    - Loading pretrained diffusion model
    - Running inference on validation images  
    - Computing and saving evaluation metrics (PSNR, SSIM, LPIPS)
"""

import torch
import numpy as np
import utils
from utils.metrics import MetricTracker
import os
import time
import logging
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF

logger = logging.getLogger('lightendiffusion')


class DiffusiveRestoration:
    """Performs image restoration using a trained diffusion model.

    Supports:
        - Automatic image padding to multiples of 64
        - Per-image PSNR/SSIM/LPIPS metric computation
        - Summary statistics with mean ± std
        - Metrics saved to JSON for easy comparison

    Args:
        diffusion: DenoisingDiffusion instance with loaded model.
        args: Arguments namespace.
        config: Config namespace.
    """

    def __init__(self, diffusion, args, config):
        super(DiffusiveRestoration, self).__init__()
        self.args = args
        self.config = config
        self.diffusion = diffusion

        if os.path.isfile(args.resume):
            self.diffusion.load_ddm_ckpt(args.resume, ema=False)
            self.diffusion.model.eval()
            self.diffusion.model.module.eval() 
        else:
            print('Pre-trained model path is missing!')

    def restore(self, val_loader, gt_dir=None):
        """Run restoration on all validation images.

        Args:
            val_loader: Validation DataLoader.
            gt_dir: Optional directory with ground-truth images for metrics.
                    If None, metrics are skipped.
        """
        image_folder = os.path.join(self.args.image_folder, self.config.data.val_dataset)

        # Initialize metric tracker
        use_lpips = getattr(self.config.evaluation, 'use_lpips', False) if hasattr(self.config, 'evaluation') else False
        compute_metrics = getattr(self.config.evaluation, 'compute_metrics', True) if hasattr(self.config, 'evaluation') else True
        tracker = MetricTracker(use_lpips=use_lpips) if compute_metrics else None

        total_time = 0.0

        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):

                x_cond = x[:, :3, :, :].to(self.diffusion.device)
                # Ground truth is the second half of the concatenated tensor
                x_gt = x[:, 3:, :, :].to(self.diffusion.device) if x.shape[1] == 6 else None

                b, c, h, w = x_cond.shape
                img_h_64 = int(64 * np.ceil(h / 64.0))
                img_w_64 = int(64 * np.ceil(w / 64.0))
                x_cond = F.pad(x_cond, (0, img_w_64 - w, 0, img_h_64 - h), 'reflect')

                t1 = time.time()
                pred_x = self.diffusion.model(torch.cat((x_cond, x_cond),
                                                        dim=1))["pred_x"][:, :, :h, :w]
                t2 = time.time()
                elapsed = t2 - t1
                total_time += elapsed

                # Save restored image
                utils.save_image(pred_x, os.path.join(image_folder, f"{y[0]}"))

                # Compute metrics if ground truth is available
                if tracker is not None and x_gt is not None:
                    pred_clamped = pred_x.clamp(0.0, 1.0)
                    gt_clamped = x_gt.clamp(0.0, 1.0)
                    psnr, ssim, lpips_val = tracker.update(pred_clamped, gt_clamped, image_name=y[0])
                    print(f"[{i+1}/{len(val_loader)}] {y[0]} | "
                          f"PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f} | "
                          f"LPIPS: {lpips_val:.4f} | Time: {elapsed:.3f}s")
                elif tracker is not None and gt_dir is not None:
                    # Try loading GT from directory
                    gt_path = os.path.join(gt_dir, y[0])
                    if os.path.exists(gt_path):
                        gt_img = Image.open(gt_path).convert('RGB')
                        gt_tensor = TF.to_tensor(gt_img).unsqueeze(0).to(pred_x.device)
                        # Resize GT to match prediction if needed
                        if gt_tensor.shape[-2:] != pred_x.shape[-2:]:
                            gt_tensor = F.interpolate(gt_tensor, size=pred_x.shape[-2:], mode='bilinear')
                        pred_clamped = pred_x.clamp(0.0, 1.0)
                        psnr, ssim, lpips_val = tracker.update(pred_clamped, gt_tensor.clamp(0.0, 1.0), image_name=y[0])
                        print(f"[{i+1}/{len(val_loader)}] {y[0]} | "
                              f"PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f} | "
                              f"LPIPS: {lpips_val:.4f} | Time: {elapsed:.3f}s")
                    else:
                        print(f"[{i+1}/{len(val_loader)}] {y[0]} | Time: {elapsed:.3f}s (no GT found)")
                else:
                    print(f"[{i+1}/{len(val_loader)}] {y[0]} | Time: {elapsed:.3f}s")

        # Print and save summary
        print(f"\nTotal inference time: {total_time:.2f}s "
              f"({total_time / max(len(val_loader), 1):.3f}s per image)")

        if tracker is not None and len(tracker.psnr_values) > 0:
            tracker.print_summary()
            metrics_path = os.path.join(image_folder, 'metrics.json')
            tracker.save(metrics_path)
