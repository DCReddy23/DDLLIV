"""
Advanced loss functions for diffusion-based image restoration.

Provides:
    - CharbonnierLoss: Smooth L1-like loss, robust to outliers (replaces MSE for noise prediction)
    - SSIMLoss: Differentiable structural similarity loss
    - PerceptualLoss: VGG-16 feature matching loss for perceptual quality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class CharbonnierLoss(nn.Module):
    """Charbonnier loss (a smooth approximation of L1).

    More robust to outliers than MSE while being differentiable everywhere,
    unlike L1 which has a non-differentiable point at zero.

    L(x) = sqrt(x^2 + eps^2)

    Args:
        eps: Small constant for numerical stability. Default: 1e-6.
    """

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps_sq = eps ** 2

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps_sq)
        return loss.mean()


class SSIMLoss(nn.Module):
    """Differentiable Structural Similarity Index (SSIM) loss.

    Computes 1 - SSIM so that minimizing the loss maximizes SSIM.
    Uses a sliding Gaussian window for local statistics.

    Args:
        window_size: Size of the Gaussian kernel. Default: 11.
        channels: Number of image channels. Default: 3.
        reduction: 'mean' or 'none'. Default: 'mean'.
    """

    def __init__(self, window_size=11, channels=3, reduction='mean'):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.channels = channels
        self.reduction = reduction

        # Create Gaussian window
        self.register_buffer('window', self._create_window(window_size, channels))

    @staticmethod
    def _gaussian(window_size, sigma=1.5):
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g

    def _create_window(self, window_size, channels):
        _1d = self._gaussian(window_size)
        _2d = _1d.unsqueeze(1) @ _1d.unsqueeze(0)
        window = _2d.unsqueeze(0).unsqueeze(0).expand(channels, 1, window_size, window_size).contiguous()
        return window

    def forward(self, pred, target):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        window = self.window.to(pred.device)
        pad = self.window_size // 2

        mu1 = F.conv2d(pred, window, padding=pad, groups=self.channels)
        mu2 = F.conv2d(target, window, padding=pad, groups=self.channels)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu12 = mu1 * mu2

        sigma1_sq = F.conv2d(pred * pred, window, padding=pad, groups=self.channels) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=pad, groups=self.channels) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=pad, groups=self.channels) - mu12

        ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if self.reduction == 'mean':
            return 1.0 - ssim_map.mean()
        else:
            return 1.0 - ssim_map


class PerceptualLoss(nn.Module):
    """VGG-16 based perceptual loss.

    Extracts features from intermediate VGG-16 layers and computes
    L1 distance between predicted and target features. This encourages
    perceptually similar outputs rather than pixel-exact matches.

    Args:
        layers: List of VGG layer indices to extract features from.
            Default: [3, 8, 15, 22] = conv1_2, conv2_2, conv3_3, conv4_3
        weights: Per-layer loss weights. Default: [1.0, 1.0, 1.0, 1.0]
    """

    def __init__(self, layers=None, weights=None):
        super(PerceptualLoss, self).__init__()
        if layers is None:
            layers = [3, 8, 15, 22]
        if weights is None:
            weights = [1.0, 1.0, 1.0, 1.0]

        self.weights = weights
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features

        self.blocks = nn.ModuleList()
        prev = 0
        for layer_idx in layers:
            self.blocks.append(nn.Sequential(*list(vgg.children())[prev:layer_idx + 1]))
            prev = layer_idx + 1

        # Freeze VGG parameters
        for param in self.parameters():
            param.requires_grad = False

        # ImageNet normalization constants
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize(self, x):
        """Normalize input from [0, 1] to ImageNet statistics."""
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def forward(self, pred, target):
        pred = self._normalize(pred)
        target = self._normalize(target)

        loss = 0.0
        x = pred
        y = target
        for block, weight in zip(self.blocks, self.weights):
            x = block(x)
            y = block(y)
            loss += weight * F.l1_loss(x, y)

        return loss
