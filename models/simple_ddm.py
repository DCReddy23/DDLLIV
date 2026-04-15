import os
import torch
import torch.nn as nn

from models.simple_unet import SimpleUNet
from utils.metrics import compute_psnr, compute_ssim
from utils.simple_losses import SimpleL1L2Loss


class SimpleDDM(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=32):
        super().__init__()
        self.unet = SimpleUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
        )

    def forward(self, x):
        return self.unet(x)


class SimpleDDMTrainer:
    def __init__(
        self,
        model,
        device,
        lr=1e-4,
        weight_decay=0.0,
        step_size=20,
        gamma=0.5,
        l1_weight=1.0,
        l2_weight=1.0,
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=step_size,
            gamma=gamma,
        )
        self.loss_fn = SimpleL1L2Loss(l1_weight=l1_weight, l2_weight=l2_weight)

    def train_one_epoch(self, loader):
        self.model.train()
        running = {"total": 0.0, "l1": 0.0, "l2": 0.0}

        for batch, _ in loader:
            low = batch[:, :3].to(self.device)
            high = batch[:, 3:].to(self.device)

            pred = self.model(low)
            loss, parts = self.loss_fn(pred, high)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running["total"] += loss.item()
            running["l1"] += parts["l1"].item()
            running["l2"] += parts["l2"].item()

        num_batches = max(len(loader), 1)
        return {k: v / num_batches for k, v in running.items()}

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        running = {"total": 0.0, "l1": 0.0, "l2": 0.0, "psnr": 0.0, "ssim": 0.0}

        for batch, _ in loader:
            low = batch[:, :3].to(self.device)
            high = batch[:, 3:].to(self.device)

            pred = self.model(low)
            loss, parts = self.loss_fn(pred, high)

            pred_clamped = pred.clamp(0.0, 1.0)
            high_clamped = high.clamp(0.0, 1.0)

            running["total"] += loss.item()
            running["l1"] += parts["l1"].item()
            running["l2"] += parts["l2"].item()
            running["psnr"] += compute_psnr(pred_clamped, high_clamped)
            running["ssim"] += compute_ssim(pred_clamped, high_clamped, channels=pred_clamped.shape[1])

        num_batches = max(len(loader), 1)
        return {k: v / num_batches for k, v in running.items()}

    @torch.no_grad()
    def predict(self, x):
        self.model.eval()
        return self.model(x.to(self.device))

    def step_scheduler(self):
        self.scheduler.step()

    def current_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def save_checkpoint(self, path, epoch, best_psnr):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "best_psnr": best_psnr,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        return checkpoint.get("epoch", 0), checkpoint.get("best_psnr", float("-inf"))
