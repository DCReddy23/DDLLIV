import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=32):
        super().__init__()
        enc1_channels = base_channels
        enc2_channels = base_channels * 2
        enc3_channels = base_channels * 4
        enc4_channels = base_channels * 8

        self.enc1 = ConvBlock(in_channels, enc1_channels)
        self.enc2 = ConvBlock(enc1_channels, enc2_channels)
        self.enc3 = ConvBlock(enc2_channels, enc3_channels)
        self.enc4 = ConvBlock(enc3_channels, enc4_channels)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(enc4_channels, enc4_channels * 2)

        self.up4 = nn.ConvTranspose2d(enc4_channels * 2, enc4_channels, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(enc4_channels * 2, enc4_channels)
        self.up3 = nn.ConvTranspose2d(enc4_channels, enc3_channels, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(enc3_channels * 2, enc3_channels)
        self.up2 = nn.ConvTranspose2d(enc3_channels, enc2_channels, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(enc2_channels * 2, enc2_channels)
        self.up1 = nn.ConvTranspose2d(enc2_channels, enc1_channels, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(enc1_channels * 2, enc1_channels)

        self.out_conv = nn.Conv2d(enc1_channels, out_channels, kernel_size=1)

    @staticmethod
    def _align_and_concat(x, skip):
        """Resize decoder feature to skip size, then concatenate along channels."""
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return torch.cat([x, skip], dim=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self.dec4(self._align_and_concat(d4, e4))
        d3 = self.up3(d4)
        d3 = self.dec3(self._align_and_concat(d3, e3))
        d2 = self.up2(d3)
        d2 = self.dec2(self._align_and_concat(d2, e2))
        d1 = self.up1(d2)
        d1 = self.dec1(self._align_and_concat(d1, e1))
        return self.out_conv(d1)
