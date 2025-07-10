import torch
import torch.nn as nn
import torch.nn.functional as F


# Residual Block (No BatchNorm)
class ResidualBlock(nn.Module):
    def __init__(self, num_feats):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feats, num_feats, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_feats, num_feats, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return x + out  # Residual connection


# Upsampler using PixelShuffle
class Upsampler(nn.Module):
    def __init__(self, scale, num_feats):
        super().__init__()
        modules = []
        for _ in range(int(scale).bit_length() - 1):  # e.g. 2x → 1 step, 4x → 2 steps
            modules += [
                nn.Conv2d(num_feats, 4 * num_feats, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True),
            ]
        self.upsample = nn.Sequential(*modules)

    def forward(self, x):
        return self.upsample(x)


class SR(nn.Module):
    def __init__(self, num_blocks=16, num_feats=64, scale=4, in_channels=3, out_channels=3):
        super().__init__()

        self.head = nn.Conv2d(in_channels, num_feats, kernel_size=3, padding=1)

        # Body (Residual Blocks)
        self.body = nn.Sequential(*[ResidualBlock(num_feats) for _ in range(num_blocks)])
        self.body_tail = nn.Conv2d(num_feats, num_feats, kernel_size=3, padding=1)

        # Upsampling
        self.upsampler = Upsampler(scale, num_feats)

        # Final conv
        self.tail = nn.Conv2d(num_feats, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res = self.body_tail(res)
        x = x + res  # Global residual

        x = self.upsampler(x)
        x = self.tail(x)
        return x
