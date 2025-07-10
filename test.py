import torch
import torch.nn as nn
import torch.nn.functional as F

from model import SR


if __name__ == "__main__":
    model = SR(num_blocks=8, num_feats=64, scale=4)
    model.eval()

    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        y = model(x)

    print(f"Input: {x.shape}, Output: {y.shape}")
    # Output should be (1, 3, 256, 256)
