import os
import argparse
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import numpy as np

from model import SR

# Usage:
# python3 inference.py --model_path sr_epoch100.pth --lr_dir data/LR --hr_dir data/HR --out_dir results --scale 4


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image


def save_image(tensor, path):
    tensor = tensor.squeeze().clamp(0, 1).cpu()
    image = T.ToPILImage()(tensor)
    image.save(path)


def to_numpy(img_tensor):
    """CHW Tensor → HWC Numpy"""
    img = img_tensor.squeeze().clamp(0, 1).cpu().numpy()
    img = np.transpose(img, (1, 2, 0))  # CHW → HWC
    return img


@torch.no_grad()
def inference(model_path, lr_dir, hr_dir=None, out_dir="results", scale=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(out_dir, exist_ok=True)

    # Model setup
    model = SR(scale=scale)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)

    transform = T.ToTensor()

    psnr_total = 0
    ssim_total = 0
    count = 0

    filenames = sorted(os.listdir(lr_dir))
    for fname in filenames:
        lr_img = load_image(os.path.join(lr_dir, fname))
        lr_tensor = transform(lr_img).unsqueeze(0).to(device)

        # Inference
        sr_tensor = model(lr_tensor)

        # Store output
        save_path = os.path.join(out_dir, fname)
        save_image(sr_tensor, save_path)

        # PSNR / SSIM
        if hr_dir:
            hr_img = load_image(os.path.join(hr_dir, fname))
            hr_tensor = transform(hr_img).unsqueeze(0).to(device)

            # Resize SR to match HR in case of mismatch
            if sr_tensor.shape[-2:] != hr_tensor.shape[-2:]:
                sr_tensor = F.interpolate(sr_tensor, size=hr_tensor.shape[-2:], mode='bicubic', align_corners=False)

            sr_np = to_numpy(sr_tensor)
            hr_np = to_numpy(hr_tensor)

            psnr = compare_psnr(hr_np, sr_np, data_range=1.0)
            ssim = compare_ssim(hr_np, sr_np, data_range=1.0, multichannel=True)

            psnr_total += psnr
            ssim_total += ssim
            count += 1

            print(f"[{fname}] PSNR: {psnr:.2f}, SSIM: {ssim:.4f}")

    if count > 0:
        print(f"\n📊 평균 PSNR: {psnr_total / count:.2f}")
        print(f"📊 평균 SSIM: {ssim_total / count:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Trained .pth model path")
    parser.add_argument("--lr_dir", type=str, required=True, help="Low-res image directory")
    parser.add_argument("--hr_dir", type=str, default=None, help="Optional GT high-res directory")
    parser.add_argument("--out_dir", type=str, default="results", help="Directory to save SR outputs")
    parser.add_argument("--scale", type=int, default=4, help="Upscale factor (default: 4)")

    args = parser.parse_args()

    inference(args.model_path, args.lr_dir, args.hr_dir, args.out_dir, args.scale)
