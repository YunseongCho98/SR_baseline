import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class SRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, size=1024):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.filenames = sorted(os.listdir(lr_dir))
        self.size = size

        # Low-resolution images are resized to 1/4 of the original size
        self.lr_transform = T.Compose([
            T.Resize((self.size // 4, self.size // 4), interpolation=Image.BICUBIC),
            T.ToTensor()
        ])
        
        # High-resolution images are resized to the same size as low-resolution images
        self.hr_transform = T.Compose([
            T.Resize((self.size, self.size), interpolation=Image.BICUBIC),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        lr_path = os.path.join(self.lr_dir, fname)
        hr_path = os.path.join(self.hr_dir, fname)

        lr_img = Image.open(lr_path).convert("RGB")
        hr_img = Image.open(hr_path).convert("RGB")

        lr_tensor = self.lr_transform(lr_img)
        hr_tensor = self.hr_transform(hr_img)

        return lr_tensor, hr_tensor

