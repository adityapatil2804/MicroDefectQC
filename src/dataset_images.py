import os
from typing import Tuple
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class ImageFolderDataset(Dataset):
    """
    Loads images from a directory (recursively).
    Returns: (image_tensor, path)
    """
    def __init__(self, root_dir: str, img_size: int = 256):
        self.root_dir = root_dir
        self.img_size = img_size
        self.paths = []
        for r, _, files in os.walk(root_dir):
            for f in files:
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                    self.paths.append(os.path.join(r, f))
        self.paths.sort()

        if len(self.paths) == 0:
            raise RuntimeError(f"No images found under: {root_dir}")

    def __len__(self):
        return len(self.paths)

    def _to_tensor(self, img: Image.Image) -> torch.Tensor:
        img = img.convert("RGB").resize((self.img_size, self.img_size))
        arr = np.array(img).astype(np.float32) / 255.0  # HWC
        arr = np.transpose(arr, (2, 0, 1))  # CHW
        return torch.from_numpy(arr)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        path = self.paths[idx]
        img = Image.open(path)
        x = self._to_tensor(img)
        return x, path
