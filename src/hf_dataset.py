from typing import Tuple
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from datasets import load_from_disk


class HFDatasetImages(Dataset):
    """
    Loads images from a HuggingFace dataset saved with save_to_disk().
    Returns: (image_tensor, index)
    """
    def __init__(self, ds_path: str, split: str = "train", img_size: int = 256):
        self.img_size = img_size
        ds = load_from_disk(ds_path)

        if hasattr(ds, "keys"):
            self.ds = ds[split]
        else:
            self.ds = ds

        if "image" not in self.ds.column_names:
            raise RuntimeError(f"'image' column not found. Columns: {self.ds.column_names}")

    def __len__(self):
        return len(self.ds)

    def _to_tensor(self, img: Image.Image) -> torch.Tensor:
        img = img.convert("RGB").resize((self.img_size, self.img_size))
        arr = np.array(img).astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.ds[idx]
        img = sample["image"]  # PIL Image
        x = self._to_tensor(img)
        return x, idx
