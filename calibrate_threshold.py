import argparse
import numpy as np
import torch
from datasets import load_from_disk
from tqdm import tqdm
from PIL import Image

from src.anomaly_model import ConvAutoencoder


def pil_to_tensor(img: Image.Image, img_size: int) -> torch.Tensor:
    img = img.convert("RGB").resize((img_size, img_size))
    arr = np.array(img).astype(np.float32) / 255.0
    x = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(x).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="data/raw/OD_MetalDAM")
    ap.add_argument("--model", default="models/ae_best.pt")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--percentile", type=float, default=95.0, help="threshold percentile for anomaly map")
    args = ap.parse_args()

    ds = load_from_disk(args.ds)["train"]

    ckpt = torch.load(args.model, map_location="cpu")
    img_size = int(ckpt.get("img_size", 256))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvAutoencoder().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    n = len(ds) if args.limit == 0 else min(args.limit, len(ds))

    all_pixels = []

    for i in tqdm(range(n), desc="Collecting anomaly pixels"):
        img = ds[i]["image"]
        x = pil_to_tensor(img, img_size).to(device)

        with torch.no_grad():
            recon = model(x)

        diff = (x - recon) ** 2
        amap = diff.mean(dim=1).squeeze(0).detach().cpu().numpy()

        amap_norm = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
        all_pixels.append(amap_norm.reshape(-1))

    all_pixels = np.concatenate(all_pixels, axis=0)

    thr = float(np.percentile(all_pixels, args.percentile))
    print("Auto threshold =", thr)
    print("Use it like:")
    print(f'python batch_qc.py --threshold {thr:.4f}')


if __name__ == "__main__":
    main()
