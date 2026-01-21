import os
import csv
import json
import argparse
import numpy as np
import cv2
from datasets import load_from_disk
from PIL import Image
import torch

from src.anomaly_model import ConvAutoencoder
from src.utils import ensure_dir


def pil_to_tensor(img: Image.Image, img_size: int) -> torch.Tensor:
    img = img.convert("RGB").resize((img_size, img_size))
    arr = np.array(img).astype(np.float32) / 255.0
    x = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(x).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="data/raw/OD_MetalDAM", help="HF dataset path (save_to_disk folder)")
    ap.add_argument("--model", default="models/ae_best.pt")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--threshold", type=float, default=0.15)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    ensure_dir(args.outdir)
    ensure_dir(os.path.join(args.outdir, "batch_overlays"))
    ensure_dir(os.path.join(args.outdir, "batch_reports"))

    ds = load_from_disk(args.ds)["train"]

    ckpt = torch.load(args.model, map_location="cpu")
    img_size = int(ckpt.get("img_size", 256))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvAutoencoder().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    n = len(ds) if args.limit == 0 else min(args.limit, len(ds))

    csv_path = os.path.join(args.outdir, "batch_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index", "threshold", "defect_percent", "defect_count",
            "avg_defect_size_px", "max_defect_size_px", "grade",
            "overlay_path", "report_path"
        ])

        for i in range(n):
            img = ds[i]["image"]
            x = pil_to_tensor(img, img_size).to(device)

            with torch.no_grad():
                recon = model(x)

            diff = (x - recon) ** 2
            amap = diff.mean(dim=1).squeeze(0).detach().cpu().numpy()

            amap_norm = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
            defect_mask = (amap_norm >= args.threshold).astype(np.uint8)

            defect_area = int(defect_mask.sum())
            total_area = int(defect_mask.size)
            defect_percent = 100.0 * defect_area / max(total_area, 1)

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(defect_mask, connectivity=8)
            areas = [int(stats[j, cv2.CC_STAT_AREA]) for j in range(1, num_labels)]

            defect_count = len(areas)
            avg_defect_size_px = float(np.mean(areas)) if areas else 0.0
            max_defect_size_px = float(np.max(areas)) if areas else 0.0

            if defect_percent < 1.0:
                grade = "A"
            elif defect_percent < 3.0:
                grade = "B"
            else:
                grade = "C"

            # overlay
            heat = (amap_norm * 255).astype(np.uint8)
            heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
            orig = np.array(img.convert("RGB").resize((img_size, img_size)))
            orig_bgr = orig[:, :, ::-1]
            overlay = cv2.addWeighted(orig_bgr, 0.7, heat, 0.3, 0)

            overlay_path = os.path.join(args.outdir, "batch_overlays", f"idx_{i:04d}_overlay.png")
            cv2.imwrite(overlay_path, overlay)

            report = {
                "index": i,
                "threshold": args.threshold,
                "defect_area_px": defect_area,
                "total_area_px": total_area,
                "defect_percent": defect_percent,
                "defect_count": defect_count,
                "avg_defect_size_px": avg_defect_size_px,
                "max_defect_size_px": max_defect_size_px,
                "grade": grade,
                "overlay_path": overlay_path,
            }

            report_path = os.path.join(args.outdir, "batch_reports", f"idx_{i:04d}_report.json")
            with open(report_path, "w", encoding="utf-8") as jf:
                json.dump(report, jf, indent=2)

            writer.writerow([
                i, args.threshold, defect_percent, defect_count,
                avg_defect_size_px, max_defect_size_px, grade,
                overlay_path, report_path
            ])

            if (i + 1) % 10 == 0 or (i + 1) == n:
                print(f"Processed {i+1}/{n}")

    print("Saved CSV:", csv_path)
    print("Overlays:", os.path.join(args.outdir, "batch_overlays"))
    print("Reports :", os.path.join(args.outdir, "batch_reports"))


if __name__ == "__main__":
    main()
