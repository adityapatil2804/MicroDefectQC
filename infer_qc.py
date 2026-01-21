import os
import json
import argparse
import numpy as np
from PIL import Image
import cv2
import torch

from src.anomaly_model import ConvAutoencoder
from src.utils import ensure_dir


def load_image(path: str, img_size: int):
    img = Image.open(path).convert("RGB").resize((img_size, img_size))
    arr = np.array(img).astype(np.float32) / 255.0
    x = np.transpose(arr, (2, 0, 1))  # CHW
    return torch.from_numpy(x).unsqueeze(0), arr  # (1,3,H,W), HWC float


def anomaly_map(x, recon):
    # per-pixel MSE across channels -> (H,W)
    diff = (x - recon) ** 2
    m = diff.mean(dim=1).squeeze(0)  # (H,W)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to microstructure image")
    ap.add_argument("--model", default="models/ae_best.pt")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--threshold", type=float, default=0.08, help="defect threshold on anomaly map (0-1)")
    args = ap.parse_args()

    ckpt = torch.load(args.model, map_location="cpu")
    img_size = int(ckpt.get("img_size", 256))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvAutoencoder().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ensure_dir(args.outdir)
    ensure_dir(os.path.join(args.outdir, "overlays"))
    ensure_dir(os.path.join(args.outdir, "reports"))

    x, orig = load_image(args.image, img_size)
    x = x.to(device)

    with torch.no_grad():
        recon = model(x)
    amap = anomaly_map(x, recon).detach().cpu().numpy()  # H,W

    # Normalize for visualization
    amap_norm = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)

    # Defect mask
    defect_mask = (amap_norm >= args.threshold).astype(np.uint8)  # 0/1

    # Metrics
    defect_area = int(defect_mask.sum())
    total_area = int(defect_mask.size)
    defect_percent = 100.0 * defect_area / max(total_area, 1)

    # Connected components for "defect count" + sizes
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(defect_mask, connectivity=8)
    # stats: [label, x, y, w, h, area]
    areas = []
    for i in range(1, num_labels):  # skip background 0
        areas.append(int(stats[i, cv2.CC_STAT_AREA]))

    defect_count = len(areas)
    avg_defect_size_px = float(np.mean(areas)) if areas else 0.0
    max_defect_size_px = float(np.max(areas)) if areas else 0.0

    # Quality grade (simple)
    if defect_percent < 1.0:
        grade = "A"
    elif defect_percent < 3.0:
        grade = "B"
    else:
        grade = "C"

    # Overlay heatmap on image
    heat = (amap_norm * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    orig_bgr = (orig[:, :, ::-1] * 255).astype(np.uint8)
    overlay = cv2.addWeighted(orig_bgr, 0.7, heat, 0.3, 0)

    base = os.path.splitext(os.path.basename(args.image))[0]
    overlay_path = os.path.join(args.outdir, "overlays", f"{base}_overlay.png")
    cv2.imwrite(overlay_path, overlay)

    report = {
        "image": args.image,
        "model": args.model,
        "img_size": img_size,
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

    report_path = os.path.join(args.outdir, "reports", f"{base}_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Saved overlay:", overlay_path)
    print("Saved report :", report_path)
    print("Grade:", grade, "| defect_percent:", round(defect_percent, 3), "%")


if __name__ == "__main__":
    main()
