# MicroDefectQC (Microstructure Defect Quality Control)

MicroDefectQC is a deep-learning based quality control tool for **microstructure images**.
It learns the normal texture/pattern distribution using an **Autoencoder (unsupervised)** and flags abnormal regions using **reconstruction error heatmaps**.

This project generates:
- Defect heatmap overlay (visual localization)
- QC report JSON (defect %, defect count, defect size stats, grade)
- Batch CSV summary for the full dataset

---

## Key Features
-  Unsupervised anomaly detection (no labels/masks required)
-  Defect localization via pixel-level anomaly heatmaps
-  Automated QC scoring + A/B/C grading
-  Batch processing + CSV export for reporting
-  Works on any microstructure image input

---

## Dataset Used
HuggingFace dataset: **Voxel51/OD_MetalDAM**  
(Downloaded locally using `datasets` and saved using `save_to_disk()`)

---

## Tech Stack
- Python 3.13
- PyTorch (GPU supported)
- HuggingFace Datasets
- OpenCV
- NumPy, Pillow, tqdm

---

## Installation
pip install torch torchvision datasets opencv-python tqdm pillow numpy

MicroDefectQC/
│── data/
│   ├── raw/
│   ├── processed/
│   ├── splits/
│── models/
│── outputs/
│   ├── overlays/
│   ├── reports/
│   ├── batch_overlays/
│   ├── batch_reports/
│── src/
│── download_dataset.py
│── train_anomaly.py
│── infer_qc.py
│── batch_qc.py
│── calibrate_threshold.py
│── export_one_image.py
│── README.md

