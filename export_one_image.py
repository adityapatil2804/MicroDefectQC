import os
from datasets import load_from_disk

def main():
    ds = load_from_disk("data/raw/OD_MetalDAM")["train"]

    out_dir = "data/processed/sample_images"
    os.makedirs(out_dir, exist_ok=True)

    img = ds[0]["image"]   # PIL image
    out_path = os.path.join(out_dir, "micrograph_0.jpg")
    img.save(out_path)

    print("Saved:", out_path)

if __name__ == "__main__":
    main()
