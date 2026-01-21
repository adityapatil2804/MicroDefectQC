from datasets import load_dataset
import os

def main():
    save_dir = "data/raw/OD_MetalDAM"
    os.makedirs(save_dir, exist_ok=True)

    print("Downloading dataset...")
    ds = load_dataset("Voxel51/OD_MetalDAM")

    ds.save_to_disk(save_dir)
    print("Saved to:", save_dir)

if __name__ == "__main__":
    main()
