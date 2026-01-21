import os
from datasets import load_from_disk

def main():
    ds_path = "data/raw/OD_MetalDAM"
    if not os.path.exists(ds_path):
        raise FileNotFoundError("Dataset not found at: " + ds_path)

    ds = load_from_disk(ds_path)

    print("Type:", type(ds))

    if hasattr(ds, "keys"):
        splits = list(ds.keys())
        print("Splits:", splits)
        split0 = splits[0]
        d = ds[split0]
        print("Using split:", split0, "| len =", len(d))
    else:
        d = ds
        print("len:", len(d))

    print("Columns:", d.column_names)

    sample = d[0]
    print("\nSample keys:", list(sample.keys()))

    for k, v in sample.items():
        extra = ""
        if hasattr(v, "size"):
            extra = " size=" + str(v.size)
        elif hasattr(v, "shape"):
            extra = " shape=" + str(v.shape)
        print("-", k, ":", type(v), extra)

if __name__ == "__main__":
    main()
