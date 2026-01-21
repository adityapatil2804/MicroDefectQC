import os

FOLDERS = [
    "data/raw",
    "data/processed",
    "data/splits",
    "models",
    "src",
    "outputs/overlays",
    "outputs/reports",
]

def main():
    for f in FOLDERS:
        os.makedirs(f, exist_ok=True)
        print("Created:", f)

if __name__ == "__main__":
    main()
