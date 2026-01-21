import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.config import Config
from src.utils import set_seed, ensure_dir
from src.hf_dataset import HFDatasetImages
from src.anomaly_model import ConvAutoencoder


def main():
    cfg = Config()
    set_seed(cfg.seed)

    # Load HF dataset saved via save_to_disk()
    ds = HFDatasetImages(ds_path="data/raw/OD_MetalDAM", split="train", img_size=cfg.img_size)

    n = len(ds)
    n_train = int(0.8 * n)
    n_val = n - n_train
    train_ds, val_ds = random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(cfg.seed)
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvAutoencoder().to(device)

    criterion = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    ensure_dir(cfg.models_dir)
    best_path = os.path.join(cfg.models_dir, "ae_best.pt")
    best_val = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        for x, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs} [train]"):
            x = x.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            recon = model(x)
            loss = criterion(recon, x)
            loss.backward()
            opt.step()
            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, _ in tqdm(val_loader, desc=f"Epoch {epoch}/{cfg.epochs} [val]"):
                x = x.to(device, non_blocking=True)
                recon = model(x)
                loss = criterion(recon, x)
                val_loss += loss.item() * x.size(0)

        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch}: train_mse={train_loss:.6f} val_mse={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state": model.state_dict(), "img_size": cfg.img_size}, best_path)
            print("Saved best model to:", best_path)

    print("Done. Best val MSE:", best_val)


if __name__ == "__main__":
    main()
