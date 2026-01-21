from dataclasses import dataclass

@dataclass
class Config:
    # Paths
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    splits_dir: str = "data/splits"
    models_dir: str = "models"
    outputs_dir: str = "outputs"

    # Image settings
    img_size: int = 256
    batch_size: int = 8
    num_workers: int = 2

    # Training
    lr: float = 1e-4
    epochs: int = 15
    seed: int = 42

    # Dataset split
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
