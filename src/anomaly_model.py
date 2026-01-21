import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    """
    Simple convolutional autoencoder.
    Train on "normal" images. High reconstruction error => anomaly.
    Input: (B,3,H,W) with H=W=img_size (256)
    """
    def __init__(self, in_ch: int = 3, base: int = 32):
        super().__init__()

        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 128
            nn.Conv2d(base, base * 2, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64
            nn.Conv2d(base * 2, base * 4, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32
            nn.Conv2d(base * 4, base * 8, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16
        )

        # Decoder
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2), nn.ReLU(inplace=True),  # 32
            nn.Conv2d(base * 4, base * 4, 3, padding=1), nn.ReLU(inplace=True),

            nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2), nn.ReLU(inplace=True),  # 64
            nn.Conv2d(base * 2, base * 2, 3, padding=1), nn.ReLU(inplace=True),

            nn.ConvTranspose2d(base * 2, base, 2, stride=2), nn.ReLU(inplace=True),      # 128
            nn.Conv2d(base, base, 3, padding=1), nn.ReLU(inplace=True),

            nn.ConvTranspose2d(base, base, 2, stride=2), nn.ReLU(inplace=True),          # 256
            nn.Conv2d(base, in_ch, 3, padding=1),
            nn.Sigmoid(),  # outputs in [0,1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.enc(x)
        out = self.dec(z)
        return out
