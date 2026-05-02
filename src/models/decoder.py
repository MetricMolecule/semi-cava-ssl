import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, zc_dim, znc_dim, K):
        super().__init__()
        in_dim = zc_dim + znc_dim + K

        self.fc = nn.Sequential(
            nn.Linear(in_dim, 256 * 7 * 7),
            nn.ReLU(True)
        )

        self.up = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, zc, znc, y_onehot):
        z = torch.cat([zc, znc, y_onehot], dim=1)
        x = self.fc(z).view(-1, 256, 7, 7)
        return self.up(x)
