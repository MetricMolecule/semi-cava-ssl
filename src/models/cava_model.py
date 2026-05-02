import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder
from .classifier import Classifier
from .decoder import Decoder


class LatentHead(nn.Module):
    def __init__(self, in_dim, z_dim):
        super().__init__()
        self.mu = nn.Linear(in_dim, z_dim)
        self.logvar = nn.Linear(in_dim, z_dim)

    def forward(self, h):
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar


class SemiCAVA(nn.Module):
    def __init__(
        self,
        zc_dim=32,
        znc_dim=32,
        num_classes=10,
        encoder_kwargs=None
    ):
        super().__init__()

        if encoder_kwargs is None:
            encoder_kwargs = {}

        self.encoder = Encoder(**encoder_kwargs)
        h_dim = self.encoder.out_dim

        self.zc_head = LatentHead(h_dim, zc_dim)
        self.znc_head = LatentHead(h_dim, znc_dim)

        self.classifier = Classifier(zc_dim, num_classes)

        self.decoder = Decoder(zc_dim, znc_dim, num_classes)

        self.num_classes = num_classes

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y_onehot=None):
        h = self.encoder(x)

        mu_c, logvar_c = self.zc_head(h)
        mu_nc, logvar_nc = self.znc_head(h)

        logvar_c = torch.clamp(logvar_c, -10, 10)
        logvar_nc = torch.clamp(logvar_nc, -10, 10)

        zc = self.reparameterize(mu_c, logvar_c)
        znc = self.reparameterize(mu_nc, logvar_nc)

        y_logits = self.classifier(zc)
        y_probs = F.softmax(y_logits, dim=1)

        if y_onehot is None:
            y_hat = torch.argmax(y_logits, dim=1)
            y_onehot = F.one_hot(y_hat, num_classes=self.num_classes).float().detach()

        recon = self.decoder(zc, znc, y_onehot)

        return {
            "zc": zc,
            "znc": znc,
            "y_logits": y_logits,
            "y_probs": y_probs,
            "recon": recon,
            "mu_c": mu_c,
            "logvar_c": logvar_c,
            "mu_nc": mu_nc,
            "logvar_nc": logvar_nc
        }
