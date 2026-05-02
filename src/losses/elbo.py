import torch
import torch.nn.functional as F


def kl_divergence(mu, logvar):
    """
    KL divergence between N(mu, sigma) and N(0,1)
    """
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)


def reconstruction_loss(x, recon_x):
    """
    Use MSE (works well for MNIST-like datasets)
    """
    return F.mse_loss(recon_x, x, reduction="none").view(x.size(0), -1).sum(dim=1)


def elbo_labeled(outputs, x):
    """
    ELBO = Reconstruction + KL(zc) + KL(znc)
    """
    recon = outputs["recon"]
    mu_c, logvar_c = outputs["mu_c"], outputs["logvar_c"]
    mu_nc, logvar_nc = outputs["mu_nc"], outputs["logvar_nc"]

    recon_loss = reconstruction_loss(x, recon)
    kl_c = kl_divergence(mu_c, logvar_c)
    kl_nc = kl_divergence(mu_nc, logvar_nc)

    elbo = recon_loss + kl_c + kl_nc

    return elbo.mean()


def elbo_unlabeled(outputs, x):
    """
    Same structure, but uses pseudo-label internally (handled in model)
    """
    recon = outputs["recon"]
    mu_c, logvar_c = outputs["mu_c"], outputs["logvar_c"]
    mu_nc, logvar_nc = outputs["mu_nc"], outputs["logvar_nc"]

    recon_loss = reconstruction_loss(x, recon)
    kl_c = kl_divergence(mu_c, logvar_c)
    kl_nc = kl_divergence(mu_nc, logvar_nc)

    elbo = recon_loss + kl_c + kl_nc

    return elbo.mean()



def elbo_loss(outputs, x, beta=1.0):
    """
    Generic ELBO (if you want scaling)
    """
    recon = outputs["recon"]
    mu_c, logvar_c = outputs["mu_c"], outputs["logvar_c"]
    mu_nc, logvar_nc = outputs["mu_nc"], outputs["logvar_nc"]

    recon_loss = reconstruction_loss(x, recon)
    kl_c = kl_divergence(mu_c, logvar_c)
    kl_nc = kl_divergence(mu_nc, logvar_nc)

    elbo = recon_loss + beta * (kl_c + kl_nc)

    return elbo.mean()
