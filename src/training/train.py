import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

from src.models.cava_model import SemiCAVA
from src.losses.elbo import elbo_labeled, elbo_unlabeled


loss_history = []
# -------------------------------
# Mixup
# -------------------------------
def mixup(x, lam):
    idx = torch.randperm(x.size(0), device=x.device)
    mixed = lam * x + (1 - lam) * x[idx]
    return mixed, idx


# -------------------------------
# Dataset Split
# -------------------------------
def split_labeled_unlabeled(dataset, num_labeled=1000):
    indices = torch.randperm(len(dataset))
    labeled_idx = indices[:num_labeled]
    unlabeled_idx = indices[num_labeled:]
    return Subset(dataset, labeled_idx), Subset(dataset, unlabeled_idx)


# -------------------------------
# DataLoader
# -------------------------------
def get_dataloaders(batch_size=64, num_labeled=1000):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    labeled_set, unlabeled_set = split_labeled_unlabeled(dataset, num_labeled)

    labeled_loader = DataLoader(
        labeled_set, batch_size=batch_size, shuffle=True, num_workers=2
    )
    unlabeled_loader = DataLoader(
        unlabeled_set, batch_size=batch_size, shuffle=True, num_workers=2
    )

    return labeled_loader, unlabeled_loader


# -------------------------------
# Consistency Loss
# -------------------------------
def consistency_loss(p1, p2):
    return F.mse_loss(p1, p2)


# -------------------------------
# Training
# -------------------------------
def train(
    epochs=10,
    batch_size=64,
    lr=1e-3,
    num_labeled=1000,
    lambda_u=1.0,
    lambda_cons=5.0,
    zc_dim=32,
    znc_dim=32,
    num_classes=10,
    save_dir="checkpoints",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = SemiCAVA(
        zc_dim=zc_dim,
        znc_dim=znc_dim,
        num_classes=num_classes
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    labeled_loader, unlabeled_loader = get_dataloaders(batch_size, num_labeled)

    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        correct_total = 0
        count_total = 0

        for (x_l, y_l), (x_u, _) in zip(labeled_loader, unlabeled_loader):

            x_l, y_l = x_l.to(device), y_l.to(device)
            x_u = x_u.to(device)

            # -----------------------
            # LABELED
            # -----------------------
            y_onehot = F.one_hot(y_l, num_classes=10).float()
            out_l = model(x_l, y_onehot)

            loss_elbo_l = elbo_labeled(out_l, x_l)
            loss_cls = F.cross_entropy(out_l["y_logits"], y_l)

            preds = out_l["y_logits"].argmax(dim=1)
            correct_total += (preds == y_l).sum().item()
            count_total += y_l.size(0)

            # -----------------------
            # UNLABELED
            # -----------------------
            out_u = model(x_u)
            loss_elbo_u = elbo_unlabeled(out_u, x_u)

            # -----------------------
            # MIXUP CONSISTENCY
            # -----------------------
            lam = torch.distributions.Beta(0.75, 0.75).sample().item()

            x_mix, idx = mixup(x_u, lam)
            out_mix = model(x_mix)

            p1 = F.softmax(out_u["y_logits"], dim=1)
            p2 = p1[idx]

            p_mix_target = lam * p1 + (1 - lam) * p2
            p_mix = F.softmax(out_mix["y_logits"], dim=1)

            loss_cons = consistency_loss(p_mix, p_mix_target.detach())

            # -----------------------
            # TOTAL LOSS
            # -----------------------
            loss = (
                loss_elbo_l
                + loss_cls
                + lambda_u * loss_elbo_u
                + lambda_cons * loss_cons
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_acc = correct_total / count_total

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Loss: {total_loss:.2f} | "
            f"Acc: {epoch_acc:.4f}",
            flush=True,
        )

        # -----------------------
        # Save checkpoint
        # -----------------------
        if (epoch + 1) % 2 == 0:
            torch.save(model.state_dict(), f"{save_dir}/model_epoch_{epoch+1}.pth")
        
        loss_history.append(total_loss)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs+1), loss_history, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    result_dir = "results"
    os.makedirs(result_dir, exist_ok=True)
    save_path = os.path.join(result_dir, "loss_curve.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved loss curve to {save_path}")



# -------------------------------
# Entry
# -------------------------------
if __name__ == "__main__":
    train()