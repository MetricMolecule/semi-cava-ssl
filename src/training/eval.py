import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.models.cava_model import SemiCAVA


def evaluate(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])

    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128)

    model = SemiCAVA(zc_dim=32, znc_dim=32, num_classes=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            out = model(x)
            preds = out["y_logits"].argmax(dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    print(f"Test Accuracy: {correct / total:.4f}")


if __name__ == "__main__":
    evaluate("checkpoints/model_epoch_10.pth")
