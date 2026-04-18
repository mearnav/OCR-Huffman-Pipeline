import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from torchvision import datasets, transforms
from pathlib import Path
from PIL import Image

# 47 classes: 0-9, A-Z, plus 11 lowercase letters
NUM_CLASSES = 47
EMNIST_MEAN = (0.1751,)
EMNIST_STD = (0.3332,)
EMNIST_LABELS = [
    '0','1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'a','b','d','e','f','g','h','n','q','r','t',
]


class _FixEMNIST:
    """EMNIST images are stored transposed (column-major); PIL.TRANSPOSE (5) fixes it."""
    def __call__(self, img):
        return img.transpose(5)


class OCRNet(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        return self.classifier(x)


def predict(model: OCRNet, img_tensor: torch.Tensor, device: torch.device) -> str:
    model.eval()
    with torch.no_grad():
        logits = model(img_tensor.unsqueeze(0).to(device))
        return EMNIST_LABELS[logits.argmax(dim=1).item()].lower()


def predict_batch(model: OCRNet, tensors: list[torch.Tensor], device: torch.device) -> list[str]:
    model.eval()
    with torch.no_grad():
        batch = torch.stack(tensors).to(device)
        indices = model(batch).argmax(dim=1).tolist()
        return [EMNIST_LABELS[i].lower() for i in indices]


class NoisyOfficeCharDataset(Dataset):
    """Labeled character crops extracted from NoisyOffice by prepare_dataset.py."""
    def __init__(self, split: str, root: Path, transform=None):
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        split_dir = root / split
        if not split_dir.exists():
            return
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir() or class_dir.name not in EMNIST_LABELS:
                continue
            label_idx = EMNIST_LABELS.index(class_dir.name)
            for img_path in class_dir.glob("*.png"):
                self.samples.append((img_path, label_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, label


class _AddGaussianNoise:
    def __init__(self, std: float = 0.15):
        self.std = std

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return (t + torch.randn_like(t) * self.std).clamp(0, 1)


class _AddSaltAndPepper:
    def __init__(self, rate: float = 0.05):
        self.rate = rate

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        mask = torch.rand_like(t)
        t = t.clone()
        t[mask < self.rate / 2] = 0.0
        t[mask > 1 - self.rate / 2] = 1.0
        return t


def _noise_transform(noise_type: str | None, std: float = 0.15, rate: float = 0.05):
    base = transforms.Compose([_FixEMNIST(), transforms.ToTensor()])
    if noise_type == "gaussian":
        return transforms.Compose([base, _AddGaussianNoise(std)])
    if noise_type == "salt_pepper":
        return transforms.Compose([base, _AddSaltAndPepper(rate)])
    return base


def _eval_accuracy(model: OCRNet, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(epochs: int = 20, batch_size: int = 256, lr: float = 1e-3):
    device = _get_device()
    print(f"Training on: {device}")
    model = OCRNet().to(device)

    emnist_train_tf = transforms.Compose([
        _FixEMNIST(),
        transforms.RandomAffine(degrees=5, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.RandomApply([_AddGaussianNoise(0.15)], p=0.4),
        transforms.RandomApply([_AddSaltAndPepper(0.05)], p=0.3),
        transforms.Normalize(EMNIST_MEAN, EMNIST_STD),
    ])
    # NoisyOffice crops are already 28x28, correctly oriented, no _FixEMNIST needed
    no_train_tf = transforms.Compose([
        transforms.RandomAffine(degrees=3, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.RandomApply([_AddGaussianNoise(0.1)], p=0.3),
        transforms.Normalize(EMNIST_MEAN, EMNIST_STD),
    ])
    no_val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(EMNIST_MEAN, EMNIST_STD),
    ])
    emnist_val_tf = transforms.Compose([
        _FixEMNIST(),
        transforms.ToTensor(),
        transforms.Normalize(EMNIST_MEAN, EMNIST_STD),
    ])

    data_root = Path(__file__).parent.parent / "data"
    train_ds_balanced = datasets.EMNIST(data_root, split="balanced", train=True, download=True, transform=emnist_train_tf)
    train_ds_mnist = datasets.EMNIST(data_root, split="mnist", train=True, download=True, transform=emnist_train_tf)
    train_ds = ConcatDataset([train_ds_balanced, train_ds_mnist])

    crops_root = data_root / "noisyoffice_crops"
    no_train = NoisyOfficeCharDataset("train", crops_root, no_train_tf)
    no_val = NoisyOfficeCharDataset("val", crops_root, no_val_tf)

    if len(no_train) > 0:
        print(f"NoisyOffice crops: {len(no_train)} train, {len(no_val)} val")
        train_ds = ConcatDataset([train_ds, no_train])
        val_ds = no_val
        val_source = "NoisyOffice"
    else:
        print("No NoisyOffice crops found — run prepare_dataset.py first. Training on EMNIST only.")
        val_ds = datasets.EMNIST(data_root, split="balanced", train=False, download=True, transform=emnist_val_tf)
        val_source = "EMNIST"

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=4)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(imgs), labels)
            loss.backward()
            opt.step()
            total_loss += loss.item() * imgs.size(0)
        scheduler.step()
        acc = _eval_accuracy(model, val_loader, device)
        print(f"Epoch {epoch:02d}/{epochs} | loss {total_loss / len(train_ds):.4f} | val acc [{val_source}] {acc:.4f}")

    # EMNIST MNIST split also uses transposed storage — _FixEMNIST required here too
    mnist_val_tf = transforms.Compose([
        _FixEMNIST(),
        transforms.ToTensor(),
        transforms.Normalize(EMNIST_MEAN, EMNIST_STD),
    ])
    mnist_val_ds = datasets.EMNIST(data_root, split="mnist", train=False, download=True, transform=mnist_val_tf)
    digit_acc = _eval_accuracy(model, DataLoader(mnist_val_ds, batch_size=512, num_workers=4), device)
    print(f"Digit acc (EMNIST MNIST val): {digit_acc:.4f}")

    weights_dir = Path(__file__).parent / "weights"
    weights_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), weights_dir / "ocr.pth")
    (weights_dir / "metrics.json").write_text(json.dumps({"emnist_digit_val_acc": round(digit_acc, 4)}))
    print("Saved weights/ocr.pth")
    return model


if __name__ == "__main__":
    train()