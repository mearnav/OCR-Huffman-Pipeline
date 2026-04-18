import re
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, base_ch: int = 32):
        super().__init__()
        ch = base_ch
        self.enc1 = ConvBlock(1, ch)
        self.enc2 = ConvBlock(ch, ch * 2)
        self.enc3 = ConvBlock(ch * 2, ch * 4)
        self.enc4 = ConvBlock(ch * 4, ch * 8)
        self.bottleneck = ConvBlock(ch * 8, ch * 16)

        self.up4 = nn.ConvTranspose2d(ch * 16, ch * 8, 2, stride=2)
        self.dec4 = ConvBlock(ch * 16, ch * 8)
        self.up3 = nn.ConvTranspose2d(ch * 8, ch * 4, 2, stride=2)
        self.dec3 = ConvBlock(ch * 8, ch * 4)
        self.up2 = nn.ConvTranspose2d(ch * 4, ch * 2, 2, stride=2)
        self.dec2 = ConvBlock(ch * 4, ch * 2)
        self.up1 = nn.ConvTranspose2d(ch * 2, ch, 2, stride=2)
        self.dec1 = ConvBlock(ch * 2, ch)

        self.out = nn.Conv2d(ch, 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.out(d1))


def denoise(model: UNet, img: Image.Image, device: torch.device,
            patch_size: int = 128, stride: int = 96) -> Image.Image:
    model.eval()
    gray = np.array(img.convert("L")).astype(np.float32) / 255.0
    H, W = gray.shape

    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    padded = np.pad(gray, ((0, pad_h), (0, pad_w)), mode="reflect")
    Ph, Pw = padded.shape

    output = np.zeros_like(padded)
    count = np.zeros_like(padded)

    ys = list(range(0, Ph - patch_size + 1, stride)) or [0]
    xs = list(range(0, Pw - patch_size + 1, stride)) or [0]

    with torch.no_grad():
        for y in ys:
            for x in xs:
                patch = padded[y:y + patch_size, x:x + patch_size]
                t = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                pred = model(t).squeeze().cpu().numpy()
                output[y:y + patch_size, x:x + patch_size] += pred
                count[y:y + patch_size, x:x + patch_size] += 1

    result = (output / count.clip(min=1))[:H, :W]
    return Image.fromarray((result * 255).clip(0, 255).astype(np.uint8))


class NoisyOfficeDataset(Dataset):
    def __init__(self, split: str, root: Path, patch_size: int = 256, patches_per_img: int = 16):
        tag = {"train": "TR", "val": "VA", "test": "TE"}[split]
        noisy_dir = root / "simulated_noisy_images_grayscale"
        clean_dir = root / "clean_images_grayscale"
        self.pairs = [
            (noisy_dir / f.name, clean_dir / re.sub(r"_Noise[cfpw]_", "_Clean_", f.name))
            for f in sorted(noisy_dir.glob("*.png"))
            if tag in f.stem
        ]
        self.patch_size = patch_size
        self.patches_per_img = patches_per_img
        self.is_train = split == "train"

    def __len__(self) -> int:
        return len(self.pairs) * self.patches_per_img

    def __getitem__(self, idx: int):
        noisy_path, clean_path = self.pairs[idx % len(self.pairs)]
        noisy = np.array(Image.open(noisy_path).convert("L"), dtype=np.float32) / 255.0
        clean = np.array(Image.open(clean_path).convert("L"), dtype=np.float32) / 255.0

        H, W = noisy.shape
        p = self.patch_size
        y = random.randint(0, max(H - p, 0))
        x = random.randint(0, max(W - p, 0))

        noisy_patch = noisy[y:y + p, x:x + p]
        clean_patch = clean[y:y + p, x:x + p]

        def pad(arr):
            ph = p - arr.shape[0]
            pw = p - arr.shape[1]
            return np.pad(arr, ((0, ph), (0, pw)), mode="reflect") if (ph > 0 or pw > 0) else arr

        noisy_patch = pad(noisy_patch)
        clean_patch = pad(clean_patch)

        if self.is_train:
            if random.random() > 0.5:
                noisy_patch = np.fliplr(noisy_patch).copy()
                clean_patch = np.fliplr(clean_patch).copy()
            if random.random() > 0.5:
                noisy_patch = np.flipud(noisy_patch).copy()
                clean_patch = np.flipud(clean_patch).copy()
            if random.random() < 0.3:
                noisy_patch = (noisy_patch + np.random.normal(0, 0.05, noisy_patch.shape)).clip(0, 1).astype(np.float32)

        noisy_t = torch.tensor(noisy_patch).unsqueeze(0)
        clean_t = torch.tensor(clean_patch).unsqueeze(0)
        return noisy_t, clean_t


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(epochs: int = 40, batch_size: int = 8, lr: float = 1e-4):
    device = _get_device()
    print(f"Training on: {device}")
    root = Path(__file__).parent.parent / "Simulated Noisy Office"

    # 128×128 patches — enough to capture document noise, 4× fewer pixels than 256
    train_ds = NoisyOfficeDataset("train", root, patch_size=128, patches_per_img=32)
    val_ds = NoisyOfficeDataset("val", root, patch_size=128, patches_per_img=8)
    # num_workers=0 avoids macOS spawn overhead on small datasets
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = UNet(base_ch=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            opt.zero_grad()
            pred = model(noisy)
            loss = F.mse_loss(pred, clean)
            loss.backward()
            opt.step()
            train_loss += loss.item() * noisy.size(0)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                val_loss += F.mse_loss(model(noisy), clean).item() * noisy.size(0)

        print(f"Epoch {epoch:02d}/{epochs} | train MSE {train_loss / len(train_ds):.6f} | val MSE {val_loss / len(val_ds):.6f}")

    weights_dir = Path(__file__).parent / "weights"
    weights_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), weights_dir / "denoiser.pth")
    print("Saved weights/denoiser.pth")
    return model


if __name__ == "__main__":
    train()