"""Baseline training loop: fine-tune Faster R-CNN on SSDD."""

import argparse

import torch
from torch.utils.data import DataLoader, random_split

from dataset import SSDDDataset
from model import build_model


def collate_fn(batch):
    return tuple(zip(*batch))


def to_tensor_transform(image, target):
    from torchvision.transforms import functional as F
    return F.to_tensor(image), target


def train(data_root: str, epochs: int, batch_size: int, lr: float, output_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = SSDDDataset(data_root, transforms=to_tensor_transform)
    val_size = int(0.15 * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = build_model().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs} — train loss: {avg_loss:.4f}")

        torch.save(model.state_dict(), f"{output_dir}/checkpoint_epoch{epoch + 1}.pt")

    print("Training complete. Run evaluate.py against a checkpoint for metrics.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="../data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--output-dir", default="../outputs")
    args = parser.parse_args()

    train(args.data_root, args.epochs, args.batch_size, args.lr, args.output_dir)
