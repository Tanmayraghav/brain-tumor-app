"""
train.py
--------
Training script for the Brain Tumor Detection CNN.

Usage
-----
    python train.py

The script expects the following dataset layout:

    dataset/
    ├── tumor/          ← MRI images WITH a tumour
    └── no_tumor/       ← MRI images WITHOUT a tumour

A 80/20 train-validation split is applied automatically.

Download a suitable dataset from Kaggle:
  https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection

Place the images in the folders above, then run this script.
The trained model is saved to  model/brain_tumor_model.pt
"""

import os
import sys
import time
import copy
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder

# ── Local imports ──
sys.path.insert(0, os.path.dirname(__file__))
from model.cnn_model import BrainTumorCNN
from utils.preprocessing import get_train_transforms, get_val_transforms, IMG_SIZE


# ──────────────────────────────────────────────
# Configuration (edit here or via CLI args)
# ──────────────────────────────────────────────
DEFAULT_CONFIG = {
    "dataset_dir":  "dataset",
    "model_dir":    "model",
    "model_name":   "brain_tumor_model.pt",
    "epochs":       25,
    "batch_size":   16,
    "lr":           1e-3,
    "weight_decay": 1e-4,
    "val_split":    0.2,
    "num_workers":  0,      # 0 = main process (safe on Windows/macOS)
    "patience":     5,      # early-stopping patience
}


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def print_banner():
    print("\n" + "═" * 55)
    print("   🧠  Brain Tumor Detection – Training Script")
    print("═" * 55)


def build_dataloaders(cfg: dict):
    """
    Load the full dataset with train augmentations, split into
    train / val subsets, and return DataLoaders + class names.
    """
    dataset_path = cfg["dataset_dir"]

    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(
            f"Dataset directory '{dataset_path}' not found.\n"
            "Please create  dataset/tumor/  and  dataset/no_tumor/  "
            "and populate them with MRI images."
        )

    # Load full dataset with train transforms (augmentations removed from val below)
    full_dataset = ImageFolder(root=dataset_path, transform=get_train_transforms())
    class_names  = full_dataset.classes
    total        = len(full_dataset)

    if total == 0:
        raise ValueError("No images found in dataset directory.")

    # Split
    val_size   = int(total * cfg["val_split"])
    train_size = total - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(42))

    # Override val subset transform (no augmentation)
    val_ds.dataset = copy.deepcopy(full_dataset)
    val_ds.dataset.transform = get_val_transforms()

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True,  num_workers=cfg["num_workers"],
                              pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"],
                              shuffle=False, num_workers=cfg["num_workers"],
                              pin_memory=False)

    print(f"\n  Dataset : {total} images  |  Classes: {class_names}")
    print(f"  Train   : {train_size}   |  Val: {val_size}")
    print(f"  Batch   : {cfg['batch_size']}  |  Image size: {IMG_SIZE}×{IMG_SIZE}\n")

    return train_loader, val_loader, class_names


def run_epoch(model, loader, criterion, optimizer, device, phase="train"):
    """Run one epoch. Returns (loss, accuracy)."""
    is_train = phase == "train"
    model.train() if is_train else model.eval()

    running_loss = 0.0
    correct      = 0
    total        = 0

    with torch.set_grad_enabled(is_train):
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            if is_train:
                optimizer.zero_grad()

            outputs = model(inputs)
            loss    = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds         = outputs.argmax(dim=1)
            correct      += (preds == labels).sum().item()
            total        += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc


# ──────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────

def train(cfg: dict):
    print_banner()

    device = torch.device("cpu")
    print(f"  Device  : {device}")

    # Data
    train_loader, val_loader, class_names = build_dataloaders(cfg)

    # Model
    model     = BrainTumorCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                     factor=0.5, patience=3)

    best_val_acc  = 0.0
    best_weights  = copy.deepcopy(model.state_dict())
    patience_ctr  = 0

    print(f"  {'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Val Loss':>8}  {'Val Acc':>7}  {'LR':>8}")
    print("  " + "─" * 55)

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()

        tr_loss, tr_acc = run_epoch(model, train_loader, criterion,
                                    optimizer, device, phase="train")
        vl_loss, vl_acc = run_epoch(model, val_loader, criterion,
                                    optimizer, device, phase="val")
        scheduler.step(vl_acc)

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]

        print(f"  {epoch:>5}  {tr_loss:>10.4f}  {tr_acc:>8.2%}  "
              f"{vl_loss:>8.4f}  {vl_acc:>6.2%}  {lr_now:>8.2e}  "
              f"({elapsed:.1f}s)")

        # Save best model
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_weights = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= cfg["patience"]:
                print(f"\n  ⚡ Early stopping triggered at epoch {epoch}.")
                break

    # Restore best weights and save
    model.load_state_dict(best_weights)
    os.makedirs(cfg["model_dir"], exist_ok=True)
    save_path = os.path.join(cfg["model_dir"], cfg["model_name"])
    torch.save(model.state_dict(), save_path)

    print("\n" + "═" * 55)
    print(f"  ✅  Training complete!")
    print(f"  Best Val Accuracy : {best_val_acc:.2%}")
    print(f"  Model saved to    : {save_path}")
    print("═" * 55 + "\n")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train Brain Tumor CNN")
    parser.add_argument("--epochs",      type=int,   default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch_size",  type=int,   default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr",          type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--dataset_dir", type=str,   default=DEFAULT_CONFIG["dataset_dir"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = DEFAULT_CONFIG.copy()
    cfg.update(vars(args))
    train(cfg)
