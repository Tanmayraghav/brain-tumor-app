"""
model/cnn_model.py
------------------
CNN architecture for binary brain tumour classification.

Two options are provided:
  1. BrainTumorCNN   – a lightweight custom CNN (fast to train on CPU)
  2. get_resnet_model – fine-tuned ResNet-18 (higher accuracy, still CPU-friendly)

The training script uses BrainTumorCNN by default for speed, but you can
switch to ResNet-18 for better accuracy on a larger dataset.
"""

import torch
import torch.nn as nn
import torchvision.models as models


# ──────────────────────────────────────────────
# Option 1 – Custom lightweight CNN
# ──────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv → BatchNorm → ReLU → MaxPool block."""
    def __init__(self, in_ch: int, out_ch: int, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BrainTumorCNN(nn.Module):
    """
    Lightweight custom CNN for 224×224 RGB MRI images.
    ~1.6 M parameters – trains in minutes on CPU.

    Architecture
    ------------
    Input  (3, 224, 224)
    Block1 (32, 112, 112)
    Block2 (64,  56,  56)
    Block3 (128, 28,  28)
    Block4 (256, 14,  14)
    GAP    (256,  1,   1)
    FC     512 → 256 → 2
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.4):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3,   32,  pool=True),
            ConvBlock(32,  64,  pool=True),
            ConvBlock(64,  128, pool=True),
            ConvBlock(128, 256, pool=True),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))   # Global Average Pooling

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


# ──────────────────────────────────────────────
# Option 2 – Transfer learning with ResNet-18
# ──────────────────────────────────────────────

def get_resnet_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    Load a pre-trained ResNet-18 and replace the final FC layer.
    Freezes all layers except the new head for fast fine-tuning.

    Args:
        num_classes: 2 for binary (tumour / no tumour)
        pretrained:  load ImageNet weights (recommended)

    Returns:
        PyTorch model ready for training
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False

    # Replace head – only this will be trained
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes),
    )

    return model


# ──────────────────────────────────────────────
# Utility: load saved model
# ──────────────────────────────────────────────

def load_model(model_path: str, device: torch.device = None) -> BrainTumorCNN:
    """
    Load a saved BrainTumorCNN from a .pt file.

    Args:
        model_path: path to the saved model file
        device:     torch.device to load onto (defaults to CPU)

    Returns:
        Model in eval mode
    """
    if device is None:
        device = torch.device("cpu")

    model = BrainTumorCNN(num_classes=2)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
