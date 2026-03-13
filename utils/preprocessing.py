"""
utils/preprocessing.py
-----------------------
Image preprocessing utilities for the Brain Tumor Detection System.
Handles resizing, normalization, augmentation, and tensor conversion.
"""

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import io


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
IMG_SIZE = 224          # Input size expected by the CNN
MEAN = [0.485, 0.456, 0.406]   # ImageNet mean (used for transfer-learning fine-tuning)
STD  = [0.229, 0.224, 0.225]   # ImageNet std


# ──────────────────────────────────────────────
# Transform pipelines
# ──────────────────────────────────────────────

def get_train_transforms() -> transforms.Compose:
    """
    Returns augmentation + normalization pipeline used during training.
    Augmentations help the model generalise to unseen scan orientations/contrasts.
    """
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def get_val_transforms() -> transforms.Compose:
    """
    Returns deterministic pipeline used during validation / inference.
    No augmentation – only resize + normalise.
    """
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


# ──────────────────────────────────────────────
# Image loading helpers
# ──────────────────────────────────────────────

def load_image_from_path(path: str) -> Image.Image:
    """Load an image from a file path and convert to RGB PIL Image."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image at path: {path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


def load_image_from_bytes(file_bytes: bytes) -> Image.Image:
    """
    Load an image from raw bytes (e.g., from a Streamlit UploadedFile).
    Returns a PIL Image in RGB mode.
    """
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return img
    except Exception as e:
        raise ValueError(f"Invalid image file: {e}")


def preprocess_for_inference(pil_image: Image.Image) -> torch.Tensor:
    """
    Apply val-time transforms and add batch dimension.
    Returns a FloatTensor of shape (1, 3, IMG_SIZE, IMG_SIZE).
    """
    transform = get_val_transforms()
    tensor = transform(pil_image)          # (3, H, W)
    return tensor.unsqueeze(0)             # (1, 3, H, W)


# ──────────────────────────────────────────────
# OpenCV-based enhancement (optional display)
# ──────────────────────────────────────────────

def enhance_mri_display(pil_image: Image.Image) -> np.ndarray:
    """
    Apply mild CLAHE contrast enhancement for better display in the UI.
    Does NOT affect model input – purely cosmetic.
    Returns an RGB numpy array.
    """
    img = np.array(pil_image)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_gray)

    # Convert back to RGB for display
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    return enhanced_rgb


def validate_image(pil_image: Image.Image) -> bool:
    """
    Basic sanity checks on the uploaded image.
    Returns True if the image looks like a valid grayscale/colour scan.
    """
    w, h = pil_image.size
    if w < 32 or h < 32:
        return False
    if w > 4096 or h > 4096:
        return False
    return True
