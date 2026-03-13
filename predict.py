"""
predict.py
----------
Inference module for the Brain Tumor Detection System.

Can be used:
  • As a library  –  from predict import BrainTumorPredictor
  • From CLI      –  python predict.py --image path/to/mri.jpg
"""

import os
import sys
import argparse
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from model.cnn_model import BrainTumorCNN, load_model
from utils.preprocessing import (
    load_image_from_path,
    load_image_from_bytes,
    preprocess_for_inference,
    validate_image,
)


# ──────────────────────────────────────────────
# Result dataclass
# ──────────────────────────────────────────────

@dataclass
class PredictionResult:
    label:          str    # "Tumor Detected" | "No Tumor Detected"
    confidence:     float  # 0.0 – 1.0
    tumor_prob:     float  # raw probability for tumour class
    no_tumor_prob:  float  # raw probability for no-tumour class
    is_tumor:       bool

    def __str__(self):
        return (
            f"Result     : {self.label}\n"
            f"Confidence : {self.confidence:.1%}\n"
            f"Tumor prob : {self.tumor_prob:.1%}\n"
            f"NoTumor prob: {self.no_tumor_prob:.1%}"
        )


# ──────────────────────────────────────────────
# Predictor class
# ──────────────────────────────────────────────

class BrainTumorPredictor:
    """
    Wraps the trained CNN for easy inference.

    Args:
        model_path: path to the saved .pt file
        device:     'cpu' (default) or 'cuda'
    """

    # Class indices as produced by ImageFolder (alphabetical order):
    # 0 = no_tumor, 1 = tumor
    CLASS_NAMES = ["no_tumor", "tumor"]

    def __init__(self, model_path: str, device: str = "cpu"):
        self.device     = torch.device(device)
        self.model_path = model_path

        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Model file not found: '{model_path}'\n"
                "Please run  python train.py  first."
            )

        self.model = load_model(model_path, device=self.device)

    # ── Core inference ──────────────────────────

    def predict(self, pil_image: Image.Image) -> PredictionResult:
        """
        Run inference on a PIL Image.

        Returns a PredictionResult with label, confidence, and raw probabilities.
        """
        if not validate_image(pil_image):
            raise ValueError(
                "Image is too small or too large for inference. "
                "Please upload a valid MRI scan."
            )

        tensor = preprocess_for_inference(pil_image).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)               # (1, 2)
            probs  = F.softmax(logits, dim=1)[0]      # (2,)

        no_tumor_prob = probs[0].item()
        tumor_prob    = probs[1].item()

        is_tumor   = tumor_prob > no_tumor_prob
        label      = "Tumor Detected" if is_tumor else "No Tumor Detected"
        confidence = tumor_prob if is_tumor else no_tumor_prob

        return PredictionResult(
            label=label,
            confidence=confidence,
            tumor_prob=tumor_prob,
            no_tumor_prob=no_tumor_prob,
            is_tumor=is_tumor,
        )

    def predict_from_path(self, image_path: str) -> PredictionResult:
        """Load image from file path and predict."""
        img = load_image_from_path(image_path)
        return self.predict(img)

    def predict_from_bytes(self, file_bytes: bytes) -> PredictionResult:
        """Load image from raw bytes and predict."""
        img = load_image_from_bytes(file_bytes)
        return self.predict(img)


# ──────────────────────────────────────────────
# Demo model (for UI testing without training)
# ──────────────────────────────────────────────

def create_demo_model(save_path: str):
    """
    Save an UNTRAINED model so the Streamlit app can be tested
    before a real training run completes.

    ⚠️  Predictions will be random – use only for UI testing!
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model = BrainTumorCNN(num_classes=2)
    torch.save(model.state_dict(), save_path)
    print(f"[Demo] Untrained model saved to {save_path}")
    print("[Demo] Run  python train.py  for a real trained model.")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Brain Tumor Predictor (CLI)")
    parser.add_argument("--image",   required=True,  help="Path to MRI image")
    parser.add_argument("--model",   default="model/brain_tumor_model.pt",
                        help="Path to trained .pt model")
    parser.add_argument("--device",  default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    predictor = BrainTumorPredictor(args.model, device=args.device)
    result    = predictor.predict_from_path(args.image)

    print("\n" + "═" * 40)
    print("  Brain Tumor Detection – CLI Result")
    print("═" * 40)
    print(result)
    print("═" * 40 + "\n")


if __name__ == "__main__":
    main()
