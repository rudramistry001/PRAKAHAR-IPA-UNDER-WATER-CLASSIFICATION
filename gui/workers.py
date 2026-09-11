"""
AquaVision AI — Async Workers Module
================────────────────=====
PySide6 QThread worker classes for model loading, single image inference,
and batch directory processing without locking the UI main thread.
"""

import os
import glob
import time
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PySide6.QtCore import QThread, Signal

from src.models import FishClassifier
from src.transforms import get_val_transforms, get_tta_transforms
from gui.species_data import get_species_info

NUM_CLASSES = 23


class ModelLoaderWorker(QThread):
    """Loads a PyTorch checkpoint asynchronously."""
    loaded = Signal(object, str, str)  # (model, arch_name, info_msg)
    error = Signal(str)

    def __init__(self, checkpoint_path, device):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.device = device

    def run(self):
        try:
            if not os.path.exists(self.checkpoint_path):
                # Fallback to initialized model for testing/demo mode
                model_name = "convnext_base" if "convnext" in self.checkpoint_path.lower() else "efficientnet_v2m"
                try:
                    model = FishClassifier(model_name=model_name, num_classes=NUM_CLASSES, pretrained=True)
                    msg = f"Initialized {model_name} (Pretrained backbone)"
                except Exception as err:
                    model = FishClassifier(model_name=model_name, num_classes=NUM_CLASSES, pretrained=False)
                    msg = f"Initialized {model_name} (Demo mode - train model via train.py)"
                model = model.to(self.device).eval()
                self.loaded.emit(model, model_name, msg)
                return

            ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            model_name = ckpt.get('model_name', 'convnext_base')
            model = FishClassifier(model_name=model_name, num_classes=NUM_CLASSES, pretrained=False)

            if 'ema_state_dict' in ckpt:
                model.load_state_dict(ckpt['ema_state_dict'])
                msg = f"Loaded EMA weights from {os.path.basename(self.checkpoint_path)}"
            elif 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'])
                msg = f"Loaded model weights from {os.path.basename(self.checkpoint_path)}"
            elif isinstance(ckpt, dict):
                model.load_state_dict(ckpt)
                msg = f"Loaded raw state dict from {os.path.basename(self.checkpoint_path)}"
            else:
                raise ValueError("Unrecognized checkpoint format.")

            model = model.to(self.device).eval()
            self.loaded.emit(model, model_name, msg)
        except Exception as e:
            self.error.emit(f"Failed to load checkpoint: {str(e)}")


class SingleInferenceWorker(QThread):
    """Performs inference on a single image asynchronously."""
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, image_path, model, device, image_size=288, use_tta=False):
        super().__init__()
        self.image_path = image_path
        self.model = model
        self.device = device
        self.image_size = image_size
        self.use_tta = use_tta

    def run(self):
        try:
            start_time = time.perf_counter()
            image = cv2.imread(self.image_path)
            if image is None:
                self.error.emit(f"Could not load image file: {self.image_path}")
                return
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if self.use_tta:
                transforms_list = get_tta_transforms(self.image_size)
            else:
                transforms_list = [get_val_transforms(self.image_size)]

            all_probs = np.zeros((NUM_CLASSES,), dtype=np.float32)

            self.model.eval()
            with torch.no_grad():
                for tfm in transforms_list:
                    augmented = tfm(image=image_rgb)
                    img_tensor = augmented['image'].unsqueeze(0).to(self.device)
                    logits = self.model(img_tensor)
                    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                    all_probs += probs

            all_probs /= len(transforms_list)
            latency_ms = (time.perf_counter() - start_time) * 1000.0

            top_class_id = int(np.argmax(all_probs))
            top_confidence = float(all_probs[top_class_id])

            # Top 5 indices
            top5_indices = np.argsort(all_probs)[::-1][:5]
            top5_list = []
            for idx in top5_indices:
                info = get_species_info(int(idx))
                top5_list.append({
                    "class_id": int(idx),
                    "class_name": info["class_name"],
                    "scientific_name": info["scientific_name"],
                    "common_name": info["common_name"],
                    "probability": float(all_probs[idx])
                })

            species_info = get_species_info(top_class_id)
            result = {
                "image_path": self.image_path,
                "top_class_id": top_class_id,
                "class_name": species_info["class_name"],
                "confidence": top_confidence,
                "scientific_name": species_info["scientific_name"],
                "common_name": species_info["common_name"],
                "family": species_info["family"],
                "description": species_info["description"],
                "latency_ms": latency_ms,
                "top5": top5_list,
                "use_tta": self.use_tta,
                "all_probs": all_probs.tolist()
            }
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(f"Inference error: {str(e)}")


class BatchInferenceWorker(QThread):
    """Performs inference on a folder of images asynchronously."""
    progress = Signal(int, int)  # (current, total)
    image_result = Signal(dict)
    batch_finished = Signal(list)
    error = Signal(str)

    def __init__(self, folder_path, model, device, image_size=288, use_tta=False):
        super().__init__()
        self.folder_path = folder_path
        self.model = model
        self.device = device
        self.image_size = image_size
        self.use_tta = use_tta

    def run(self):
        try:
            extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
            image_files = []
            for ext in extensions:
                image_files.extend(glob.glob(os.path.join(self.folder_path, ext)))
                image_files.extend(glob.glob(os.path.join(self.folder_path, '**', ext), recursive=True))

            image_files = sorted(list(set(image_files)))
            total = len(image_files)

            if total == 0:
                self.error.emit(f"No image files found in {self.folder_path}")
                return

            val_tfm = get_val_transforms(self.image_size)
            results = []

            self.model.eval()
            with torch.no_grad():
                for idx, img_path in enumerate(image_files):
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    augmented = val_tfm(image=img_rgb)
                    tensor = augmented['image'].unsqueeze(0).to(self.device)
                    logits = self.model(tensor)
                    probs = F.softmax(logits, dim=1).cpu().numpy()[0]

                    class_id = int(np.argmax(probs))
                    conf = float(probs[class_id])
                    info = get_species_info(class_id)

                    res_item = {
                        "filename": os.path.basename(img_path),
                        "image_path": img_path,
                        "class_id": class_id,
                        "confidence": conf,
                        "scientific_name": info["scientific_name"],
                        "common_name": info["common_name"]
                    }
                    results.append(res_item)
                    self.image_result.emit(res_item)
                    self.progress.emit(idx + 1, total)

            self.batch_finished.emit(results)
        except Exception as e:
            self.error.emit(f"Batch processing error: {str(e)}")
