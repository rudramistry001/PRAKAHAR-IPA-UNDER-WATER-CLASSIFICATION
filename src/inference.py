"""
Fish4Knowledge Ensemble & Inference
=====================================
Loads two trained models (ConvNeXt-Base + EfficientNetV2-M), optionally
applies TTA, and performs probability-weighted ensembling.

Usage:
    python src/inference.py \
        --checkpoint-a runs/run_convnext.../best_macro_f1.pt \
        --checkpoint-b runs/run_effnet.../best_macro_f1.pt \
        --ensemble --tta --split test
"""

import argparse
import os
import sys
import json
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import Fish4KDataset
from src.transforms import get_val_transforms, get_tta_transforms
from src.models import FishClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('inference')

NUM_CLASSES = 23


# ──────────────────────────────────────────────────────────────
#  Inference helpers
# ──────────────────────────────────────────────────────────────
def run_model_inference(model, df, data_root, device, image_size, batch_size,
                        use_tta=False, num_workers=4):
    """Return softmax probability matrix (N, C) and targets array."""
    if use_tta:
        transforms_list = get_tta_transforms(image_size)
    else:
        transforms_list = [get_val_transforms(image_size)]

    all_probs = None
    targets = None

    for tfm in transforms_list:
        ds = Fish4KDataset(df, data_root, transform=tfm)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
        probs_list = []
        tgt_list = []

        model.eval()
        with torch.no_grad():
            for images, labels in dl:
                images = images.to(device, non_blocking=True)
                logits = model(images)
                probs_list.append(F.softmax(logits, dim=1).cpu().numpy())
                tgt_list.extend(labels.numpy())

        probs_arr = np.vstack(probs_list)
        if all_probs is None:
            all_probs = probs_arr
            targets = np.array(tgt_list)
        else:
            all_probs += probs_arr

    all_probs /= len(transforms_list)
    return all_probs, targets


def eval_probs(probs, targets):
    preds = np.argmax(probs, axis=1)
    acc = accuracy_score(targets, preds)
    bal = balanced_accuracy_score(targets, preds)
    _, _, f1, _ = precision_recall_fscore_support(
        targets, preds, average='macro', zero_division=0)
    return acc, bal, f1


def load_fish_model(checkpoint_path, device):
    """Load a FishClassifier from a training checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_name = ckpt.get('model_name', 'convnext_base')
    model = FishClassifier(model_name=model_name, num_classes=NUM_CLASSES,
                           pretrained=False, dropout=0.2)
    # Prefer EMA weights
    if 'ema_state_dict' in ckpt:
        model.load_state_dict(ckpt['ema_state_dict'])
        logger.info(f"Loaded EMA weights from {checkpoint_path}")
    elif 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        logger.info(f"Loaded model weights from {checkpoint_path}")
    else:
        model.load_state_dict(ckpt)
    return model.to(device)


# ──────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Fish4Knowledge Ensemble Inference")
    p.add_argument('--checkpoint-a', type=str, required=True,
                   help='ConvNeXt checkpoint')
    p.add_argument('--checkpoint-b', type=str, required=True,
                   help='EfficientNetV2 checkpoint')
    p.add_argument('--data-root', type=str,
                   default=r'd:\prakhar ipa\my dataset\fish4konwledge')
    p.add_argument('--output-dir', type=str, default=None)
    p.add_argument('--image-size', type=int, default=288)
    p.add_argument('--alpha', type=float, default=0.6,
                   help='Weight for model A in ensemble')
    p.add_argument('--tta', action='store_true')
    p.add_argument('--ensemble', action='store_true')
    p.add_argument('--split', type=str, default='val', choices=['val', 'test'])
    p.add_argument('--batch-size', type=int, default=32)
    args = p.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.path.dirname(args.checkpoint_a), 'ensemble_results')
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # ── Load split CSV ───────────────────────────────────────────
    csv_path = os.path.join(args.data_root, 'dataset', 'metadata', f'{args.split}.csv')
    if not os.path.exists(csv_path):
        logger.error(f"Split CSV not found: {csv_path}")
        return
    df = pd.read_csv(csv_path)
    logger.info(f"Split: {args.split} — {len(df)} images")

    # ── Model A ──────────────────────────────────────────────────
    model_a = load_fish_model(args.checkpoint_a, device)
    logger.info("Running inference for Model A...")
    probs_a, targets = run_model_inference(
        model_a, df, args.data_root, device,
        args.image_size, args.batch_size, args.tta)
    acc_a, bal_a, f1_a = eval_probs(probs_a, targets)
    logger.info(f"Model A — Acc: {acc_a:.4f}, Bal: {bal_a:.4f}, F1: {f1_a:.4f}")
    del model_a; torch.cuda.empty_cache()

    # ── Model B ──────────────────────────────────────────────────
    model_b = load_fish_model(args.checkpoint_b, device)
    logger.info("Running inference for Model B...")
    probs_b, _ = run_model_inference(
        model_b, df, args.data_root, device,
        args.image_size, args.batch_size, args.tta)
    acc_b, bal_b, f1_b = eval_probs(probs_b, targets)
    logger.info(f"Model B — Acc: {acc_b:.4f}, Bal: {bal_b:.4f}, F1: {f1_b:.4f}")
    del model_b; torch.cuda.empty_cache()

    # ── Alpha tuning on validation ───────────────────────────────
    if args.split == 'val':
        alphas = [0.3, 0.4, 0.5, 0.6, 0.7]
        best_alpha, best_f1 = args.alpha, 0.0
        logger.info("Tuning ensemble alpha on validation...")
        for a in alphas:
            probs_ens = a * probs_a + (1 - a) * probs_b
            acc_e, bal_e, f1_e = eval_probs(probs_ens, targets)
            logger.info(f"  α={a:.1f}: Acc={acc_e:.4f}, Bal={bal_e:.4f}, F1={f1_e:.4f}")
            if f1_e > best_f1:
                best_f1, best_alpha = f1_e, a
        logger.info(f"Best α: {best_alpha} (Macro F1: {best_f1:.4f})")
        final_alpha = best_alpha
    else:
        final_alpha = args.alpha

    # ── Final ensemble ───────────────────────────────────────────
    probs_ens = final_alpha * probs_a + (1 - final_alpha) * probs_b
    acc_e, bal_e, f1_e = eval_probs(probs_ens, targets)

    # ── Comparison table ─────────────────────────────────────────
    rows = [
        {'Model': 'ConvNeXt-Base (A)', 'Accuracy': acc_a,
         'Balanced Acc': bal_a, 'Macro F1': f1_a},
        {'Model': 'EfficientNetV2-M (B)', 'Accuracy': acc_b,
         'Balanced Acc': bal_b, 'Macro F1': f1_b},
        {'Model': f'Ensemble (α={final_alpha:.1f})', 'Accuracy': acc_e,
         'Balanced Acc': bal_e, 'Macro F1': f1_e},
    ]
    results_df = pd.DataFrame(rows)
    print("\n" + "=" * 60)
    print("  COMPARISON TABLE")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print("=" * 60 + "\n")
    results_df.to_csv(os.path.join(args.output_dir, 'ensemble_comparison.csv'), index=False)

    # ── Save predictions ─────────────────────────────────────────
    preds = np.argmax(probs_ens, axis=1)
    class_names = [f'fish_{str(i+1).zfill(2)}' for i in range(NUM_CLASSES)]
    pred_df = pd.DataFrame({
        'image_path': df['image_path'].values,
        'true_label': targets,
        'predicted_label': preds,
        'confidence': np.max(probs_ens, axis=1),
        'true_class': [class_names[t] for t in targets],
        'pred_class': [class_names[p] for p in preds],
    })
    pred_df.to_csv(os.path.join(args.output_dir, 'ensemble_predictions.csv'), index=False)

    metrics = {
        'best_alpha': float(final_alpha),
        'model_a_accuracy': float(acc_a), 'model_a_macro_f1': float(f1_a),
        'model_b_accuracy': float(acc_b), 'model_b_macro_f1': float(f1_b),
        'ensemble_accuracy': float(acc_e),
        'ensemble_balanced_accuracy': float(bal_e),
        'ensemble_macro_f1': float(f1_e),
    }
    with open(os.path.join(args.output_dir, 'ensemble_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    logger.info(f"Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
