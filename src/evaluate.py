"""
Fish4Knowledge Evaluation Pipeline
====================================
Evaluates a trained checkpoint on val or test split with comprehensive
metrics, plots, and optional TTA.

Usage:
    python src/evaluate.py --checkpoint runs/run_xxx/best_macro_f1.pt --split val
    python src/evaluate.py --checkpoint runs/run_xxx/best_macro_f1.pt --split test --tta --final-test
"""

import argparse
import os
import sys
import json
import logging
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import Fish4KDataset
from src.transforms import get_val_transforms, get_tta_transforms
from src.models import FishClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('evaluate')

NUM_CLASSES = 23
CLASS_NAMES = [f'fish_{str(i+1).zfill(2)}' for i in range(NUM_CLASSES)]


# ──────────────────────────────────────────────────────────────
#  Metrics
# ──────────────────────────────────────────────────────────────
def compute_ece(probs, targets, n_bins=15):
    bin_bounds = np.linspace(0, 1, n_bins + 1)
    confs = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    accs  = (preds == targets).astype(float)
    ece = 0.0
    for lo, hi in zip(bin_bounds[:-1], bin_bounds[1:]):
        mask = (confs > lo) & (confs <= hi)
        if mask.sum() > 0:
            ece += mask.mean() * abs(accs[mask].mean() - confs[mask].mean())
    return float(ece)


# ──────────────────────────────────────────────────────────────
#  Inference
# ──────────────────────────────────────────────────────────────
def run_inference(model, df, data_root, device, image_size, batch_size,
                  use_tta=False, num_workers=4):
    """Run inference and return (probs, targets, image_paths)."""
    if use_tta:
        tta_tfms = get_tta_transforms(image_size)
    else:
        tta_tfms = [get_val_transforms(image_size)]

    all_probs = None
    targets = None
    paths = None

    for t_idx, tfm in enumerate(tta_tfms):
        ds = Fish4KDataset(df, data_root, transform=tfm)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

        probs_list = []
        targets_list = []
        paths_list = []

        model.eval()
        with torch.no_grad():
            for images, labels in dl:
                images = images.to(device, non_blocking=True)
                logits = model(images)
                probs_list.append(F.softmax(logits, dim=1).cpu().numpy())
                targets_list.extend(labels.numpy())

        probs_arr = np.vstack(probs_list)
        if all_probs is None:
            all_probs = probs_arr
            targets = np.array(targets_list)
            # Collect image paths
            paths = [os.path.join(data_root, row['image_path'])
                     for _, row in df.iterrows()]
        else:
            all_probs += probs_arr

    all_probs /= len(tta_tfms if use_tta else [None])
    return all_probs, targets, paths


# ──────────────────────────────────────────────────────────────
#  Plotting helpers
# ──────────────────────────────────────────────────────────────
def plot_confusion(cm, class_names, output_dir, normalized=True):
    plt.figure(figsize=(20, 16))
    data = cm.astype('float') / cm.sum(axis=1, keepdims=True) if normalized else cm
    fmt = '.2f' if normalized else 'd'
    tag = 'Normalized' if normalized else 'Raw'
    sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{tag} Confusion Matrix'); plt.ylabel('True'); plt.xlabel('Predicted')
    plt.xticks(rotation=90); plt.tight_layout()
    fname = 'confusion_matrix_normalized.png' if normalized else 'confusion_matrix_raw.png'
    plt.savefig(os.path.join(output_dir, fname), dpi=150); plt.close()


def plot_bar(values, names, title, ylabel, fname, output_dir):
    plt.figure(figsize=(15, 8))
    colors = sns.color_palette("viridis", len(names))
    plt.bar(range(len(names)), values, color=colors)
    plt.xticks(range(len(names)), names, rotation=90)
    plt.title(title); plt.ylabel(ylabel); plt.ylim(0, 1.05); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, fname), dpi=150); plt.close()


def plot_roc(targets, probs, class_names, output_dir):
    plt.figure(figsize=(14, 11))
    for i, name in enumerate(class_names):
        y_true = (targets == i).astype(int)
        if y_true.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true, probs[:, i])
        auc = roc_auc_score(y_true, probs[:, i])
        plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.2f})')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.title('ROC Curves (One-vs-Rest)')
    plt.legend(loc='lower right', fontsize='small', ncol=2); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=150); plt.close()


def plot_pr(targets, probs, class_names, output_dir):
    plt.figure(figsize=(14, 11))
    for i, name in enumerate(class_names):
        y_true = (targets == i).astype(int)
        if y_true.sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(y_true, probs[:, i])
        ap = average_precision_score(y_true, probs[:, i])
        plt.plot(recall, precision, label=f'{name} (AP={ap:.2f})')
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title('Precision-Recall Curves (One-vs-Rest)')
    plt.legend(loc='lower left', fontsize='small', ncol=2); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pr_curves.png'), dpi=150); plt.close()


def plot_reliability(targets, probs, output_dir, n_bins=15):
    confs = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    accs  = (preds == targets).astype(float)
    bin_bounds = np.linspace(0, 1, n_bins + 1)
    bin_accs, bin_confs = [], []
    for lo, hi in zip(bin_bounds[:-1], bin_bounds[1:]):
        mask = (confs > lo) & (confs <= hi)
        if mask.any():
            bin_accs.append(accs[mask].mean())
            bin_confs.append(confs[mask].mean())
        else:
            bin_accs.append(0)
            bin_confs.append((lo + hi) / 2)
    plt.figure(figsize=(8, 8))
    plt.plot([0,1],[0,1],'k--', label='Perfect')
    plt.plot(bin_confs, bin_accs, 's-', label='Model')
    plt.xlabel('Confidence'); plt.ylabel('Accuracy')
    ece = compute_ece(probs, targets)
    plt.title(f'Reliability Diagram (ECE={ece:.4f})'); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reliability_diagram.png'), dpi=150); plt.close()


def plot_error_gallery(targets, probs, paths, class_names, output_dir, top_n=20):
    preds = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)
    wrong = np.where(preds != targets)[0]
    if len(wrong) == 0:
        return
    sorted_wrong = wrong[np.argsort(confs[wrong])[::-1]][:top_n]
    cols = 5
    rows = int(np.ceil(len(sorted_wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
    axes = axes.flatten() if rows > 1 else [axes] if rows == 1 and cols == 1 else axes.flatten()
    for i, idx in enumerate(sorted_wrong):
        img = cv2.imread(paths[idx])
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.zeros((100, 100, 3), dtype=np.uint8)
        axes[i].imshow(img)
        axes[i].set_title(
            f"True: {class_names[targets[idx]]}\n"
            f"Pred: {class_names[preds[idx]]} ({confs[idx]:.2f})", fontsize=8
        )
        axes[i].axis('off')
    for j in range(len(sorted_wrong), len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_gallery.png'), dpi=150); plt.close()


def plot_confused_pairs(cm, class_names, output_dir, top_n=10):
    cm_copy = cm.copy().astype(float)
    np.fill_diagonal(cm_copy, 0)
    pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if cm_copy[i, j] > 0:
                pairs.append((class_names[i], class_names[j], int(cm_copy[i, j])))
    pairs.sort(key=lambda x: x[2], reverse=True)
    pairs = pairs[:top_n]
    if not pairs:
        return
    labels = [f"{a} → {b}" for a, b, _ in pairs]
    counts = [c for _, _, c in pairs]
    plt.figure(figsize=(10, 6))
    sns.barplot(y=labels, x=counts, color='salmon')
    plt.title('Most Confused Class Pairs'); plt.xlabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'most_confused_pairs.png'), dpi=150); plt.close()


# ──────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Fish4Knowledge Evaluation")
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--data-root', type=str,
                   default=r'd:\prakhar ipa\my dataset\fish4konwledge')
    p.add_argument('--output-dir', type=str, default=None)
    p.add_argument('--model', type=str, default='convnext_base')
    p.add_argument('--image-size', type=int, default=288)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--split', type=str, default='val', choices=['val', 'test'])
    p.add_argument('--tta', action='store_true')
    p.add_argument('--final-test', action='store_true')
    args = p.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.checkpoint), 'eval_results')
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # ── Load split CSV ───────────────────────────────────────────
    split_name = 'test' if args.final_test else args.split
    csv_path = os.path.join(args.data_root, 'dataset', 'metadata', f'{split_name}.csv')
    if not os.path.exists(csv_path):
        logger.error(f"Split CSV not found: {csv_path}")
        logger.error("Run train.py first to generate metadata splits.")
        return
    df = pd.read_csv(csv_path)
    logger.info(f"Evaluating on {split_name} split: {len(df)} images")

    # ── Load model ───────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Determine model name from checkpoint or args
    model_name = ckpt.get('model_name', args.model)
    if model_name == 'efficientnetv2_m':
        model_name = 'tf_efficientnetv2_m'

    logger.info(f"Loading {model_name} from {args.checkpoint}")
    model = FishClassifier(model_name=model_name, num_classes=NUM_CLASSES,
                           pretrained=False, dropout=0.2)

    # Prefer EMA weights
    if 'ema_state_dict' in ckpt:
        model.load_state_dict(ckpt['ema_state_dict'])
        logger.info("Loaded EMA weights")
    elif 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        logger.info("Loaded model weights")
    else:
        model.load_state_dict(ckpt)

    model = model.to(device)

    # ── Inference ────────────────────────────────────────────────
    probs, targets, paths = run_inference(
        model, df, args.data_root, device,
        args.image_size, args.batch_size, use_tta=args.tta
    )
    preds = np.argmax(probs, axis=1)
    logger.info("Inference complete.")

    # ── Metrics ──────────────────────────────────────────────────
    acc = accuracy_score(targets, preds)
    bal_acc = balanced_accuracy_score(targets, preds)
    prec_per, rec_per, f1_per, _ = precision_recall_fscore_support(
        targets, preds, average=None, zero_division=0)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        targets, preds, average='macro', zero_division=0)
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        targets, preds, average='weighted', zero_division=0)

    # Top-k
    top3 = np.mean([1 if t in np.argsort(probs[i])[-3:] else 0
                     for i, t in enumerate(targets)])
    top5 = np.mean([1 if t in np.argsort(probs[i])[-5:] else 0
                     for i, t in enumerate(targets)])

    # ROC-AUC / PR-AUC
    try:
        roc_auc = roc_auc_score(targets, probs, multi_class='ovr', average='macro')
    except ValueError:
        roc_auc = float('nan')
    try:
        pr_auc = average_precision_score(
            np.eye(NUM_CLASSES)[targets], probs, average='macro')
    except Exception:
        pr_auc = float('nan')

    ece = compute_ece(probs, targets)

    metrics = {
        'accuracy': float(acc),
        'balanced_accuracy': float(bal_acc),
        'macro_precision': float(macro_p),
        'macro_recall': float(macro_r),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'top1_accuracy': float(acc),
        'top3_accuracy': float(top3),
        'top5_accuracy': float(top5),
        'roc_auc_macro': float(roc_auc),
        'pr_auc_macro': float(pr_auc),
        'ece': float(ece),
    }

    logger.info("=" * 50)
    for k, v in metrics.items():
        logger.info(f"  {k:25s}: {v:.4f}")
    logger.info("=" * 50)

    # Save
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    # Predictions CSV
    pred_df = pd.DataFrame({
        'image_path': [os.path.relpath(p, args.data_root) for p in paths],
        'true_label': targets,
        'predicted_label': preds,
        'confidence': np.max(probs, axis=1),
        'true_class': [CLASS_NAMES[t] for t in targets],
        'pred_class': [CLASS_NAMES[p] for p in preds],
    })
    pred_df.to_csv(os.path.join(args.output_dir, 'predictions.csv'), index=False)

    # ── Plots ────────────────────────────────────────────────────
    logger.info("Generating plots...")
    cm = confusion_matrix(targets, preds, labels=list(range(NUM_CLASSES)))
    plot_confusion(cm, CLASS_NAMES, args.output_dir, normalized=True)
    plot_confusion(cm, CLASS_NAMES, args.output_dir, normalized=False)
    plot_bar(f1_per, CLASS_NAMES, 'Per-class F1', 'F1', 'per_class_f1.png', args.output_dir)
    plot_bar(rec_per, CLASS_NAMES, 'Per-class Recall', 'Recall', 'per_class_recall.png', args.output_dir)
    plot_bar(prec_per, CLASS_NAMES, 'Per-class Precision', 'Precision', 'per_class_precision.png', args.output_dir)
    plot_roc(targets, probs, CLASS_NAMES, args.output_dir)
    plot_pr(targets, probs, CLASS_NAMES, args.output_dir)
    plot_reliability(targets, probs, args.output_dir)
    plot_error_gallery(targets, probs, paths, CLASS_NAMES, args.output_dir)
    plot_confused_pairs(cm, CLASS_NAMES, args.output_dir)

    # Top-k bar
    plt.figure(figsize=(8, 6))
    vals = [acc, top3, top5]
    sns.barplot(x=['Top-1', 'Top-3', 'Top-5'], y=vals, palette='Greens_d')
    for i, v in enumerate(vals):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
    plt.title('Top-k Accuracy'); plt.ylim(0, 1.05)
    plt.savefig(os.path.join(args.output_dir, 'top_k_accuracy.png'), dpi=150); plt.close()

    logger.info(f"All results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
