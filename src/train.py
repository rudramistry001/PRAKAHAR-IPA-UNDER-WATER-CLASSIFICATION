"""
Fish4Knowledge Training Pipeline
=================================
Trains ConvNeXt-Base or EfficientNetV2-M on the Fish4Knowledge dataset
with trajectory-aware splitting, class-balanced loss, AMP, EMA, and
comprehensive logging.

Usage:
    python src/train.py --model convnext_base --epochs 50
    python src/train.py --model efficientnetv2_m --epochs 50
"""

import os
import sys
import argparse
import time
import json
import math
import random
import logging
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, balanced_accuracy_score
)

# Ensure src modules can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import (
    build_metadata,
    create_trajectory_split,
    Fish4KDataset,
    get_class_weights,
    get_balanced_sampler
)
from src.transforms import get_train_transforms, get_val_transforms
from src.losses import get_loss_function, MixUp, CutMix
from src.models import FishClassifier, ModelEMA

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────
#  Argument parsing
# ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Fish4Knowledge Classifier Training")
    p.add_argument('--model', type=str, default='convnext_base',
                   choices=['convnext_base', 'efficientnetv2_m', 'tf_efficientnetv2_m'],
                   help='Model architecture')
    p.add_argument('--image-size', type=int, default=288)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--data-root', type=str,
                   default=r'd:\prakhar ipa\my dataset\fish4konwledge')
    p.add_argument('--output-dir', type=str,
                   default=r'd:\prakhar ipa\my dataset\fish4konwledge\runs')
    p.add_argument('--loss', type=str, default='class_balanced',
                   choices=['ce', 'weighted_ce', 'focal', 'class_balanced', 'balanced_softmax'])
    p.add_argument('--grad-accum', type=int, default=2)
    p.add_argument('--warmup-epochs', type=int, default=3)
    p.add_argument('--patience', type=int, default=8)
    p.add_argument('--resume', type=str, default='')
    p.add_argument('--label-smoothing', type=float, default=0.1)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--max-samples', type=int, default=0,
                   help='Maximum samples per split; 0 uses the full split')
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────
#  Utilities
# ──────────────────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(log_dir):
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(os.path.join(log_dir, 'train.log'))
    ch = logging.StreamHandler()
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def cosine_lr_lambda(warmup_steps, total_steps, min_lr_ratio=1e-2):
    """Returns a lambda for LambdaLR: linear warmup then cosine decay."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return lr_lambda


def compute_metrics(targets, preds):
    return {
        'accuracy': accuracy_score(targets, preds),
        'balanced_accuracy': balanced_accuracy_score(targets, preds),
        'macro_f1': f1_score(targets, preds, average='macro', zero_division=0),
        'macro_precision': precision_score(targets, preds, average='macro', zero_division=0),
        'macro_recall': recall_score(targets, preds, average='macro', zero_division=0),
    }


def limit_split_samples(dataframe, max_samples, seed):
    """Select a deterministic, class-aware subset without changing groups."""
    if max_samples <= 0 or len(dataframe) <= max_samples:
        return dataframe.reset_index(drop=True)

    rng = np.random.default_rng(seed)
    selected_counts = {class_id: 1 for class_id in dataframe['class_id'].unique()}
    remaining = max_samples - len(selected_counts)
    capacities = {
        class_id: len(group) - 1
        for class_id, group in dataframe.groupby('class_id', sort=True)
    }
    total_capacity = sum(capacities.values())

    if remaining > 0 and total_capacity:
        for class_id, capacity in capacities.items():
            selected_counts[class_id] += min(
                capacity, int(remaining * capacity / total_capacity)
            )
        while sum(selected_counts.values()) < max_samples:
            candidates = [
                class_id for class_id, capacity in capacities.items()
                if selected_counts[class_id] < capacity + 1
            ]
            if not candidates:
                break
            class_id = candidates[int(rng.integers(len(candidates)))]
            selected_counts[class_id] += 1

    selected = []
    for class_id, group in dataframe.groupby('class_id', sort=True):
        indices = rng.permutation(len(group))[:selected_counts[class_id]]
        selected.append(group.iloc[indices])
    return pd.concat(selected).sample(frac=1, random_state=seed).reset_index(drop=True)


def save_training_plots(history, run_dir):
    """Save loss, accuracy, macro-F1, and LR curves."""
    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title('Training vs Validation Loss'); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(run_dir, 'train_val_loss.png'), dpi=150)
    plt.close()

    # Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['val_acc'], label='Val Accuracy')
    plt.plot(epochs, history['val_balanced_acc'], label='Val Balanced Acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.title('Validation Accuracy'); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(run_dir, 'train_val_accuracy.png'), dpi=150)
    plt.close()

    # Macro F1
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['val_macro_f1'], label='Val Macro F1', color='green')
    plt.xlabel('Epoch'); plt.ylabel('Macro F1')
    plt.title('Validation Macro F1'); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(run_dir, 'validation_macro_f1.png'), dpi=150)
    plt.close()

    # Learning rate
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(history['lr'])), history['lr'], color='orange')
    plt.xlabel('Step'); plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule'); plt.grid(True)
    plt.savefig(os.path.join(run_dir, 'learning_rate.png'), dpi=150)
    plt.close()


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # Remap friendly name -> timm name
    model_name = args.model
    if model_name == 'efficientnetv2_m':
        model_name = 'tf_efficientnetv2_m'

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Output directory ──────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f'run_{model_name}_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    logger = setup_logger(run_dir)
    writer = SummaryWriter(log_dir=os.path.join(run_dir, 'tb'))

    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # ── Data pipeline ────────────────────────────────────────────────
    metadata_dir = os.path.join(args.data_root, 'dataset', 'metadata')
    os.makedirs(metadata_dir, exist_ok=True)

    meta_csv = os.path.join(metadata_dir, 'metadata.csv')
    if os.path.exists(meta_csv):
        logger.info(f"Loading cached metadata from {meta_csv}")
        metadata = pd.read_csv(meta_csv)
    else:
        logger.info("Building metadata (scanning images — this may take a few minutes)...")
        metadata = build_metadata(args.data_root)
        metadata.to_csv(meta_csv, index=False)
        logger.info(f"Metadata saved to {meta_csv}  ({len(metadata)} images)")

    train_csv = os.path.join(metadata_dir, 'train.csv')
    val_csv   = os.path.join(metadata_dir, 'val.csv')
    test_csv  = os.path.join(metadata_dir, 'test.csv')

    if os.path.exists(train_csv) and os.path.exists(val_csv) and os.path.exists(test_csv):
        logger.info("Loading cached train/val/test splits...")
        train_df = pd.read_csv(train_csv)
        val_df   = pd.read_csv(val_csv)
        test_df  = pd.read_csv(test_csv)
    else:
        logger.info("Creating trajectory-aware split...")
        train_df, val_df, test_df = create_trajectory_split(metadata, seed=args.seed)
        train_df.to_csv(train_csv, index=False)
        val_df.to_csv(val_csv, index=False)
        test_df.to_csv(test_csv, index=False)
        logger.info(f"Split saved — Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    if args.max_samples > 0:
        train_df = limit_split_samples(train_df, args.max_samples, args.seed)
        val_df = limit_split_samples(val_df, args.max_samples, args.seed + 1)
        test_df = limit_split_samples(test_df, args.max_samples, args.seed + 2)
        logger.info(f"Applied max-samples={args.max_samples} independently to train/val/test")

    logger.info(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Class distribution for loss function
    num_classes = 23
    samples_per_cls = []
    for c in range(num_classes):
        count = len(train_df[train_df['class_id'] == c])
        samples_per_cls.append(max(count, 1))  # avoid zero
    logger.info(f"Samples per class (train): {samples_per_cls}")

    # Datasets
    train_transform = get_train_transforms(args.image_size)
    val_transform   = get_val_transforms(args.image_size)

    train_dataset = Fish4KDataset(train_df, args.data_root, transform=train_transform)
    val_dataset   = Fish4KDataset(val_df,   args.data_root, transform=val_transform)

    # Sampler
    train_sampler = get_balanced_sampler(train_dataset)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=(args.num_workers > 0), drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=(args.num_workers > 0)
    )

    logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── Model ────────────────────────────────────────────────────────
    logger.info(f"Creating {model_name} with pretrained weights...")
    model = FishClassifier(model_name=model_name, num_classes=num_classes,
                           pretrained=True, dropout=0.2)
    model = model.to(device)
    ema = ModelEMA(model, decay=0.9998)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Loss ─────────────────────────────────────────────────────────
    criterion = get_loss_function(
        args.loss,
        samples_per_cls=samples_per_cls,
        num_classes=num_classes,
        label_smoothing=args.label_smoothing
    ).to(device)
    logger.info(f"Loss function: {args.loss}")

    # ── MixUp / CutMix ──────────────────────────────────────────────
    mixup  = MixUp(alpha=0.2, p=0.15)
    cutmix = CutMix(alpha=1.0, p=0.15)

    # ── AMP scaler ───────────────────────────────────────────────────
    scaler = GradScaler('cuda')

    # ── Training state ───────────────────────────────────────────────
    start_epoch = 0
    best_macro_f1 = 0.0
    best_acc = 0.0
    patience_counter = 0

    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = updates_per_epoch * args.epochs
    warmup_steps = updates_per_epoch * args.warmup_epochs

    # History for plots
    history = {
        'train_loss': [], 'val_loss': [],
        'val_acc': [], 'val_balanced_acc': [], 'val_macro_f1': [],
        'lr': []
    }

    # ── Phase 1 setup: freeze backbone, train head only ──────────────
    model.freeze_backbone()
    head_params = list(model.head.parameters())
    optimizer = optim.AdamW(head_params, lr=1e-3, weight_decay=0.01)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, cosine_lr_lambda(warmup_steps, total_steps)
    )

    # ── Resume ───────────────────────────────────────────────────────
    if args.resume and os.path.isfile(args.resume):
        logger.info(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        ema.ema_model.load_state_dict(ckpt['ema_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_macro_f1 = ckpt.get('best_macro_f1', 0.0)
        best_acc = ckpt.get('best_acc', 0.0)
        logger.info(f"Resumed at epoch {start_epoch}")

    # ── Training loop ────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Starting training loop")
    logger.info("=" * 60)
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # ── Phase transition ─────────────────────────────────────────
        if epoch == args.warmup_epochs:
            logger.info(">>> Unfreezing backbone — switching to discriminative LRs")
            model.unfreeze_backbone()
            param_groups = model.get_param_groups(
                lr_backbone=args.lr * 0.1, lr_head=args.lr
            )
            optimizer = optim.AdamW(param_groups, weight_decay=0.05)
            remaining = updates_per_epoch * (args.epochs - args.warmup_epochs)
            scheduler = optim.lr_scheduler.LambdaLR(
                optimizer, cosine_lr_lambda(0, remaining)
            )

        phase = "Warmup (head only)" if epoch < args.warmup_epochs else "Full fine-tuning"
        logger.info(f"Epoch {epoch+1}/{args.epochs} — [{phase}]")

        # ── Train ────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        all_preds, all_targets = [], []
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Train", leave=False)
        for step, (images, labels) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Randomly apply MixUp or CutMix (mutually exclusive)
            use_mix = False
            r = random.random()
            if r < 0.15:
                images, soft_labels = mixup(images, labels, num_classes=num_classes)
                use_mix = True
            elif r < 0.30:
                images, soft_labels = cutmix(images, labels, num_classes=num_classes)
                use_mix = True

            try:
                with autocast('cuda'):
                    logits = model(images)
                    if use_mix:
                        # soft labels: cross entropy on distributions
                        log_probs = torch.nn.functional.log_softmax(logits, dim=1)
                        loss = -(soft_labels * log_probs).sum(dim=1).mean()
                    else:
                        loss = criterion(logits, labels)
                    loss = loss / args.grad_accum

                scaler.scale(loss).backward()

                if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    ema.update(model)

                    current_lr = optimizer.param_groups[0]['lr']
                    history['lr'].append(current_lr)
                    scheduler.step()

                running_loss += loss.item() * args.grad_accum

                if not use_mix:
                    preds = logits.argmax(dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(labels.cpu().numpy())

                pbar.set_postfix({
                    'loss': f"{loss.item() * args.grad_accum:.4f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
                })

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(
                        "CUDA OOM! Reduce --batch-size or --image-size.\n"
                        f"Current: batch_size={args.batch_size}, image_size={args.image_size}"
                    )
                    torch.cuda.empty_cache()
                    sys.exit(1)
                raise

        avg_train_loss = running_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # ── Validate (EMA model) ─────────────────────────────────────
        ema.ema_model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating", leave=False):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with autocast('cuda'):
                    logits = ema.ema_model(images)
                    loss = criterion(logits, labels)
                val_loss += loss.item()
                val_preds.extend(logits.argmax(dim=1).cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        metrics = compute_metrics(val_targets, val_preds)

        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(metrics['accuracy'])
        history['val_balanced_acc'].append(metrics['balanced_accuracy'])
        history['val_macro_f1'].append(metrics['macro_f1'])

        # ── Logging ──────────────────────────────────────────────────
        elapsed = time.time() - epoch_start
        logger.info(
            f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}\n"
            f"  Acc: {metrics['accuracy']:.4f} | Bal Acc: {metrics['balanced_accuracy']:.4f}\n"
            f"  Macro F1: {metrics['macro_f1']:.4f} | Prec: {metrics['macro_precision']:.4f} | Rec: {metrics['macro_recall']:.4f}\n"
            f"  Epoch time: {elapsed:.0f}s | GPU Mem: {torch.cuda.memory_allocated()/1024**2:.0f} MB"
        )

        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Val', avg_val_loss, epoch)
        writer.add_scalar('Metrics/Accuracy', metrics['accuracy'], epoch)
        writer.add_scalar('Metrics/Balanced_Accuracy', metrics['balanced_accuracy'], epoch)
        writer.add_scalar('Metrics/Macro_F1', metrics['macro_f1'], epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        # ── Checkpoints ──────────────────────────────────────────────
        ckpt = {
            'epoch': epoch,
            'model_name': model_name,
            'model_state_dict': model.state_dict(),
            'ema_state_dict': ema.ema_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_macro_f1': best_macro_f1,
            'best_acc': best_acc,
            'config': vars(args),
            'metrics': metrics,
        }

        torch.save(ckpt, os.path.join(run_dir, 'last.pt'))

        if metrics['macro_f1'] > best_macro_f1:
            best_macro_f1 = metrics['macro_f1']
            ckpt['best_macro_f1'] = best_macro_f1
            torch.save(ckpt, os.path.join(run_dir, 'best_macro_f1.pt'))
            patience_counter = 0
            logger.info(f"  ★ New best Macro F1: {best_macro_f1:.4f}")
        else:
            patience_counter += 1

        if metrics['accuracy'] > best_acc:
            best_acc = metrics['accuracy']
            ckpt['best_acc'] = best_acc
            torch.save(ckpt, os.path.join(run_dir, 'best_accuracy.pt'))
            logger.info(f"  ★ New best Accuracy: {best_acc:.4f}")

        if patience_counter >= args.patience:
            logger.info(f"Early stopping triggered (no improvement for {args.patience} epochs)")
            break

    # ── End of training ──────────────────────────────────────────────
    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"Training complete in {total_time/3600:.2f} hours")
    logger.info(f"Best Macro F1:  {best_macro_f1:.4f}")
    logger.info(f"Best Accuracy:  {best_acc:.4f}")
    logger.info(f"Checkpoints in: {run_dir}")
    logger.info("=" * 60)

    # Save training curves
    save_training_plots(history, run_dir)
    logger.info("Training plots saved.")

    # Save metrics history
    pd.DataFrame({
        'epoch': list(range(1, len(history['train_loss']) + 1)),
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'val_acc': history['val_acc'],
        'val_balanced_acc': history['val_balanced_acc'],
        'val_macro_f1': history['val_macro_f1'],
    }).to_csv(os.path.join(run_dir, 'metrics_history.csv'), index=False)

    writer.close()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
