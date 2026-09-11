import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)
            inputs = inputs.transpose(1, 2)
            inputs = inputs.contiguous().view(-1, inputs.size(2))
        
        # Cross entropy loss with optional label smoothing
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            # Handle alpha weight per class
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            
            if targets.dim() == 1:
                alpha_t = self.alpha[targets]
            else:
                # Soft targets support
                alpha_t = (targets * self.alpha).sum(dim=-1)
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class ClassBalancedLoss(nn.Module):
    def __init__(self, samples_per_cls, num_classes=23, loss_type='focal', beta=0.9999, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.loss_type = loss_type
        
        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * num_classes
        self.weights = torch.tensor(weights, dtype=torch.float32)
        
        if loss_type == 'focal':
            self.loss_fn = FocalLoss(alpha=self.weights, gamma=gamma, label_smoothing=label_smoothing, reduction='mean')
        else:
            self.loss_fn = nn.CrossEntropyLoss(weight=self.weights, label_smoothing=label_smoothing)
            
    def forward(self, inputs, targets):
        if self.weights.device != inputs.device:
            self.weights = self.weights.to(inputs.device)
            if self.loss_type == 'focal':
                self.loss_fn.alpha = self.weights
            else:
                self.loss_fn.weight = self.weights
        return self.loss_fn(inputs, targets)

class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, samples_per_cls, label_smoothing=0.1):
        super().__init__()
        samples_per_cls = np.array(samples_per_cls)
        freq = samples_per_cls / np.sum(samples_per_cls)
        self.log_freq = torch.tensor(np.log(freq), dtype=torch.float32)
        self.label_smoothing = label_smoothing
        
    def forward(self, inputs, targets):
        if self.log_freq.device != inputs.device:
            self.log_freq = self.log_freq.to(inputs.device)
        adjusted_inputs = inputs + self.log_freq
        return F.cross_entropy(adjusted_inputs, targets, label_smoothing=self.label_smoothing)

def get_loss_function(loss_type, samples_per_cls=None, num_classes=23, label_smoothing=0.1):
    if loss_type == 'ce':
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    elif loss_type == 'weighted_ce':
        if samples_per_cls is None:
            raise ValueError("samples_per_cls must be provided for weighted_ce")
        weights = 1.0 / np.array(samples_per_cls)
        weights = weights / np.sum(weights) * num_classes
        return nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32), label_smoothing=label_smoothing)
    elif loss_type == 'focal':
        return FocalLoss(label_smoothing=label_smoothing)
    elif loss_type == 'class_balanced':
        if samples_per_cls is None:
            raise ValueError("samples_per_cls must be provided for class_balanced")
        return ClassBalancedLoss(samples_per_cls, num_classes=num_classes, loss_type='focal', label_smoothing=label_smoothing)
    elif loss_type == 'balanced_softmax':
        if samples_per_cls is None:
            raise ValueError("samples_per_cls must be provided for balanced_softmax")
        return BalancedSoftmaxLoss(samples_per_cls, label_smoothing=label_smoothing)
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")


class MixUp:
    def __init__(self, alpha=0.2, p=0.15):
        self.alpha = alpha
        self.p = p

    def __call__(self, images, targets, num_classes=23):
        if np.random.rand() > self.p:
            if targets.dim() == 1:
                targets = F.one_hot(targets, num_classes=num_classes).float()
            return images, targets

        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = images.size(0)
        index = torch.randperm(batch_size).to(images.device)

        mixed_images = lam * images + (1 - lam) * images[index]
        
        if targets.dim() == 1:
            targets = F.one_hot(targets, num_classes=num_classes).float()
            
        mixed_targets = lam * targets + (1 - lam) * targets[index]
        return mixed_images, mixed_targets

class CutMix:
    def __init__(self, alpha=1.0, p=0.15):
        self.alpha = alpha
        self.p = p

    def __call__(self, images, targets, num_classes=23):
        if np.random.rand() > self.p:
            if targets.dim() == 1:
                targets = F.one_hot(targets, num_classes=num_classes).float()
            return images, targets

        lam = np.random.beta(self.alpha, self.alpha)
        batch_size, _, h, w = images.size()
        index = torch.randperm(batch_size).to(images.device)

        cut_rat = np.sqrt(1. - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)

        cx = np.random.randint(w)
        cy = np.random.randint(h)

        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        mixed_images = images.clone()
        mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]

        lam_adjusted = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        
        if targets.dim() == 1:
            targets = F.one_hot(targets, num_classes=num_classes).float()

        mixed_targets = lam_adjusted * targets + (1 - lam_adjusted) * targets[index]
        return mixed_images, mixed_targets
