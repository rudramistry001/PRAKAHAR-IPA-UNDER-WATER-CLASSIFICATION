"""
Fish4Knowledge Augmentation Transforms
========================================
Training, validation, and TTA augmentation pipelines using
Albumentations. Includes underwater-specific augmentations.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transforms(image_size=288):
    """Training augmentations with underwater-specific transforms."""
    return A.Compose([
        A.RandomResizedCrop(size=(image_size, image_size), scale=(0.7, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.25, hue=0.05, p=0.7),
        A.GaussianBlur(blur_limit=(3, 5), p=0.1),
        A.Perspective(scale=(0.02, 0.05), p=0.1),

        # Underwater-specific augmentations
        A.RandomBrightnessContrast(
            brightness_limit=(-0.3, 0.1), contrast_limit=(-0.2, 0.2), p=0.3),
        A.HueSaturationValue(
            hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.3),
        A.GaussNoise(p=0.15),

        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])


def get_val_transforms(image_size=288):
    """Deterministic validation transforms."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])


def get_tta_transforms(image_size=288):
    """Returns a list of 3 TTA transform pipelines."""
    return [
        # 1. Standard validation
        A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ]),
        # 2. Horizontal flip
        A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ]),
        # 3. Slightly larger resize + center crop
        A.Compose([
            A.Resize(int(image_size * 1.15), int(image_size * 1.15)),
            A.CenterCrop(image_size, image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ])
    ]
