# Fish4Knowledge High-Accuracy Fish Species Classification

A reproducible deep-learning pipeline for training a high-accuracy **23-class underwater fish species classifier** on the [Fish4Knowledge recognition ground-truth dataset](https://github.com/Callmewuxin/fish4konwledge).

The project is designed for a **6 GB NVIDIA RTX 4050 Laptop GPU** and prioritizes:

- leakage-free evaluation
- trajectory-aware train/validation/test splitting
- severe class-imbalance handling
- underwater-specific augmentation
- pretrained transfer learning
- mixed-precision training
- EMA checkpoints
- test-time augmentation
- model ensembling
- detailed plots and error analysis

> **Core principle:** do not optimize only for validation accuracy. The final result must be measured on a trajectory-independent test set, with Macro F1 and balanced accuracy reported alongside overall accuracy.

---

## 1. Dataset

### Fish4Knowledge Recognition Ground Truth

The official Fish4Knowledge recognition ground-truth dataset contains:

- **27,370 verified fish images**
- **23 species/classes**
- fish images and corresponding masks
- tracking IDs / fish IDs
- severe class imbalance

Official documentation states that images with the same **tracking ID belong to the same fish trajectory**. This is critical because consecutive frames from the same trajectory are highly correlated.

The GitHub repository used for this project mirrors the 23-class `fish_image` dataset.

### Official resources

- GitHub repository:  
  https://github.com/Callmewuxin/fish4konwledge
- Official Fish4Knowledge recognition ground truth:  
  https://homepages.inf.ed.ac.uk/rbf/Fish4Knowledge/GROUNDTRUTH/RECOG/

---

# 2. Why this project uses trajectory-aware splitting

A random image split is dangerous for this dataset.

Suppose these images exist:

```text
fish_000000009598_05281.png
fish_000000009598_05283.png
fish_000000009598_05285.png
fish_000000009598_05287.png
```

The tracking ID is shared, so these images belong to the same trajectory.

If one frame goes into training and another frame goes into validation/test, the model can effectively see almost the same fish during training and evaluation.

That creates **data leakage** and can produce an unrealistically high score.

Therefore:

```text
TRAIN trajectories
        ∩
VALIDATION trajectories
        = ∅

TRAIN trajectories
        ∩
TEST trajectories
        = ∅

VALIDATION trajectories
        ∩
TEST trajectories
        = ∅
```

The official Fish4Knowledge documentation and project material also describe separating training and testing so fish images from the same trajectory sequence are not shared between them.

---

# 3. Project objective

The primary goal is:

> Train a robust high-accuracy 23-species classifier that generalizes to previously unseen fish trajectories.

The target pipeline is:

```text
Fish4Knowledge
      ↓
Dataset inspection
      ↓
Metadata generation
      ↓
Trajectory extraction
      ↓
Group/trajectory-aware splitting
      ↓
Image + mask experiments
      ↓
Underwater augmentation
      ↓
Pretrained ConvNeXt-Base
      ↓
Class-balanced optimization
      ↓
AMP + EMA
      ↓
Error analysis
      ↓
EfficientNetV2-M second model
      ↓
Probability ensemble
      ↓
Test-time augmentation
      ↓
Final test evaluation
```

---

# 4. Recommended project structure

```text
fish4knowledge_project/
│
├── dataset/
│   ├── raw/
│   │   └── fish_image/
│   │       ├── fish_01/
│   │       ├── fish_02/
│   │       ├── ...
│   │       └── fish_23/
│   │
│   ├── masks/
│   │   ├── mask_01/
│   │   ├── mask_02/
│   │   └── ...
│   │
│   ├── metadata/
│   │   ├── metadata.csv
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   │
│   └── splits/
│       └── split_seed_42.json
│
├── src/
│   ├── dataset.py
│   ├── transforms.py
│   ├── losses.py
│   ├── sampler.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   └── utils.py
│
├── configs/
│   ├── baseline.yaml
│   ├── convnext_base.yaml
│   └── efficientnet_v2m.yaml
│
├── notebooks/
│   ├── 01_dataset_analysis.ipynb
│   ├── 02_training_analysis.ipynb
│   └── 03_error_analysis.ipynb
│
├── checkpoints/
├── results/
│   ├── metrics/
│   ├── predictions/
│   └── plots/
│
├── logs/
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 5. Environment setup

Recommended:

```text
Python 3.11
PyTorch with CUDA
Torchvision
timm
Albumentations
OpenCV
Pillow
NumPy
Pandas
scikit-learn
Matplotlib
Seaborn
TensorBoard
PyYAML
```

Example setup:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux:

```bash
source .venv/bin/activate
```

Install PyTorch using the CUDA wheel appropriate for your installed NVIDIA driver.

Then:

```bash
pip install timm albumentations opencv-python pillow numpy pandas scikit-learn matplotlib seaborn tensorboard pyyaml tqdm
```

Verify CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Expected:

```text
True
NVIDIA GeForce RTX 4050 Laptop GPU
```

---

# 6. Hardware target

Primary hardware:

```text
GPU: NVIDIA RTX 4050 Laptop GPU
VRAM: 6 GB
```

Recommended starting configuration:

```yaml
image_size: 288
batch_size: 16
gradient_accumulation: 2
amp: true
num_workers: 4
pin_memory: true
persistent_workers: true
```

Effective batch size:

```text
16 × 2 = 32
```

If the GPU memory is insufficient at 288×288:

```text
image_size = 224
```

and keep the rest unchanged.

---

# 7. Dataset preparation

## 7.1 Clone the dataset repository

```bash
git clone https://github.com/Callmewuxin/fish4konwledge.git
```

Place the repository or its `fish_image` directory under:

```text
dataset/raw/
```

Result:

```text
dataset/raw/fish_image/
├── fish_01/
├── fish_02/
├── ...
└── fish_23/
```

---

# 8. Metadata generation

Create a metadata table with at least:

```text
image_path
class_id
class_name
trajectory_id
fish_id
width
height
```

Example:

```csv
image_path,class_id,class_name,trajectory_id,fish_id,width,height
fish_01/xxx.png,0,Dascyllus reticulatus,000000009598,5281,200,150
fish_01/yyy.png,0,Dascyllus reticulatus,000000009598,5283,200,150
```

The trajectory ID must be parsed from the filename or official reverse mapping, not randomly generated.

---

# 9. Metadata validation

Before training, verify:

### Total images

```text
~27,370
```

### Number of classes

```text
23
```

### No missing classes

Every class should appear in the metadata.

### No corrupted images

Open every image once during the dataset validation stage.

### No duplicate file names

Each path must uniquely identify one image.

### Consistent labels

The directory/class label must agree with the official mapping.

---

# 10. Required dataset analysis

Run the dataset-analysis notebook before any serious model training.

Generate:

### Plot 1 — Class distribution

A horizontal bar chart showing the number of images per species.

### Plot 2 — Log-scale class distribution

Use a logarithmic y-axis to make rare species visible.

### Plot 3 — Number of trajectories per species

This is important because image count alone does not describe independent observations.

### Plot 4 — Images per trajectory

Shows how correlated the frames are.

### Plot 5 — Image dimensions

Plot:

```text
width distribution
height distribution
aspect-ratio distribution
```

### Plot 6 — Class sample gallery

Show several examples from all 23 classes.

### Plot 7 — Mask gallery

Display:

```text
original image
mask
masked fish
```

### Plot 8 — Dataset split verification

Display:

```text
train images
validation images
test images

train trajectories
validation trajectories
test trajectories
```

---

# 11. Train/validation/test split

Recommended target:

```text
Train: 70%
Validation: 15%
Test: 15%
```

The split unit is:

```text
trajectory_id
```

not:

```text
image
```

The split should be approximately stratified by species while ensuring that every trajectory exists in exactly one split.

---

# 12. Rare-class handling during splitting

Several species contain very few examples/trajectories.

Therefore do not blindly run a normal `train_test_split` on image paths.

For rare classes:

```text
at least one trajectory → validation
at least one trajectory → test
remaining trajectories → train
```

when the class has enough trajectories to support this.

For classes with extremely few trajectories, document the exact split rather than silently creating an unstable random partition.

Use a fixed seed:

```text
seed = 42
```

and save the split file.

---

# 13. Split integrity checks

The training pipeline must fail if trajectory leakage is detected.

Example:

```python
assert train_trajectories.isdisjoint(val_trajectories)
assert train_trajectories.isdisjoint(test_trajectories)
assert val_trajectories.isdisjoint(test_trajectories)
```

Also verify:

```text
class distribution
trajectory distribution
image counts
```

before training.

---

# 14. Input preprocessing

The project should compare:

## Experiment A

Original RGB image.

## Experiment B

Mask-derived background suppression.

```text
original RGB
      ×
fish mask
      ↓
masked RGB
```

## Experiment C

Four-channel experiment:

```text
R
G
B
Mask
```

Start with 3-channel pretrained networks first.

Do not introduce a custom 4-channel model until the baseline pipeline is stable.

---

# 15. Recommended image sizes

Primary configuration:

```text
288 × 288
```

Secondary experiments:

```text
224 × 224
320 × 320
```

Because the Fish4Knowledge images themselves are relatively small, increasing the input resolution indefinitely will not create additional image information.

---

# 16. Training augmentations

Use moderate augmentations.

Recommended:

```text
RandomResizedCrop
HorizontalFlip
RandomRotation
ColorJitter
RandomAffine
GaussianBlur
```

Approximate starting values:

```text
Horizontal flip: 0.50
Rotation: ±15°
Brightness: ±20%
Contrast: ±20%
Saturation: ±25%
Hue: ±5–8%
Blur: low probability
Perspective: low probability
```

Do not use extremely destructive geometric transformations because species identification depends on body shape, fins, markings and proportions.

---

# 17. Underwater-specific augmentation

Add transformations that simulate underwater conditions.

## 17.1 Color cast

Simulate:

```text
blue/green cast
red attenuation
reduced saturation
```

## 17.2 Turbidity

Simulate:

```text
blur
contrast loss
light haze
backscatter
```

## 17.3 Lighting

Simulate:

```text
dark scenes
bright scenes
uneven illumination
```

The purpose is to prevent the network from memorizing one camera/environmental appearance.

---

# 18. Class imbalance strategy

The dataset has extreme long-tail imbalance.

Do not rely on ordinary accuracy alone.

Primary metrics:

```text
Macro F1
Balanced Accuracy
Macro Recall
```

Secondary metrics:

```text
Overall Accuracy
Weighted F1
Top-3 Accuracy
Top-5 Accuracy
```

Compare these training approaches:

```text
Experiment A: CrossEntropy
Experiment B: Weighted CrossEntropy
Experiment C: WeightedRandomSampler
Experiment D: Class-balanced loss
Experiment E: Balanced Softmax / logit adjustment
```

Do not apply extreme oversampling and aggressive class weighting simultaneously without validating the effect.

---

# 19. Recommended model strategy

## Main model

```text
ConvNeXt-Base
```

with:

```text
ImageNet pretrained weights
23-class output head
```

## Secondary model

```text
EfficientNetV2-M
```

The secondary model is used for ensemble diversity, not just parameter count.

A larger model is not automatically better on a small, low-resolution dataset.

---

# 20. Classifier head

Recommended starting head:

```text
Backbone
   ↓
Global Pool
   ↓
LayerNorm
   ↓
Dropout(0.2)
   ↓
Linear
   ↓
23 classes
```

Keep the head relatively simple initially.

---

# 21. Transfer-learning schedule

Use three phases.

## Phase 1 — Head warm-up

Freeze backbone.

```text
2–4 epochs
LR ≈ 1e-3
```

Train only the classifier head.

---

## Phase 2 — Full fine-tuning

Unfreeze the full network.

Starting learning rates:

```text
classifier ≈ 1e-4
late backbone ≈ 3e-5
early backbone ≈ 1e-5
```

Use discriminative learning rates.

---

## Phase 3 — Precision fine-tuning

Take the best checkpoint.

Train an additional:

```text
5–10 epochs
```

with lower learning rate and less aggressive augmentation.

---

# 22. Optimizer

Recommended:

```text
AdamW
```

Starting configuration:

```yaml
lr: 1.0e-4
weight_decay: 1.0e-4
betas: [0.9, 0.999]
```

Tune only after obtaining a stable baseline.

---

# 23. Learning-rate schedule

Use:

```text
3 epoch warm-up
+
cosine decay
```

Example:

```text
Epoch 1–3:
warm-up

Epoch 4–40/50:
cosine decay
```

End near:

```text
1e-6
```

---

# 24. Loss

Start with:

```python
CrossEntropyLoss(label_smoothing=0.1)
```

Then compare:

```text
CrossEntropy
Weighted CrossEntropy
Focal Loss
Class-Balanced Loss
Balanced Softmax
```

Select the loss using:

```text
validation Macro F1
```

not validation accuracy alone.

---

# 25. MixUp and CutMix

Use conservatively.

Starting probabilities:

```text
MixUp: 0.15
CutMix: 0.15
```

Because fish morphology is important, excessive CutMix can hide the very features needed for classification.

---

# 26. EMA

Maintain an Exponential Moving Average of model parameters.

At validation/test time:

```text
evaluate EMA model
```

rather than assuming the final raw checkpoint is optimal.

---

# 27. Mixed precision

Enable CUDA mixed precision.

Conceptually:

```python
with torch.autocast(device_type="cuda", dtype=torch.float16):
    outputs = model(images)
    loss = criterion(outputs, labels)
```

Use a CUDA gradient scaler for stable FP16 training when required by the implementation.

This reduces memory usage and improves throughput on the RTX 4050.

---

# 28. Training configuration — recommended first serious run

```yaml
seed: 42

dataset:
  num_classes: 23
  split:
    train: 0.70
    val: 0.15
    test: 0.15
  group_key: trajectory_id

model:
  name: convnext_base
  pretrained: true
  num_classes: 23
  dropout: 0.2

image:
  size: 288

training:
  epochs: 50
  batch_size: 16
  gradient_accumulation: 2
  optimizer: adamw
  lr: 1.0e-4
  weight_decay: 1.0e-4
  warmup_epochs: 3
  scheduler: cosine
  label_smoothing: 0.1
  amp: true
  ema: true

imbalance:
  method: class_balanced

augmentation:
  horizontal_flip: 0.5
  rotation: 15
  color_jitter: true
  underwater_aug: true

early_stopping:
  monitor: macro_f1
  patience: 8

dataloader:
  num_workers: 4
  pin_memory: true
  persistent_workers: true
```

---

# 29. Training command

Example:

```bash
python src/train.py --config configs/convnext_base.yaml
```

The script should save:

```text
checkpoints/
├── best_macro_f1.pt
├── best_accuracy.pt
└── last.pt
```

and:

```text
results/
├── metrics/
├── predictions/
└── plots/
```

---

# 30. Checkpoint policy

Save at minimum:

```text
best_macro_f1.pt
best_balanced_accuracy.pt
best_accuracy.pt
last.pt
```

The primary checkpoint is:

```text
best_macro_f1.pt
```

because the dataset is highly imbalanced.

---

# 31. Early stopping

Recommended:

```text
monitor = validation Macro F1
patience = 8
```

Do not select the final model based on test performance.

The test set remains untouched until the final experiment is frozen.

---

# 32. Evaluation metrics

For the validation and final test sets calculate:

### Overall

```text
Accuracy
```

### Class-balanced

```text
Macro Precision
Macro Recall
Macro F1
Balanced Accuracy
```

### Ranking

```text
Top-1 accuracy
Top-3 accuracy
Top-5 accuracy
```

### Probabilistic

```text
ROC-AUC
PR-AUC
Expected Calibration Error
```

---

# 33. Required plots

Every serious run should generate these plots.

## Plot 1 — Training vs validation loss

```text
epoch → loss
```

Purpose:

- detect overfitting
- detect underfitting
- compare experiments

---

## Plot 2 — Training vs validation accuracy

```text
epoch → accuracy
```

---

## Plot 3 — Validation Macro F1

```text
epoch → macro F1
```

This is the main model-selection curve.

---

## Plot 4 — Learning rate

```text
epoch → learning rate
```

Useful for verifying the warm-up + cosine schedule.

---

## Plot 5 — Normalized confusion matrix

23 × 23.

Use normalized row values so rare classes remain interpretable.

---

## Plot 6 — Raw confusion matrix

Useful to understand absolute error counts.

---

## Plot 7 — Per-class F1

Bar chart:

```text
species → F1
```

---

## Plot 8 — Per-class recall

Important for discovering minority classes that the model is ignoring.

---

## Plot 9 — Per-class precision

Useful for identifying classes with many false positives.

---

## Plot 10 — ROC curves

One-vs-rest ROC curves for all species plus macro average.

---

## Plot 11 — Precision-Recall curves

One-vs-rest PR curves for all species plus macro average.

This is particularly informative for rare classes.

---

## Plot 12 — Top-k accuracy

Compare:

```text
Top-1
Top-3
Top-5
```

---

## Plot 13 — Reliability diagram

Compare:

```text
confidence
vs
empirical accuracy
```

Report:

```text
Expected Calibration Error
```

---

## Plot 14 — Most-confused class pairs

Display the top class confusion pairs.

Example:

```text
Species A → Species B
Species C → Species D
...
```

---

## Plot 15 — Error gallery

Show:

```text
true label
predicted label
confidence
image
```

for the highest-confidence wrong predictions.

---

# 34. Error analysis

After the first strong model is trained, run inference over the validation set.

Save each prediction:

```text
image_path
trajectory_id
true_class
predicted_class
confidence
top_5_classes
top_5_probabilities
```

Then inspect:

### A. High-confidence wrong predictions

These are the most dangerous errors.

### B. Low-confidence correct predictions

These indicate ambiguous samples.

### C. Rare-class failures

Focus on the long-tail classes.

### D. Repeated confusion pairs

Determine whether the issue is:

```text
pose
background
color
occlusion
motion blur
poor illumination
low resolution
similar morphology
```

---

# 35. Hard-example mining

Use the first model to identify hard samples.

Pipeline:

```text
Model V1
   ↓
Prediction dump
   ↓
Find wrong / uncertain samples
   ↓
Analyze dominant error modes
   ↓
Adjust augmentation / sampling / preprocessing
   ↓
Train Model V2
```

Do not simply increase the number of epochs when performance plateaus.

---

# 36. Test-time augmentation (TTA)

For final evaluation use lightweight TTA.

Recommended:

```text
original
horizontal flip
slightly different crop/resize
```

Average the class probabilities:

```text
P_final =
(P_original +
 P_flip +
 P_crop) / 3
```

Then:

```text
prediction = argmax(P_final)
```

Do not use aggressive transformations during TTA.

---

# 37. Model ensemble

Train:

```text
Model A = ConvNeXt-Base
Model B = EfficientNetV2-M
```

For each image:

```text
P_A = softmax(Model A)
P_B = softmax(Model B)
```

Then:

```text
P_final = α P_A + (1 - α) P_B
```

Try:

```text
α = 0.5
α = 0.6
α = 0.7
```

Select the best value on validation Macro F1.

Only after choosing the ensemble configuration should the final test set be evaluated.

---

# 38. Recommended experiment matrix

Run experiments in controlled order.

| Experiment | Backbone | Input | Imbalance | Mask | TTA |
|---|---|---:|---|---|---|
| E1 | ResNet50 baseline | 224 | None | No | No |
| E2 | ConvNeXt-Base | 224 | CE | No | No |
| E3 | ConvNeXt-Base | 224 | Weighted CE | No | No |
| E4 | ConvNeXt-Base | 224 | Class-balanced | No | No |
| E5 | ConvNeXt-Base | 288 | Class-balanced | No | No |
| E6 | ConvNeXt-Base | 288 | Class-balanced | Yes | No |
| E7 | ConvNeXt-Base | 320 | Class-balanced | Yes | No |
| E8 | EfficientNetV2-M | 288 | Class-balanced | Yes | No |
| E9 | Ensemble | 288 | Best | Yes | No |
| E10 | Ensemble | 288 | Best | Yes | Yes |

Every experiment should use the exact same test protocol.

---

# 39. Experiment tracking

Each run should produce:

```text
runs/
├── run_001/
│   ├── config.yaml
│   ├── metrics.csv
│   ├── best.pt
│   ├── classification_report.csv
│   └── plots/
│
├── run_002/
│   └── ...
└── ...
```

Track:

```text
seed
model
input resolution
augmentation
loss
sampling strategy
learning rate
weight decay
epochs
best epoch
validation accuracy
validation Macro F1
validation balanced accuracy
```

Use TensorBoard:

```bash
tensorboard --logdir logs/
```

---

# 40. Reproducibility

Set:

```text
random seed = 42
```

for:

```text
Python
NumPy
PyTorch
CUDA
data split
```

Save:

```text
split file
configuration
class mapping
package versions
GPU information
best checkpoint
```

A trained model without its split/configuration is not fully reproducible.

---

# 41. Final evaluation protocol

The test set must remain untouched during:

```text
architecture selection
loss selection
augmentation selection
resolution selection
ensemble weight selection
checkpoint selection
```

Final workflow:

```text
Train
↓
Validate
↓
Select best configuration
↓
Freeze all choices
↓
Run test exactly once
↓
Generate final report
```

---

# 42. Final report table

Generate:

| Metric | ConvNeXt-Base | EfficientNetV2-M | Ensemble |
|---|---:|---:|---:|
| Accuracy | — | — | — |
| Balanced Accuracy | — | — | — |
| Macro Precision | — | — | — |
| Macro Recall | — | — | — |
| Macro F1 | — | — | — |
| Weighted F1 | — | — | — |
| Top-3 Accuracy | — | — | — |
| Top-5 Accuracy | — | — | — |
| ROC-AUC | — | — | — |
| ECE | — | — | — |

Populate these values only from the frozen test set.

---

# 43. RTX 4050 runtime estimate

These are planning estimates rather than guaranteed timings.

Assumptions:

```text
RTX 4050 Laptop
6 GB VRAM
AMP enabled
local SSD
batch size ≈ 16
~19k training images after a 70/15/15 image-count split
```

Approximate order of magnitude:

| Model | Resolution | Estimated epoch | 50 epochs |
|---|---:|---:|---:|
| ResNet50 | 224 | 1–3 min | 1–2.5 h |
| ConvNeXt-Tiny | 224 | 2–4 min | 1.5–3.5 h |
| ConvNeXt-Base | 224 | 3–6 min | 2.5–5 h |
| ConvNeXt-Base | 288 | 5–8 min | 4–7 h |
| EfficientNetV2-M | 288 | 4–8 min | 3.5–7 h |
| EfficientNetV2-M | 320 | 5–10 min | 4–8.5 h |

Actual runtime depends strongly on:

```text
GPU TGP
CPU
RAM
storage speed
augmentation cost
DataLoader workers
thermal throttling
batch size
```

### Recommended planning budget

Single high-quality model:

```text
~3–8 GPU hours
```

Full optimization campaign:

```text
~15–35 GPU hours
```

---

# 44. RTX 4050 memory strategy

If you hit:

```text
CUDA out of memory
```

reduce in this order:

### Option 1

```text
batch_size: 16 → 8
```

### Option 2

Keep effective batch size using accumulation:

```text
batch_size: 8
gradient_accumulation: 4
```

### Option 3

```text
288 → 224
```

### Option 4

Enable gradient checkpointing if implemented.

Do not immediately remove AMP.

---

# 45. Recommended first serious experiment

Run this first:

```text
Dataset:
Fish4Knowledge 23 classes

Split:
70/15/15 by trajectory

Backbone:
ConvNeXt-Base pretrained

Resolution:
288 × 288

Batch:
16

Gradient accumulation:
2

Loss:
CrossEntropy + label smoothing 0.1

Imbalance:
class-balanced method

Augmentation:
moderate + underwater-specific

Optimizer:
AdamW

LR:
1e-4

Scheduler:
3 epoch warm-up + cosine

AMP:
enabled

EMA:
enabled

Epochs:
50

Early stopping:
8 epochs

Monitor:
Validation Macro F1
```

This gives you a strong reference point before expanding the experiment matrix.

---

# 46. Recommended final high-accuracy configuration

After the experiments above, the target final pipeline is:

```text
Trajectory-aware split
        ↓
Mask/original preprocessing comparison
        ↓
ConvNeXt-Base 288
        ↓
Class-balanced optimization
        ↓
Underwater augmentations
        ↓
AMP
        ↓
EMA
        ↓
Hard-example mining
        ↓
EfficientNetV2-M second model
        ↓
Validation-selected ensemble weights
        ↓
Light TTA
        ↓
Final frozen test evaluation
```

---

# 47. What counts as a good result

Do not judge the project by accuracy alone.

A useful interpretation is:

```text
Accuracy
+
Macro F1
+
Balanced Accuracy
+
rare-class recall
+
confusion matrix
```

A model scoring:

```text
99% accuracy
```

with poor minority-class recall may be less useful than a model scoring:

```text
96% accuracy
```

with strong performance across all 23 classes.

Because this is a highly imbalanced dataset, report both:

```text
Overall Accuracy
Macro F1
```

prominently.

---

# 48. Optional research-grade comparison

For an especially strong project, report two protocols:

## Protocol A — Image-level random split

Used only as a diagnostic comparison.

## Protocol B — Trajectory-aware split

Used as the primary, credible benchmark.

Report:

```text
Random split accuracy
Random split Macro F1

Trajectory split accuracy
Trajectory split Macro F1
```

The difference quantifies the effect of temporal/trajectory correlation.

---

# 49. Optional deployment path

After the best classifier is selected:

```text
Teacher/ensemble
       ↓
knowledge distillation
       ↓
smaller student
       ↓
ONNX
       ↓
TensorRT / NPU deployment
```

This is recommended if the eventual goal is real-time inference on an edge device.

Do not optimize for edge deployment before the highest-quality baseline is established.

---

# 50. Minimum deliverables for the completed project

The final repository should contain:

```text
[ ] metadata.csv
[ ] train.csv
[ ] val.csv
[ ] test.csv
[ ] split integrity report
[ ] dataset analysis plots
[ ] trained ConvNeXt checkpoint
[ ] trained EfficientNetV2 checkpoint
[ ] ensemble implementation
[ ] TTA implementation
[ ] classification report
[ ] confusion matrix
[ ] per-class precision/recall/F1
[ ] Macro F1
[ ] balanced accuracy
[ ] top-k accuracy
[ ] ROC/PR plots
[ ] calibration plot
[ ] error gallery
[ ] hard-example analysis
[ ] final test results
[ ] exact training configuration
```

---

# 51. Recommended final results directory

```text
results/
├── metrics/
│   ├── validation_metrics.json
│   ├── test_metrics.json
│   └── classification_report.csv
│
├── predictions/
│   ├── val_predictions.csv
│   └── test_predictions.csv
│
└── plots/
    ├── class_distribution.png
    ├── class_distribution_log.png
    ├── trajectory_distribution.png
    ├── image_dimensions.png
    ├── train_val_loss.png
    ├── train_val_accuracy.png
    ├── validation_macro_f1.png
    ├── learning_rate.png
    ├── confusion_matrix_raw.png
    ├── confusion_matrix_normalized.png
    ├── per_class_precision.png
    ├── per_class_recall.png
    ├── per_class_f1.png
    ├── roc_curves.png
    ├── pr_curves.png
    ├── top_k_accuracy.png
    ├── calibration.png
    ├── confusion_pairs.png
    └── error_gallery.png
```

---

# 52. Final implementation checklist

## Dataset

- [ ] Dataset downloaded
- [ ] 23 classes verified
- [ ] Metadata generated
- [ ] Corrupted images removed
- [ ] Duplicates checked
- [ ] Trajectory IDs extracted
- [ ] Masks located and validated

## Split

- [ ] 70/15/15 split created
- [ ] Split is trajectory-aware
- [ ] No trajectory overlap
- [ ] Rare classes inspected
- [ ] Split saved to disk

## Training

- [ ] Pretrained weights enabled
- [ ] AMP enabled
- [ ] Class imbalance strategy selected
- [ ] Underwater augmentation enabled
- [ ] EMA enabled
- [ ] Cosine schedule enabled
- [ ] Macro F1 monitored

## Evaluation

- [ ] Confusion matrix
- [ ] Per-class metrics
- [ ] Macro F1
- [ ] Balanced accuracy
- [ ] Top-k accuracy
- [ ] ROC/PR
- [ ] Calibration
- [ ] Error gallery

## Optimization

- [ ] ConvNeXt-Base
- [ ] EfficientNetV2-M
- [ ] Mask experiment
- [ ] Resolution experiment
- [ ] Loss experiment
- [ ] TTA
- [ ] Ensemble

## Final

- [ ] Test set frozen until final run
- [ ] Final configuration documented
- [ ] Checkpoint archived
- [ ] Results exported
- [ ] README updated with actual metrics

---

# 53. References

1. Fish4Knowledge GitHub dataset repository  
   https://github.com/Callmewuxin/fish4konwledge

2. Fish4Knowledge Fish Recognition Ground-Truth  
   https://homepages.inf.ed.ac.uk/rbf/Fish4Knowledge/GROUNDTRUTH/RECOG/

3. Fish4Knowledge project documentation  
   https://groups.inf.ed.ac.uk/vision/DATASETS/FISH4KNOWLEDGE/

4. PyTorch Transfer Learning Tutorial  
   https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

5. Torchvision ConvNeXt documentation  
   https://docs.pytorch.org/vision/stable/models/convnext.html

6. Torchvision EfficientNetV2 documentation  
   https://docs.pytorch.org/vision/stable/models/efficientnetv2.html

---

# 54. Final recommended command sequence

```bash
# 1. Prepare environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify CUDA
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# 4. Build metadata
python src/build_metadata.py

# 5. Build trajectory-aware split
python src/create_split.py --seed 42

# 6. Analyze dataset
python notebooks/01_dataset_analysis.ipynb

# 7. Train baseline
python src/train.py --config configs/baseline.yaml

# 8. Train ConvNeXt-Base
python src/train.py --config configs/convnext_base.yaml

# 9. Evaluate
python src/evaluate.py --checkpoint checkpoints/best_macro_f1.pt

# 10. Train EfficientNetV2-M
python src/train.py --config configs/efficientnet_v2m.yaml

# 11. Ensemble + TTA
python src/inference.py --ensemble

# 12. Generate final report
python src/evaluate.py --final-test
```

> **Important:** the exact script names above are the recommended project interface. Implement the scripts to match this interface so that the repository remains reproducible and easy to use.

---

## Success criterion

The project is considered complete only when it can reproduce:

```text
Dataset
→ metadata
→ leakage-free trajectory split
→ training
→ checkpoint selection
→ evaluation
→ plots
→ error analysis
→ ensemble/TTA
→ final test report
```

with one command sequence and without manually editing notebook state.
