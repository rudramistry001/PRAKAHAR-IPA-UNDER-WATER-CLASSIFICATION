---
name: leakage-detection
description: Use whenever auditing a dataset, split, or reported metric improvement for data leakage — covers subject/temporal/spatial/duplicate/augmentation/preprocessing leakage with concrete checks for tabular, CV, and remote-sensing data. Load before concluding train/val/test splits are clean or before accepting a metric gain as real.
---

# Leakage Detection Checklist

Leakage means information from outside the training split (directly or via a
proxy) reached the model during training or was available at evaluation time
in a way that won't hold at deployment. Check each category; report
Observed / Inferred / Hypothesized / Unknown per the ml-general instructions
for each one — don't skip a category just because it seems unlikely.

## Categories to check

1. **Subject/entity identity leakage** — same patient, sensor, well,
   customer, or scene appearing in both train and eval splits (e.g. multiple
   crops or frames from one source scene in different splits).
2. **Temporal leakage** — eval data that precedes or overlaps in time with
   training data, or features computed using future information relative to
   the label's timestamp.
3. **Spatial leakage** (remote sensing / geospatial specific) — tile or
   scene overlap between splits, adjacent tiles from the same acquisition
   split across train/val, or spatial autocorrelation not accounted for in
   the split strategy.
4. **Duplicate samples** — exact duplicates across splits (hash the raw
   data, not just the filename).
5. **Near-duplicates** — augmented or lightly perturbed versions of a
   training sample appearing in eval (crops, rotations, color jitter of the
   same source image; overlapping time-series windows).
6. **Augmentation-order leakage** — augmentation applied before the
   train/val/test split instead of after, or augmented samples not
   consistently excluded from eval.
7. **Preprocessing leakage** — normalization statistics (mean/std, min/max
   scalers, PCA components, vocabulary/tokenizers, imputation values) fit on
   data that includes validation or test samples.
8. **Label leakage** — a feature that is a proxy for or derived from the
   label (e.g. a post-outcome field, a diagnosis code that implies the
   target).
9. **Feature leakage via joins** — features assembled by a join/merge that
   silently pulls in future or eval-only data.

## How to check (concrete techniques)

- Hash raw samples (not filenames) and diff train vs. eval hash sets.
- For images: perceptual hashing (e.g. pHash) or embedding-similarity
  nearest-neighbor search across splits to catch near-duplicates.
- For spatial data: plot or compute bounding-box/tile-ID overlap between
  splits; check the split was done by scene/flight/acquisition, not by
  individual chip.
- For preprocessing: grep the pipeline for `.fit(` / `fit_transform(` calls
  and confirm they run on the training subset only, before any split-mixing.
- For temporal data: confirm the split boundary is a single cutoff date/time
  and no feature window crosses it.

## Non-negotiable rules

- Do not modify the dataset or splits to "fix" leakage without explicit
  authorization — report it, propose the fix, wait for confirmation.
- Do not claim leakage exists without a specific piece of evidence (a file,
  a hash collision, an overlap count). "This looks suspicious" is a
  Hypothesized flag, not a Finding.
- A clean bill of health still needs a Confidence label — "no leakage found"
  should say what was and wasn't checked (e.g. "checked 1-8; feature joins
  not inspected — Unknown").
