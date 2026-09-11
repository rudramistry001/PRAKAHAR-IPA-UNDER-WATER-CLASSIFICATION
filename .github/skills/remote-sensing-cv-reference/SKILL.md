---
name: remote-sensing-cv-reference
description: Use for computer vision and remote-sensing / earth-observation work — multispectral, hyperspectral, SAR, geospatial metadata, sensor differences, atmospheric effects, cloud contamination, projections, and the physical-meaning questions to ask before applying a standard CV preprocessing step to overhead or scientific imagery. Load when a task touches satellite, aerial, hyperspectral, SAR, PCB/AOI, or scientific imaging data.
---

# Remote Sensing & Scientific CV Reference

## Before applying any "standard" CV transform, ask

1. What information does this transformation preserve, and what does it
   destroy?
2. Is it physically meaningful for this sensor/modality, or is it a
   convention borrowed from RGB natural-image pipelines?
3. Is it consistent between training-time preprocessing and inference-time
   preprocessing?
4. Could it introduce leakage (e.g. per-scene normalization stats computed
   using eval-scene pixels)?

Concretely: ImageNet mean/std normalization is meaningless for 8-band
multispectral or SAR data — those statistics were computed for 8-bit RGB
photographs. Don't apply it by default; compute band-wise statistics from
the training distribution of the actual sensor.

## Multispectral / hyperspectral

- Check bit depth (8/12/16-bit, or float radiance/reflectance) before any
  normalization — naive `/255` silently clips or corrupts non-8-bit data.
- Band count and band order vary by sensor/product level; verify against
  the product's band-to-wavelength mapping rather than assuming.
- Radiance vs. top-of-atmosphere reflectance vs. surface (BOA) reflectance
  are different physical quantities — know which one is in the file before
  computing indices (NDVI, NDWI, etc.).
- Hyperspectral cubes: watch for water-absorption bands that are typically
  noisy/masked; check whether the pipeline already drops them.

## SAR (synthetic aperture radar)

- Speckle noise is multiplicative, not additive — Gaussian-noise-oriented
  denoising assumptions don't transfer directly.
- Amplitude vs. intensity vs. dB (log) scale — confirm which is stored;
  log-scale data should not be treated as if linear.
- Polarization channels (VV, VH, HH, HV) carry different physical meaning;
  don't average them like RGB channels without justification.

## Geospatial metadata & projections

- Confirm CRS/projection consistency across all inputs before any spatial
  operation (resampling, tiling, mosaicking); mismatched CRS silently
  produces spatially wrong results without erroring.
- Spatial resolution (GSD) differences between sensors mean naive resizing
  changes the physical ground footprint per pixel — note this explicitly
  when combining sources.
- Temporal resolution / revisit cadence affects what "the same location" at
  two timestamps actually represents (cloud state, phenology, tide).

## Atmospheric & acquisition effects

- Cloud and cloud-shadow contamination: check for or apply a cloud mask
  before treating pixels as valid signal; note whether the mask itself was
  produced by a model (potential secondary error source).
- Atmospheric correction level affects downstream index computation;
  compare across sources trained on inconsistent correction levels only
  with caution — this is a common source of poor generalization across
  regions/sensors.
- Sun angle / view angle / seasonal illumination differences are a common,
  under-checked source of domain shift in "same class, different region"
  generalization failures.

## PCB / AOI (automated optical inspection)

- Lighting/illumination consistency across the imaging rig matters more
  than typical CV augmentation assumes — synthetic lighting augmentation
  should match the rig's actual variation, not generic color jitter.
- Defect classes are frequently extremely imbalanced; check class balance
  and whether the evaluation metric (accuracy vs. recall on defect class)
  matches the actual cost of a missed defect.
- Registration/alignment between reference (golden) board and test board
  images should be verified before diffing — misregistration masquerades as
  a defect signal.

## Generalization checks specific to this domain

When a model is reported to "generalize," check explicitly: different
sensor, different geographic region, different season/illumination,
different atmospheric correction pipeline. A held-out split from the same
sensor/region/season is a weaker claim than cross-sensor or cross-region
generalization — state which one was actually tested.
