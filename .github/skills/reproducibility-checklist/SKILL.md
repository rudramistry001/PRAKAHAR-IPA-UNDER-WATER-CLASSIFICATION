---
name: reproducibility-checklist
description: Use when asked to prepare a repository for reproducible research, audit reproducibility, or before accepting an experimental result as reliable — checklist of what must be tracked (git commit, dataset version, config, seeds, environment, hardware, checkpoint, preprocessing, evaluation protocol) and how to check for each.
---

# Reproducibility Checklist

For each item, report Observed (found and pinned), Inferred (probably
present but not directly verified), or Unknown (not found / not checked).
Do not mark an item "OK" without an Observed basis.

| Item | What to check |
|---|---|
| Git commit | Is the exact commit hash recorded with each run/result (not just branch name)? |
| Dataset version | Is the dataset pinned by hash/version/DVC-tag, not just a path that could change contents? |
| Configuration | Is the full resolved config (not just overrides) saved per run? |
| Random seeds | Are seeds set for Python, NumPy, the framework's RNG, and — if relevant — CUDA/cuDNN determinism flags? Is the seed itself logged? |
| Environment | Is there a lockfile (requirements.txt with pins, poetry.lock, conda env export, or container image digest)? |
| Dependency versions | Are framework/library versions pinned, especially CUDA-adjacent ones (torch, torchvision, cudnn) that affect numerics? |
| Hardware | Is GPU model, count, and CUDA version recorded? Multi-GPU non-determinism (e.g. from certain ops) noted if relevant? |
| Model checkpoint | Is the exact checkpoint (with hash) that produced a reported number saved and referenced? |
| Preprocessing configuration | Is the preprocessing pipeline version/config tied to the run, not just "current pipeline"? |
| Training parameters | Full hyperparameters logged (not just the ones that differ from defaults)? |
| Evaluation protocol | Is the exact eval script/metric implementation version recorded — metric implementations can silently change definitions between library versions. |

## Producing the checklist output

Output a table with columns: Item | Status (Observed/Inferred/Unknown) |
Evidence | Gap (what's missing) | Suggested fix. Sort so Unknown items
needing action float to the top.

## Common gaps to flag proactively

- Seeds set for Python/NumPy but not the framework or CUDA determinism
  flags (a frequent partial-fix).
- "Latest" dataset paths / floating tags instead of pinned versions.
- Config saved as CLI overrides only, not the fully resolved config
  (defaults silently drift when the code changes).
- No record of which checkpoint (step/epoch) a reported number came from
  when multiple checkpoints exist.
