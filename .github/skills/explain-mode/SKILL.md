---
name: explain-mode
description: Use when the user asks to "explain", "teach me", turns on EXPLAIN MODE, or is clearly trying to learn a concept rather than just get a change made — provides the WHAT/WHY/HOW/MATH/ASSUMPTIONS/TRADE-OFFS/FAILURE MODES/ALTERNATIVES explanation template and rules for scaling explanation depth to complexity.
---

# Explain Mode

## Template

For a decision or change worth teaching, structure the explanation as:

```
WHAT:            One or two sentences — what changed or what the concept is.
WHY:             The reasoning or motivating problem.
HOW:             Mechanism — how it actually works, concretely.
MATHEMATICS:     Only if there's real math content; omit otherwise.
ASSUMPTIONS:     What has to be true for this to be a good idea.
TRADE-OFFS:      What you give up.
FAILURE MODES:   How this breaks, and what it looks like when it does.
ALTERNATIVES:    What else could have been done, briefly.
```

## Scaling depth to complexity

- **Trivial/mechanical** (renaming a variable, fixing a typo, a config
  toggle with no side effects): no explain-mode template — just do it and
  say what you did in one sentence.
- **Moderate** (swapping an optimizer, changing a normalization scheme,
  picking a batch size): WHAT / WHY / HOW / TRADE-OFFS / ALTERNATIVES.
  Skip MATHEMATICS and FAILURE MODES if not illuminating.
- **Significant** (changing an architecture, a loss function, a
  train/val/test split strategy, an evaluation protocol): full template.

Never pad a simple answer to fill out the template — an unfilled or
skipped section is correct when it doesn't apply.

## Example (moderate complexity)

```
WHAT: Switched the optimizer from Adam to AdamW.
WHY: The config applies weight_decay=0.01, but plain Adam couples weight
     decay into the gradient update via the L2 term, which interacts with
     the adaptive learning rate in a way that makes the effective decay
     inconsistent across parameters.
HOW: AdamW decouples weight decay from the gradient-based update — it
     subtracts `lr * weight_decay * param` directly, rather than adding
     `weight_decay * param` into the gradient before the Adam moment
     estimates are computed.
TRADE-OFFS: AdamW's decay rate isn't directly comparable to Adam's — the
     effective regularization strength changes, so weight_decay may need
     retuning.
ALTERNATIVES: Keep Adam and set weight_decay=0 while adding decay as an
     explicit regularization term instead, or use SGD with decoupled decay.
```
