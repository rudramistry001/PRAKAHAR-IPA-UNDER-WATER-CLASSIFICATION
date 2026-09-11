---
description: "Core evidence, safety, and correctness rules for all ML/CV/remote-sensing work in this repo. Applies automatically to Python, config, and notebook edits."
applyTo: "**/*.py,**/*.ipynb,**/*.yaml,**/*.yml,**/*.toml,**/*.cfg"
---

# ML General Instructions

These rules apply to every agent in this repository, whether invoked directly
or as a subagent. They are repo-wide because they are conventions, not a
persona — see the Agent vs Skill vs Instruction decision matrix in
`docs/agent-architecture.md` for why this lives here instead of being
repeated in every `.agent.md`.

## Evidence policy (non-negotiable)

Label every substantial claim with one of:

- **Observed** — read directly from code, config, logs, or executed output.
- **Inferred** — a reasonable conclusion from Observed facts, but not itself
  directly read (state the inference chain).
- **Hypothesized** — a plausible explanation not yet checked against evidence.
- **Unknown** — cannot be determined with available tools/access.

Never state a dataset statistic, benchmark number, paper finding, or
experimental result you did not obtain via a tool call in this session or
read directly from a repository file. If you would otherwise "recall" a
number, mark it Unknown and say what tool call or file would resolve it.

Use hedged language for anything not yet measured: "expected to improve
X" / "should reduce Y", never "this improves X" without a benchmark to cite.

## No fabrication

Never invent: dataset statistics, benchmark results, paper citations,
package/API names, config keys, file paths, or tool names. If unsure whether
something exists, search or read for it first; if you still can't confirm it,
say so explicitly rather than guessing.

## Distinguish failure classes

When something in a training/data/eval pipeline looks wrong, classify it
explicitly as one of: **Bug**, **Inefficiency**, **Poor hyperparameter
choice**, or **Legitimate scientific design choice** — do not default to
"bug" just because a value looks unusual.

## Leakage discipline

Before trusting any metric improvement, consider: subject/patient/scene
identity leakage, temporal leakage, spatial/tile overlap, duplicate or
near-duplicate samples, augmentation applied before the split, and
preprocessing statistics (normalization, scalers) fit on data that includes
validation/test samples. See the `leakage-detection` skill for the full
checklist.

## Destructive operations

Never silently modify or delete a dataset, checkpoint, or split file. Never
run destructive shell commands (`rm -rf`, force-push, dropping DB tables,
overwriting checkpoints in place) without first stating the exact command and
getting explicit confirmation. Prefer writing to a new path over overwriting.

## Reproducibility baseline

Whenever you change training/eval behavior, note in your output: the exact
git state you're aware of, config values changed, and random seed handling —
or flag them as Unknown if not observable. See the `reproducibility-checklist`
skill for the full list.

## Output discipline

Follow the shared output contract in the `ml-output-contracts` skill for any
finding, audit, or recommendation of consequence. Keep routine, low-stakes
responses conversational — the structured contract is for claims someone
might act on.
