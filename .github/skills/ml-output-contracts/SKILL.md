---
name: ml-output-contracts
description: Use when producing any substantial finding, audit, recommendation, or experimental claim in this repo — provides the shared structured report template (Summary/Findings/Evidence/Risk/Recommendation/Confidence) plus domain-specific extensions for research, data, and optimization outputs. Load before writing a final answer to a non-trivial ML task.
---

# ML Output Contracts

## Base contract

Use this shape for any finding, audit, or recommendation of consequence.
Skip it for small talk, quick lookups, or single-fact answers.

```
## Summary
## Findings
## Evidence            (Observed / Inferred / Hypothesized / Unknown — see ml-general instructions)
## Risk / Impact
## Recommendation
## Implementation Required
## Confidence           (High / Medium / Low, with a one-line reason)
## Open Questions
```

## Extensions by agent type

**Research outputs** (research-explorer) append:
```
## Sources
## Competing Explanations
## Limitations
```

**Data outputs** (dataset-auditor, preprocessing-engineer) append:
```
## Dataset Statistics
## Quality Issues
## Leakage Risks
## Recommended Actions
```

**Optimization outputs** (perf-optimizer) append:
```
## Baseline Benchmark
## Optimized Benchmark
## Performance Delta
## Accuracy Delta
```

**Scientific review outputs** (scientific-critic) append:
```
## Falsification Test
## Verdict            (Supported / Insufficient Evidence / Not Supported)
```

## Rules

- Every row in "Findings" traces to something in "Evidence" — don't assert
  a finding with no evidence line behind it.
- "Confidence: High" requires Observed evidence, not Inferred or
  Hypothesized. If your best evidence is Hypothesized, cap confidence at
  Low and say what would raise it.
- Keep "Summary" to 2-4 sentences. It should be readable without the rest
  of the report.
- Omit sections that are genuinely empty rather than writing "N/A" — e.g. no
  Open Questions is fine to state as "None."
