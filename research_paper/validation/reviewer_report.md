# Independent reviewer report

## Reviewer 1 - AI/ML researcher

### Major concerns

1. The active ENN objective is next-token prediction, whereas hallucination detection is answer-level correctness. The manuscript now separates these objectives and does not credit the active ENN with the archived answer-level scores.
2. The archived upgraded ENN source is missing. The paper reports the results with an explicit reproducibility qualification and requests source recovery.
3. The conservative threshold has low FAR but very high FRR. Accuracy alone would be misleading under the 157/54 class imbalance.

### Required corrections

- State the exact target and uncertainty level for every ENN variant.
- Report AUROC, AUPRC, FAR, FRR, balanced accuracy, and calibration, not accuracy alone.
- Avoid calling DoLa or ENN a hallucination-prevention method without a generation-level comparison.

## Reviewer 2 - IEEE reviewer

### Major concerns

1. Results from the active executable path and archived upgraded path must not be merged.
2. All external references need authoritative metadata.
3. Figures must be tied to saved artifacts and captions must explain the evidence boundary.

### Corrections applied

- Added a provenance table and active/archived boundary in Sections III-V.
- Added IEEE-style figures, equations, algorithm, tables, and bibliography.
- Added explicit negative findings: E5b and E7 predictions are identical; the complete combined architecture is not significantly better than E5b.

## Reviewer 3 - critical evaluator

### Major concerns

1. Dataset imbalance makes always-reject accuracy 0.744; this baseline is now reported.
2. Threshold choice must be validation-only; the paper states this and reports multiple operating regimes.
3. The 211-example set combines TruthfulQA and HaluEval-QA, so domain shift and label semantics need separate reporting.
4. The active cache and source revisions are not sufficient for exact data reproduction.

### Recommended experiments

- Recover the upgraded trainer and commit it with a lockfile.
- Evaluate full benchmark splits and a fresh domain-shift test set.
- Report per-dataset metrics and confidence intervals.
- Compare abstention against retrieval, verifier, and calibrated Base baselines.

## Publication-readiness verdict

Technically coherent as a reproducibility and audit paper, but not ready for a strong empirical “hallucination reduction” claim until the missing upgraded source is restored and the benchmark is independently rerun.
