# Editorial review and validation report

## Delivered manuscript

The revised IEEE manuscript is `research_paper/main.tex`. It retains the original title, authors, affiliation, IEEEtran conference layout, equations, figure and table labels, citation keys, bibliography style, and all reported quantitative values.

## Section-wise improvements

| Section | Editorial refinement |
|---|---|
| Abstract and index terms | Tightened the motivation, defined the active-versus-archived evidence boundary early, and stated the principal operating-point trade-off without implying universal hallucination prevention. |
| Introduction | Reorganized the motivation from factuality risk to the project scope, then to the provenance boundary and contributions. Removed generic framing and made detection distinct from mitigation. |
| Related work | Linked factuality benchmarks, DoLa, self-consistency, and uncertainty calibration to the implementation-specific research gap. |
| Problem formulation | Clarified that LM-head features use the applicable normalization and that the configured contrast coefficient is not an empirically optimized value. Equations and symbols are unchanged. |
| Methodology | Improved transitions between the system boundary, DoLa, active epinet, self-consistency, and optional answer-level detection. The entropy weight remains explicitly non-calibrated. |
| Experimental setup | Made artifact provenance, the held-out designation, validation-only threshold selection, and the missing upgraded trainer explicit. |
| Results and comparative analysis | Rewrote numerical narration as evidence-based interpretation. The ablation now states the exact 0.7488-to-0.7346 accuracy change and the 0.0032 AUROC change, rather than calling an effect meaningful without quantification. |
| Discussion | Separated the behavior of DoLa, self-consistency, and the answer-correctness ENN; retained the high-specificity/high-false-reject trade-off. |
| Limitations, future work, and conclusion | Sharpened the provenance and class-imbalance limitations and aligned the conclusion with the evidence boundary. |

## Technical statements requiring author verification

1. Confirm that the archived 211-example split is genuinely held out relative to every upgraded ENN training and threshold-selection run.
2. Restore or identify the commit containing the archived upgraded trainer, feature-construction code, and evaluation code before describing the archived results as reproducible.
3. Archive the precise label definition for the positive `reliable/correct` class, prompt templates, question IDs, split assignment, and threshold-selection procedure.
4. Confirm that the full-cache active run used 600 C4 documents, 56,515 token examples, a 90/10 document-disjoint split, ten epochs, batch size 32, and eight training epistemic samples, as reported in the saved audit artifacts.
5. Verify whether the active C4 cache has sufficient source-document provenance for an independent reconstruction; the current manuscript appropriately states that it does not retain complete provenance.
6. Before submission, rerun or validate the reported software environment against a lockfile. The exact versions are audited, but `requirements.txt` does not pin them.
7. Add per-dataset results for TruthfulQA and HaluEval-QA if the paper is intended to make comparative generalization claims; the combined 211-example result cannot resolve dataset-specific behavior.

## Formatting and citation review

- IEEEtran conference class, section hierarchy, numbering, author block, bibliography style, figure/table labels, and citation keys are preserved.
- The environment table was adjusted to use edge suppression and compatible column widths, reducing overflow risk in the two-column layout.
- The figures are referenced where their evidence is discussed and distinguish active-code diagrams from archived-result visualizations.
- Citation metadata and bibliography entries remain unchanged. The pre-existing reference-verification report identifies authoritative landing pages for all cited sources.
- The revised source was compiled with Tectonic and visually inspected as a six-page letter-size IEEE PDF. Any later author-driven change should receive the same build check. The source uses `IEEEtran`, `cite`, `amsmath`, `booktabs`, `graphicx`, `url`, `xcolor`, and `array`.

## Preservation validation

The revision preserves the original technical claims and reported values, including the 211-example composition (169 TruthfulQA and 42 HaluEval-QA; 54 positive and 157 negative labels); the active TinyLlama architecture and DoLa candidate layers; the active epinet dimensions, loss, and training setup; the archived comparison table; ablation values; bootstrap intervals; McNemar results; confusion counts; and stated limitations. No equations, algorithms, numerical metrics, benchmark comparisons, datasets, citations, or author information were fabricated or altered.

The paper deliberately preserves the central scientific qualification: active C4 next-token uncertainty is not an answer-correctness probability, and the stronger answer-level benchmark originates from archived artifacts whose trainer source is absent from the active checkout.
