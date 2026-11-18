# Methodology and repository map

## Active executable path

| Component | Source | Verified role | Evidence boundary |
|---|---|---|---|
| Base LLM | `Model_load/load_llama2.py`, `main.py` | Loads local TinyLlama-1.1B-Chat-v1.0 | Model files are local; exact runtime versions are recorded in audit reports |
| DoLa | `Neural Network/dola.py` | JSD-selected premature layer and mature-minus-premature logits | Candidate set is `[0,4,8,12,16,20]`; mature layer 21 |
| Active ENN | `Neural Network/enn_torch.py` | Frozen-prior epinet trained on C4 next-token labels | Not an answer-correctness classifier |
| Feature extraction | `Neural Network/data_prep.py` | Mature/premature C4 features and document-disjoint split | Cache stores IDs but not full source-document provenance |
| Self-consistency | `Neural Network/self_consistency.py`, `run_experiment.py` | Plurality and entropy-weighted candidate voting | Weight is a heuristic, not calibrated correctness |
| Reliability head | `Neural Network/reliability_enn.py` | Optional answer-level classifier interface | Requires user-supplied labeled rows; not part of default active run |
| Evaluation | `Neural Network/evaluate.py`, `run_experiment.py` | TruthfulQA likelihood and local QA experiments | Local QA file has ten questions |

## Archived empirical path

`results/enn_upgrade/technical_report.md` and `results/audited_benchmark/` report an upgraded answer-correctness ENN (MC Dropout, deep ensemble, calibration, and threshold regimes). The source files that produced those results are absent from the active checkout. The paper treats these results as archived project evidence, not as a fully reproducible execution of the active source tree.

## Mathematical boundary

The active ENN loss is next-token cross-entropy on C4. The archived upgraded ENN uses answer-level correctness labels. Therefore, active token uncertainty and archived answer reliability must not be conflated.
