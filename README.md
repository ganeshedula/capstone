# DoLa + ENN Reliability Pipeline

Research code for TinyLlama inference with **DoLa (Decoding by Contrasting Layers)** and a PyTorch **Epistemic Neural Network (Epinet)**. The current repository also includes candidate sampling, majority-vote Self-Consistency, and ENN-confidence-weighted voting.

> This is an experimental project. Improvements in accuracy, reliability, or calibration are not assumed; use the evaluation commands and report measured results.

## Contents

- [What is implemented](#what-is-implemented)
- [Download and install](#download-and-install)
- [Model, data, and checkpoint files](#model-data-and-checkpoint-files)
- [Run the main pipeline](#run-the-main-pipeline)
- [Run the Self-Consistency experiments](#run-the-self-consistency-experiments)
- [Evaluation modes and commands](#evaluation-modes-and-commands)
- [Configuration and outputs](#configuration-and-outputs)
- [Project layout](#project-layout)
- [Troubleshooting](#troubleshooting)

## What is implemented

### Main inference and training path

1. `main.py` loads the TinyLlama tokenizer and model once.
2. `Neural Network/dola.py` runs the transformer and collects hidden states with shape `(batch, tokens, hidden_size)`. For each token it selects a premature layer using Jensen–Shannon divergence, then computes contrastive logits:

   ```text
   DoLa logits = mature logits - alpha * premature logits
   ```

3. `Neural Network/data_prep.py` extracts mature and premature features from C4, along with next-token labels. Features have shape `(examples, hidden_size)`; mature and premature features are concatenated for the ENN. Documents, rather than individual tokens, are assigned to train and validation splits.
4. `Neural Network/enn_torch.py` trains the PyTorch Epinet on C4 next-token prediction. At inference it can return sampled logits and predictive, aleatoric, and epistemic uncertainty values.
5. `Neural Network/evaluate.py` evaluates the baseline, DoLa, and DoLa + ENN using TruthfulQA multiple-choice MC1 and MC2 scores.

The older `Neural Network/enn_jax.py` and `Neural Network/enn_checkpoint.pkl` are legacy JAX artifacts. The current `main.py` uses `enn_torch.py` and expects a PyTorch checkpoint at `checkpoints/enn_best.pt`.

### Self-Consistency path

`run_experiment.py` reuses the model, DoLa decoder, and PyTorch ENN. It samples candidates with temperature/top-p settings and supports:

- `dola`: one DoLa candidate
- `dola_self_consistency`: DoLa candidates with plurality voting
- `dola_enn_self_consistency`: DoLa candidates scored by ENN predictive uncertainty, followed by confidence-weighted voting
- `all`: the six-method ablation and Self-Consistency sample-count comparisons

The ENN confidence weight is defined as `1 - predictive_entropy / log(vocabulary_size)`. It is an entropy-derived score, **not** a calibrated probability of correctness. The normalized weighted-vote result is the winning answer’s fraction of total vote weight, not a calibrated answer probability. Calibration metrics compare these scores with observed exact-match correctness.

The ENN was trained with DoLa-derived features. In the `Base + ENN` ablation, Base generates the candidate while the existing DoLa-compatible ENN features are used to estimate its uncertainty.

## Download and install

### 1. Clone this repository

```bash
git clone https://github.com/ganeshedula/capstone.git
cd capstone
```

Run project commands from this repository root. The `Neural Network/` directory name contains a space; the scripts handle its import path, so do not run `main.py` from inside that directory.

### 2. Create a Python environment

Python 3.10 is the version used for development. Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate       # macOS / Linux
# .venv\Scripts\activate       # Windows PowerShell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements install PyTorch, Transformers, Datasets, NumPy, PyYAML, Matplotlib, and tqdm. On Apple Silicon, install a PyTorch build with MPS support if you want Metal acceleration. On NVIDIA systems, install the PyTorch build matching your CUDA runtime. `main.py` automatically chooses CUDA, then MPS, then CPU; `run_experiment.py` uses the same device order and also has a `--device` override.

### 3. Download TinyLlama

The model is not included in Git. The code expects the model files under:

```text
models/TinyLlama-1.1B-Chat-v1.0/
```

Download it with the Hugging Face CLI (installed with `huggingface_hub`):

```bash
python -m pip install -U huggingface_hub
hf download TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --local-dir models/TinyLlama-1.1B-Chat-v1.0
```

The model repository is [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0); the CLI download syntax is documented in the [Hugging Face Hub CLI guide](https://huggingface.co/docs/huggingface_hub/guides/cli). The weights require several gigabytes of disk space and additional RAM/VRAM for inference. The program falls back to CPU when no accelerator is available, but generation can be very slow.

### 4. Verify the environment

```bash
python Verification/verify_pytorch.py
```

Optional checks:

```bash
python Verification/verify_datasets.py
python Verification/verify_jax.py          # legacy JAX path only
python Verification/verify_transformers.py  # downloads the small GPT-2 tokenizer
python Verification/verify_all.py           # runs every check above
```

`verify_all.py` also checks optional libraries such as JAX, SciPy, scikit-learn, and SentencePiece that are not all required by the current PyTorch pipeline. It may download GPT-2 and access Hugging Face datasets.

## Model, data, and checkpoint files

| Asset | How it is obtained | Needed for |
|---|---|---|
| TinyLlama model | Download into `models/TinyLlama-1.1B-Chat-v1.0/` as above | All model inference and training commands |
| `eval_questions.json` | Included in the repository; can be replaced with your own compatible JSON | `before-after` and `run_experiment.py` |
| C4 | Loaded from local C4 files when available; otherwise streamed from Hugging Face (`allenai/c4`, English train split) | ENN feature extraction/training and C4 evaluation |
| TruthfulQA | Loaded by the `datasets` package when evaluation runs | `eval` and `dola-sweep` |
| `Neural Network/features_cache.npz` | Included cache where available; otherwise generated by feature extraction | ENN training |
| `checkpoints/enn_best.pt` | Created by `python main.py --mode train` | ENN evaluation, interactive ENN inference, and confidence-weighted experiments |

The `models/`, `checkpoints/`, `dataset/`, `output/`, and `results/` directories are excluded by `.gitignore` (the feature cache and legacy JAX checkpoint currently tracked in `Neural Network/` are separate files). A fresh clone therefore needs the model download and, if you want ENN runs, a PyTorch ENN training run. The committed `enn_checkpoint.pkl` is not a substitute for `checkpoints/enn_best.pt`.

C4 does not need a manual download for the normal pipeline. If the expected local C4 shard directory is missing or contains Git LFS pointer files, feature extraction falls back to streaming from Hugging Face. Network access is required in that case. `c4-eval` also streams its held-out C4 examples.

## Run the main pipeline

### Full default run

```bash
python main.py
```

This runs feature extraction, ENN training, and TruthfulQA evaluation in order. Current defaults come from `config.yaml`: 20 C4 texts, maximum feature length 128, 2 ENN epochs, batch size 1, 3 epistemic samples, DoLa `alpha=0.05`, 10 TruthfulQA questions, and ENN logit weight 0.0. For useful scientific comparisons, select enough data and hardware for the experiment rather than interpreting the smoke-test defaults as benchmark results.

### Run one stage at a time

```bash
# Extract C4 features and write Neural Network/features_cache.npz
python main.py --mode extract --n-c4 20

# Train the PyTorch ENN; writes checkpoints/enn_best.pt
python main.py --mode train --n-c4 20 --epochs 2 --batch-size 1

# Evaluate Base, DoLa, and DoLa + ENN on TruthfulQA MC1/MC2
python main.py --mode eval --n-questions 10

# Evaluate a later range of TruthfulQA examples
python main.py --mode eval --n-questions 100 --start-idx 100
```

`--mode train` reuses `Neural Network/features_cache.npz` only when the cache version and extraction settings match. `--mode extract` currently forces feature recomputation each time; the `--force-extract` flag is accepted by the CLI but does not change that behavior.

The current configured `--enn-weight` default is `0.0`, so the ENN logits do not affect the combined DoLa logits by default. Use a positive value to test logit blending, for example `--enn-weight 0.1`; treat this as an experimental parameter, not evidence of improvement.

## Run the Self-Consistency experiments

First create the current PyTorch ENN checkpoint if it is absent:

```bash
python main.py --mode train
```

Run all methods and the default candidate counts (N=1,3,5,10):

```bash
python run_experiment.py
```

Run just one strategy:

```bash
python run_experiment.py --strategy dola
python run_experiment.py --strategy dola_self_consistency
python run_experiment.py --strategy dola_enn_self_consistency
```

Adjust sample counts, generation controls, reproducibility seed, and output directory:

```bash
python run_experiment.py \
  --samples 1 3 5 10 \
  --temperature 0.7 \
  --top-p 0.9 \
  --max-new-tokens 16 \
  --seed 42 \
  --alpha 0.05 \
  --output-dir results
```

`--samples` accepts any positive integer values; the default comparison is 1, 3, 5, and 10. Per-candidate random seeds are derived deterministically from `--seed`, the question index, and candidate index so runs can be reproduced while candidates remain stochastic. Candidate generation uses small batches to limit KV-cache memory and retries empty generations up to three times. A quick integration run can be limited to the first question and one candidate:

```bash
python run_experiment.py --limit 1 --samples 1 --max-new-tokens 8 --output-dir results/smoke
```

The all-strategies experiment can be compute-intensive, particularly on CPU: it generates Base and DoLa candidate pools and evaluates candidates with the ENN. The run estimates exact-match accuracy against `eval_questions.json`; it is a small local dataset, not a substitute for a larger held-out benchmark. Ground-truth labels are used only after inference for evaluation.

### Interactive semantic verification

Activate the project virtual environment, then start the interactive comparison mode:

```bash
source llm_hallucination_env/bin/activate
python main.py --mode ask --sc-samples 5 --max-new-tokens 48
```

Enter a question at each prompt; submit a blank prompt to exit. The default output compares Base TinyLlama, DoLa, Self-Consistency, DoLa + ENN, and the final selection. Add `--verbose` to display candidate answers and diagnostics. `--quiet` hides startup details. `--use-rag` enables optional Wikipedia evidence, and `--max-new-tokens` controls answer generation. `results.csv` and `comparison_results.csv` are updated for each question; no accuracy claim is made without labeled evaluation data.

The existing `checkpoints/enn_best.pt` is a token-prediction Epinet, not an answer-correctness classifier. To train the separate reliability ENN, provide a JSONL file with labeled rows such as `{"features":{"mean_token_confidence":0.7,"factuality_score":0.8},"reliable":1}` and run:

```bash
llm_hallucination_env/bin/python train_reliability_enn.py --data reliability_labels.jsonl
```

Use question-disjoint labeled data for calibration. `evaluate.py` requires annotated `ground_truth` and `hallucination_label` columns. `ablation.py` summarizes method-wise labeled predictions and requires a `method` column. Neither script treats the ten-answer demo file as a hallucination benchmark.

### Self-Consistency output files

```text
results/
├── ablation_results.csv       # accuracy, reliability, risk/coverage, time
├── calibration_results.csv    # accuracy, ECE, Brier, binary NLL, confidence means
├── confidence_analysis.csv    # detailed aggregate reliability counts and metrics
├── predictions.json           # candidates, votes, uncertainties, error analysis
└── plots/
    ├── accuracy_comparison.png
    ├── confidence_distribution.png
    ├── calibration_curve.png
    └── risk_coverage.png
```

`--enn-weight` is accepted by `run_experiment.py` for CLI compatibility, but does not change that runner's candidate-confidence weighting. The runner uses ENN predictive entropy to assign candidate weights; `--enn-weight` applies to the logit-blending path in `main.py`.

## Evaluation modes and commands

All `main.py` modes load the local TinyLlama model first. From the repository root:

| Command | What it runs |
|---|---|
| `python main.py --mode extract` | Extract/recompute C4 DoLa features and save the cache |
| `python main.py --mode train` | Train the PyTorch ENN and write `checkpoints/enn_best.pt` and `results/training_history.json` |
| `python main.py --mode eval` | Compare baseline, DoLa, and DoLa + ENN using TruthfulQA MC1/MC2 |
| `python main.py --mode eval --no-dola --no-enn` | Baseline only |
| `python main.py --mode eval --no-baseline --no-enn` | DoLa only |
| `python main.py --mode eval --no-baseline --no-dola` | DoLa + ENN only, if a PyTorch checkpoint exists |
| `python main.py --mode dola-sweep` | Sweep configured DoLa alpha values and fixed premature layers on the selected TruthfulQA range |
| `python main.py --mode ablation` | Baseline, DoLa, DoLa + ENN, then several ENN-logit weights; results print to the terminal |
| `python main.py --mode ask` | Interactive Base, DoLa, and (when checkpoint exists) DoLa + ENN generation |
| `python main.py --mode before-after` | Score custom JSON questions, extract C4 features, train ENN, score again, and save JSON/plots |
| `python main.py --mode c4-eval` | Compare token-level DoLa and DoLa + ENN metrics on C4 examples skipped past the configured training count |
| `python main.py` or `python main.py --mode all` | Extract features, train ENN, then evaluate TruthfulQA |

Examples:

```bash
# Baseline and DoLa on 50 TruthfulQA questions
python main.py --mode eval --n-questions 50 --no-enn

# DoLa + ENN with a nonzero ENN logit contribution
python main.py --mode eval --no-baseline --no-dola --enn-weight 0.1

# Tune DoLa on a validation range; do not report its best score as an unbiased test result
python main.py --mode dola-sweep --n-questions 50 --start-idx 0

# Before/after ENN training with your own question file
python main.py --mode before-after \
  --eval-questions eval_questions.json \
  --output-dir output/before_after

# Evaluate held-out C4 samples after skipping the configured training sample count
python main.py --mode c4-eval --n-c4 20 --n-c4-eval 100

# See all main.py command-line flags
python main.py --help
```

The main `ablation` mode is the earlier baseline/DoLa/ENN logit-weight sweep; it is distinct from the six-method Self-Consistency ablation in `run_experiment.py`.

## Configuration and outputs

Edit `config.yaml` to change defaults used by `main.py`:

| Section | Settings |
|---|---|
| `model` | Local model directory (the current `main.py` constant also points to `models/TinyLlama-1.1B-Chat-v1.0`) |
| `training` | C4 sample count, feature length, batch size, epochs, learning rate, ENN width and epistemic sample count |
| `dola` | Default contrastive `alpha` and mature layer setting |
| `evaluation` | TruthfulQA question count and ENN logit blending weight |
| `paths` | Intended result and checkpoint directory names |

Some paths are constants in `main.py` and `Neural Network/data_prep.py`; changing a path in the YAML file does not redirect every module.

Common generated artifacts:

- `Neural Network/features_cache.npz`: C4 token features and document IDs for a document-level train/validation split.
- `checkpoints/enn_best.pt`, `checkpoints/enn_epoch_*.pt`: current PyTorch ENN checkpoints.
- `results/training_history.json`: ENN training and validation loss history.
- `output/before_after/`: custom-question comparison JSON and plots.
- `output/c4_eval/`: held-out C4 metrics and plots.
- `results/`: Self-Consistency ablation tables, detailed predictions, and plots.

Model weights, checkpoints, outputs, and results are ignored by Git. Keep backups of any generated checkpoint or result you need to retain.

## Project layout

```text
main.py                         # main training, evaluation, sweep, and interactive CLI
run_experiment.py               # Base/DoLa/ENN/Self-Consistency ablations and calibration
config.yaml                     # main.py defaults
eval_questions.json             # small example labeled evaluation set
requirements.txt                # core Python dependencies
Model_load/load_llama2.py       # active TinyLlama loader and chat prompt formatter
Model_load/load_llama.py        # separate legacy/general Llama loader
Neural Network/dola.py          # DoLa hidden-state, layer-selection, and logits logic
Neural Network/enn_torch.py     # active PyTorch Epinet training and prediction
Neural Network/enn_jax.py       # legacy JAX Epinet implementation
Neural Network/data_prep.py     # C4 loading, feature extraction, cache, train/val split
Neural Network/evaluate.py      # TruthfulQA MC1/MC2 evaluation
Neural Network/inference.py     # autoregressive decoding and candidate sampling
Neural Network/self_consistency.py # answer normalization and voting
Neural Network/before_after_eval.py # custom labeled question evaluation
Neural Network/c4_eval.py       # held-out C4 next-token evaluation
Neural Network/*_plots.py       # evaluation visualizations
Verification/                   # environment and dependency checks
```

## Troubleshooting

- **Model files not found:** confirm that `models/TinyLlama-1.1B-Chat-v1.0/config.json`, tokenizer files, and model weights exist. Re-run the `hf download` command from the repository root.
- **ENN checkpoint missing:** run `python main.py --mode train`. The committed `Neural Network/enn_checkpoint.pkl` is a legacy JAX checkpoint and is not loaded by the current PyTorch path.
- **C4 or TruthfulQA download errors:** confirm internet access and retry. Both datasets are loaded through Hugging Face Datasets; C4 feature preparation can also use valid local shards.
- **Slow CPU inference:** this project loads a 1.1B-parameter model. CPU is supported as a fallback, but generation and layer-contrast evaluation can be very slow. Use supported CUDA or Apple MPS acceleration where available, reduce question/sample counts during development, and keep full evaluation settings for final comparisons.
- **Out-of-memory:** reduce `--batch-size` for ENN training. Self-Consistency candidate generation uses microbatches of two and retries individual candidates after an accelerator out-of-memory error.
- **Stale ENN features:** cache metadata includes feature-cache version, C4 sample count, max length, and validation fraction. Use `python main.py --mode extract` to regenerate features (this mode always recomputes at present), then retrain the ENN.
- **JAX verification fails:** JAX is not required for the active PyTorch pipeline. Use `python Verification/verify_pytorch.py` to check the runtime used by `main.py`.

## References

- [TinyLlama model on Hugging Face](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- [Hugging Face Hub download CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)
- [C4 dataset](https://huggingface.co/datasets/allenai/c4)
- [TruthfulQA dataset](https://huggingface.co/datasets/truthfulqa/truthful_qa)
- DoLa: Chuang et al., [DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models](https://arxiv.org/abs/2309.03883)
