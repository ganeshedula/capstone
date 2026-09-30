# How to Run the DoLa + ENN Hallucination-Reduction Pipeline

## Prerequisites

- Python 3.10
- Virtual environment already set up at `llm_hallucination_env/`
- TinyLlama model at `models/TinyLlama-1.1B-Chat-v1.0/`
- Apple Silicon Mac with MPS (Metal Performance Shaders) support
- Internet connection (C4 data is streamed on demand from HuggingFace — see note below)

> **C4 dataset note**: The files in `dataset/c4/en.noblocklist/` are Git LFS pointers,
> not real data. The pipeline automatically detects this and streams the required samples
> directly from HuggingFace (`allenai/c4`) — no manual download needed.

---

## Step 0 — Always activate the virtual environment first

Run all commands from the repository root:

```bash
cd /Users/ganesh/B-tech/capstone_Nerual
source llm_hallucination_env/bin/activate
```

You should see `(llm_hallucination_env)` in your terminal prompt before running anything.

> **Common mistake**: Running `python main.py` from inside a subdirectory like `Neural Network/`
> or `Model_load/` will fail. Always run from the project root.

---

## Recommended Workflow (Step-by-Step)

### 1. Quick Sanity Check (~2-3 mins)

Verify that the base environment and TruthfulQA evaluation load and execute cleanly on 20 questions:

```bash
# Baseline sanity check
python main.py --mode eval --no-dola --no-enn --n-questions 20

# DoLa sanity check (verifies contrastive logit subtraction)
python main.py --mode eval --no-baseline --no-enn --n-questions 20 --alpha 1.0
```

---

### 2. DoLa Hyperparameter Sweep (~30-60 mins)

Evaluate DoLa across different contrastive alphas ($\alpha \in \{0.1, 0.25, 0.5, 1.0\}$) and fixed premature layers to determine the best configuration on your validation questions:

```bash
python main.py --mode dola-sweep --n-questions 100
```

---

### 3. Extract C4 Features (~10-15 mins)

Extracts post-RMSNorm mature and premature hidden representations from TinyLlama on C4 text sequences. Splits into 90% train and 10% validation cache (v2 format).

```bash
python main.py --mode extract --n-c4 600 --alpha 1.0 --force-extract
```

- Output: `Neural Network/features_cache.npz`
- Use `--force-extract` to overwrite any previous or incomplete cache files.

---

### 4. Train the Epistemic Neural Network (ENN) (~30-45 mins)

Trains the Epinet with epistemic index $z \sim \mathcal{N}(0, I_6)$ and prior network on the extracted features. Loss is computed on $\text{CrossEntropy}(\text{DoLa} + \text{ENN})$.

```bash
python main.py --mode train --epochs 5 --batch-size 64 --lr 0.0001 --n-z-samples 4 --alpha 1.0
```

- Output: `Neural Network/enn_checkpoint.pkl`
- Requires `Neural Network/features_cache.npz`.

---

### 5. Evaluate All Three Methods (~15-20 mins)

Evaluates Baseline (TinyLlama), DoLa alone, and DoLa + ENN on TruthfulQA multiple-choice benchmarks (MC1 & MC2):

```bash
python main.py --mode eval --n-questions 100 --alpha 1.0 --enn-weight 0.1
```

---

### 6. Run Ablation Study (~30-45 mins)

Runs the ablation experiments to test ENN weights ($w \in \{0.01, 0.05, 0.1, 0.2, 0.5\}$) and epinet components (with vs. without prior network):

```bash
python main.py --mode ablation --n-questions 100 --alpha 1.0
```

---

### 7. Final Test Evaluation on Unseen Questions

After tuning hyperparameters on questions 0–99, test on held-out questions using `--start-idx`:

```bash
python main.py --mode eval --n-questions 200 --start-idx 100 --alpha 1.0 --enn-weight 0.1
```

---

### 8. Before vs After Training Comparison (~45-90 mins)

Evaluate custom questions **before and after** ENN training to visualise how training changes model performance. This mode runs the full pipeline automatically:

1. Loads your questions from a JSON file.
2. Evaluates DoLa-only scoring (before ENN training).
3. Extracts C4 features + trains the ENN.
4. Evaluates DoLa+ENN scoring (after ENN training).
5. Generates comparison tables, metrics, and graphs.

```bash
# Default (uses eval_questions.json with 10 sample questions)
python main.py --mode before-after

# Fast test run (fewer C4 samples, fewer epochs)
python main.py --mode before-after --n-c4 50 --epochs 2

# Custom questions file and output directory
python main.py --mode before-after --eval-questions my_questions.json --output-dir output/my_experiment
```

#### Input Format (`eval_questions.json`)

```json
[
  {
    "question": "What is the capital of France?",
    "answer": "Paris",
    "distractors": ["London", "Berlin", "Madrid"]
  },
  {
    "question": "What planet is closest to the Sun?",
    "answer": "Mercury",
    "distractors": ["Venus", "Earth", "Mars"]
  }
]
```

- `distractors` are optional. When provided, accuracy is measured via MC-style ranking (same method as TruthfulQA).
- You can add as many questions as needed. Replace `eval_questions.json` with your own.

#### Output

| File | Description |
|------|-------------|
| `output/before_after/results.json` | Full per-question results, aggregate metrics, training losses |
| `output/before_after/accuracy_comparison.png` | Before vs After MC accuracy bar chart |
| `output/before_after/log_likelihood_comparison.png` | Per-question log-likelihood grouped bars |
| `output/before_after/confidence_comparison.png` | Per-question confidence grouped bars |
| `output/before_after/entropy_comparison.png` | Per-question entropy grouped bars |
| `output/before_after/uncertainty_breakdown.png` | ENN epistemic/aleatoric/predictive decomposition |
| `output/before_after/correctness_heatmap.png` | Per-question correct/incorrect grid |
| `output/before_after/training_loss.png` | ENN train/val loss curve |
| `output/before_after/summary_dashboard.png` | Multi-panel summary of all key metrics |

---

### 9. C4 Token-Level Evaluation (~15-30 mins)

Evaluates perplexity, token accuracy, log-likelihood, and ENN uncertainty calibration on held-out C4 text sequences (not seen during ENN training):

```bash
python main.py --mode c4-eval --n-c4-eval 100
```

---

### 10. Interactive Inference (Ask)

Ask custom questions interactively to compare the baseline, DoLa, and DoLa + ENN outputs in real-time.

```bash
python main.py --mode ask
```

---

## One-Shot Full Pipeline

If you want to run feature extraction, ENN training, and evaluation in a single command:

```bash
# Full pipeline with defaults
python main.py

# Fast pipeline test
python main.py --n-c4 50 --n-questions 20 --epochs 2
```

---

## All CLI Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--mode` | string | `all` | Pipeline mode: `all`, `extract`, `train`, `eval`, `ask`, `dola-sweep`, `ablation`, `before-after`, `c4-eval` |
| `--n-c4` | int | `600` | Number of C4 samples to stream for feature extraction |
| `--n-questions` | int | `100` | Number of TruthfulQA questions to evaluate |
| `--start-idx` | int | `0` | Starting index in TruthfulQA dataset (for val/test split) |
| `--epochs` | int | `5` | ENN training epochs |
| `--batch-size` | int | `64` | ENN training mini-batch size |
| `--lr` | float | `0.0001` | ENN Adam optimizer learning rate |
| `--n-z-samples` | int | `4` | Number of epistemic index $z$ samples per step |
| `--alpha` | float | `1.0` | DoLa contrastive scaling factor $\alpha$ |
| `--enn-weight` | float | `0.1` | Logit blending weight for ENN: $\text{logits}_{\text{dola}} + w \cdot \text{logits}_{\text{enn}}$ |
| `--force-extract` | flag | `False` | Overwrite existing feature cache |
| `--no-baseline` | flag | `False` | Skip baseline model evaluation |
| `--no-dola` | flag | `False` | Skip DoLa-only evaluation |
| `--no-enn` | flag | `False` | Skip DoLa + ENN evaluation |
| `--eval-questions` | string | `eval_questions.json` | Path to JSON file with evaluation questions (for `before-after` mode) |
| `--output-dir` | string | `output/before_after` | Output directory for results and graphs (for `before-after` mode) |
| `--n-c4-eval` | int | `100` | Number of held-out C4 samples for token-level eval |
| `--c4-eval-output` | string | `output/c4_eval` | Directory for C4 evaluation results and graphs |
| `--enn-hidden-dim` | int | `512` | ENN MLP hidden width |
| `--epistemic-index-dim`| int | `6` | Epistemic index width |
| `--debug` | flag | `False` | Print model and tensor diagnostics |

---

## Cached Artifacts

| File | Created by | Used by | Description |
|------|-----------|---------|-------------|
| `Neural Network/features_cache.npz` | `--mode extract` | `--mode train` | Contains post-RMSNorm mature ($h_M$) and premature ($h_P$) hidden states, target tokens, DoLa logits, layer IDs, and train/val split (v2 format). |
| `Neural Network/enn_checkpoint.pkl` | `--mode train` | `--mode eval`, `--mode ablation`, `--mode before-after` | Serialized JAX Epinet trainable weights and hyperparameter config. |
| `output/before_after/results.json` | `--mode before-after` | — | Full per-question before/after results, aggregate metrics, training losses, and CLI args for reproducibility. |
| `output/before_after/*.png` | `--mode before-after` | — | 8 comparison graphs (accuracy, log-likelihood, confidence, entropy, uncertainty, heatmap, training loss, dashboard). |
| `output/c4_eval/c4_eval_results.json` | `--mode c4-eval` | — | Perplexity and evaluation metrics on C4 validation texts. |
| `output/c4_eval/*.png` | `--mode c4-eval` | — | C4 evaluation graphs for token-level log-likelihood, accuracy, and uncertainty. |

> **Cache Reset**: If you modify the layer extraction logic or experience corrupt cache files, delete both artifacts to start fresh:
> ```bash
> rm -f "Neural Network/features_cache.npz" "Neural Network/enn_checkpoint.pkl"
> ```

---

## Expected Output

### `--mode eval` (Summary Table)

At the conclusion of `--mode eval`, results are printed in a structured table:

```
================================================================================
EXPERIMENT SUMMARY
================================================================================
Model                           MC1 (Acc)    MC2 (Acc)    Time (s)
--------------------------------------------------------------------------------
Baseline (TinyLlama)               0.2700       0.4100        45.2
DoLa only                          0.3100       0.4400        52.1
DoLa + ENN                         0.3300       0.4650        58.3
================================================================================
```

- **MC1**: Single-answer accuracy (identifying the single best truthful answer among choices).
- **MC2**: Multi-answer accuracy (normalized probability mass assigned to all true answers vs. false ones).

### `--mode before-after` (Comparison Table)

At the conclusion of `--mode before-after`, a per-question comparison table is printed:

```
==========================================================================================
  BEFORE vs AFTER TRAINING — PER-QUESTION RESULTS
==========================================================================================
  Q#   Question                            Before        After       Change
  ---- ----------------------------------- ------------ ------------ ----------
  Q1   What is the capital of France?..    ✓ Correct    ✓ Correct
  Q2   What planet is closest to the Su..  ✗ Wrong      ✓ Correct    IMPROVED
  Q3   What is the chemical symbol for ..  ✓ Correct    ✓ Correct
  ...

==========================================================================================
  AGGREGATE METRICS
==========================================================================================
  MC Accuracy (Before) : 40.0%  (4/10)
  MC Accuracy (After)  : 60.0%  (6/10)
  Accuracy Change      : +20.0%

  Metric                       Before        After       Change
  ---------------------------- ------------ ------------ ----------
  Mean Log-Likelihood               -3.4521      -2.8934    +0.5587
  Mean Confidence                    0.0318       0.0556    +0.0238
  Mean Entropy                       8.2341       7.5612    -0.6729
==========================================================================================
```

Graphs are saved to `output/before_after/` including a summary dashboard.

---

## Troubleshooting

1. **`No such file or directory: main.py`**
   - Ensure you run from the repo root: `cd /Users/ganesh/B-tech/capstone_Nerual`
2. **`[ENN] No training samples provided` or cache incompatibility**
   - Stale or empty cache file. Delete and re-extract:
     ```bash
     rm -f "Neural Network/features_cache.npz"
     python main.py --mode extract --force-extract
     ```
3. **`ENN checkpoint not found`**
   - Run `--mode train` before running `--mode eval`.
4. **`MPS / CUDA out of memory`**
   - If running on lower memory configurations, decrease `--batch-size 32` or `--batch-size 16`.
5. **JAX / SciPy / NumPy binary compatibility**
   - Virtual environment is configured with `numpy==1.26.4` and `scipy<1.13`. Do not upgrade NumPy to 2.x in this environment.
6. **`Evaluation questions file not found`**
   - Create `eval_questions.json` in the project root or specify the path with `--eval-questions path/to/file.json`.
7. **Before-after graphs not generated**
   - Ensure `matplotlib` is installed: `pip install matplotlib`.
   - Check `output/before_after/` for the generated PNG files.
