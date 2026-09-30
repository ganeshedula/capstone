"""
main.py — DoLa + ENN hallucination-reduction pipeline  (Corrected)

Usage
-----
Full pipeline:            python main.py
Extract C4 features:      python main.py --mode extract
Train ENN:                python main.py --mode train
Evaluate all:             python main.py --mode eval
Baseline only:            python main.py --mode eval --no-dola --no-enn
DoLa only:                python main.py --mode eval --no-baseline --no-enn
DoLa sweep:               python main.py --mode dola-sweep
Ablation study:           python main.py --mode ablation
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_ROOT = os.path.dirname(os.path.abspath(__file__))
_NN   = os.path.join(_ROOT, "Neural Network")
_ML   = os.path.join(_ROOT, "Model_load")

for _p in (_NN, _ML):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------

from load_llama2 import (
    load_model as _load_model,
    load_tokenizer as _load_tokenizer,
)

from data_prep import prepare_enn_features

from enn_torch import train_enn, load_enn

from evaluate import (
    evaluate_baseline,
    evaluate_dola,
    evaluate_dola_enn,
    print_results,
    print_experiment_summary,
)

from before_after_eval import (
    load_eval_questions,
    run_before_after_evaluation,
    compute_aggregate_metrics,
    build_comparison_table,
    print_comparison_table,
    save_results,
)

from before_after_plots import generate_all_plots

from c4_eval import (
    run_c4_evaluation,
    print_c4_results,
    save_c4_results,
)

from c4_eval_plots import generate_c4_plots


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_DIR = os.path.join(_ROOT, "models", "TinyLlama-1.1B-Chat-v1.0")

ENN_CHECKPOINT = os.path.join(_ROOT, "checkpoints", "enn_best.pt")
FEATURES_CACHE = os.path.join(_NN, "features_cache.npz")


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device() -> tuple:
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


# ---------------------------------------------------------------------------
# Helper: extract vocab head weight from model
# ---------------------------------------------------------------------------

def get_vocab_head_weight(model) -> np.ndarray:
    """Extract LM head weight matrix as a numpy array (V, H)."""
    return model.lm_head.weight.float().cpu().detach().numpy()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_extract(args, model, tokenizer, device):
    """Extract C4 features and cache them."""
    (
        train_mat, train_prem, train_labels,
        val_mat, val_prem, val_labels,
    ) = prepare_enn_features(
        model, tokenizer, device,
        n_c4_samples=args.n_c4,
        force_recompute=True,
        cache_path=FEATURES_CACHE,
    )

    print(f"[main] Feature extraction complete.")
    print(f"       Train: {len(train_labels)} samples")
    print(f"       Val:   {len(val_labels)} samples")


def cmd_train(args, model, tokenizer, device):
    """Train the ENN on cached features."""
    (
        train_mat, train_prem, train_labels,
        val_mat, val_prem, val_labels,
    ) = prepare_enn_features(
        model, tokenizer, device,
        n_c4_samples=args.n_c4,
        cache_path=FEATURES_CACHE,
    )

    vh_weight = get_vocab_head_weight(model)

    prior, learnable, train_losses, val_losses = train_enn(
        mature_features=train_mat,
        premature_features=train_prem,
        labels=train_labels,
        vocab_head_weight=vh_weight,
        alpha=args.alpha,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_z_samples=args.n_z_samples,
        checkpoint_path=ENN_CHECKPOINT,
        val_mature=val_mat,
        val_premature=val_prem,
        val_labels=val_labels,
        device=device,
        enn_hidden_dim=args.enn_hidden_dim,
        epistemic_index_dim=args.epistemic_index_dim,
    )

    print("[main] ENN training complete.")
    print(f"       Final train loss: {train_losses[-1]:.4f}")
    if val_losses:
        print(f"       Final val loss:   {val_losses[-1]:.4f}")
    os.makedirs(os.path.join(_ROOT, "results"), exist_ok=True)
    history_path = os.path.join(_ROOT, "results", "training_history.json")
    with open(history_path, "w", encoding="utf-8") as fh:
        json.dump({"epochs": args.epochs, "batch_size": args.batch_size,
                   "learning_rate": args.lr, "num_epistemic_samples": args.n_z_samples,
                   "train_losses": train_losses, "val_losses": val_losses}, fh, indent=2)
    print(f"       Training history: {history_path}")


def cmd_eval(args, model, tokenizer, device):
    """Run TruthfulQA evaluation for selected modes."""
    results = {}
    vh_weight = get_vocab_head_weight(model)

    # ---- Baseline ----
    if not args.no_baseline:
        print("\n[main] Running baseline evaluation...")
        t0 = time.time()
        results["Baseline (TinyLlama)"] = evaluate_baseline(
            model, tokenizer, device,
            n_questions=args.n_questions,
            start_idx=args.start_idx,
        )
        print(f"       Time: {time.time() - t0:.1f}s")

    # ---- DoLa ----
    if not args.no_dola:
        print("\n[main] Running DoLa evaluation...")
        t0 = time.time()
        result = evaluate_dola(
            model, tokenizer, device,
            n_questions=args.n_questions,
            alpha=args.alpha,
            start_idx=args.start_idx,
        )
        results["DoLa"] = result
        print(f"       Time: {time.time() - t0:.1f}s")

    # ---- DoLa + ENN ----
    if not args.no_enn:
        if os.path.exists(ENN_CHECKPOINT):
            print(f"\n[main] Loading ENN checkpoint from {ENN_CHECKPOINT}...")
            prior_params = load_enn(ENN_CHECKPOINT, device=device)
            learnable_params = prior_params

            print("[main] Running DoLa + ENN evaluation...")
            t0 = time.time()
            result = evaluate_dola_enn(
                model, tokenizer, device,
                prior_params=prior_params,
                learnable_params=learnable_params,
                vocab_head_weight_np=vh_weight,
                n_questions=args.n_questions,
                alpha=args.alpha,
                enn_weight=args.enn_weight,
                start_idx=args.start_idx,
            )
            results["DoLa + ENN"] = result
            print(f"       Time: {time.time() - t0:.1f}s")
        else:
            print(
                f"\n[main] ENN checkpoint not found at {ENN_CHECKPOINT}.\n"
                f"       Run: python main.py --mode train"
            )

    # ---- Print results ----
    for label, res in results.items():
        print_results(label, res)

    # ---- Summary table ----
    if len(results) > 1:
        print("\nSummary")
        print(f"{'Model':<25} {'MC1':>8} {'MC2':>8} {'AvgScore':>10}")
        print("-" * 55)
        for label, res in results.items():
            avg = res.get('avg_score', 0.0)
            print(f"{label:<25} {res['mc1']:>8.4f} {res['mc2']:>8.4f} {avg:>10.4f}")
        print()


def cmd_dola_sweep(args, model, tokenizer, device):
    """
    DoLa hyperparameter sweep:  test fixed premature layers and alpha values.
    Uses the first N questions as validation set.
    """
    from dola import DoLaStats

    fixed_layers = [0, 2, 4, 6, 8, 10, 14, 18]
    alphas       = [0.1, 0.25, 0.5, 1.0]

    print("\n" + "=" * 60)
    print("  DoLa HYPERPARAMETER SWEEP")
    print("=" * 60)

    # Baseline reference
    print("\n[sweep] Baseline...")
    baseline = evaluate_baseline(
        model, tokenizer, device,
        n_questions=args.n_questions,
        start_idx=args.start_idx,
    )
    print(f"  Baseline MC1={baseline['mc1']:.4f}  MC2={baseline['mc2']:.4f}")

    # Dynamic layer with different alphas
    print("\n[sweep] Dynamic premature layer selection:")
    print(f"  {'Alpha':>6} {'MC1':>8} {'MC2':>8} {'AvgScore':>10}")
    print("  " + "-" * 36)

    best_mc1 = 0.0
    best_config = None

    for alpha in alphas:
        stats = DoLaStats()
        result = evaluate_dola(
            model, tokenizer, device,
            n_questions=args.n_questions,
            alpha=alpha,
            start_idx=args.start_idx,
            stats=stats,
        )
        print(f"  {alpha:>6.2f} {result['mc1']:>8.4f} {result['mc2']:>8.4f} {result.get('avg_score', 0):>10.4f}")

        if result["mc1"] > best_mc1:
            best_mc1 = result["mc1"]
            best_config = {"type": "dynamic", "alpha": alpha, "result": result}

    # Fixed layers with alpha=1.0
    print(f"\n[sweep] Fixed premature layers (alpha={alphas[-1]}):")
    print(f"  {'Layer':>6} {'MC1':>8} {'MC2':>8} {'AvgScore':>10}")
    print("  " + "-" * 36)

    for layer in fixed_layers:
        result = evaluate_dola(
            model, tokenizer, device,
            n_questions=args.n_questions,
            alpha=1.0,
            fixed_premature_layer=layer,
            start_idx=args.start_idx,
        )
        print(f"  {layer:>6d} {result['mc1']:>8.4f} {result['mc2']:>8.4f} {result.get('avg_score', 0):>10.4f}")

        if result["mc1"] > best_mc1:
            best_mc1 = result["mc1"]
            best_config = {"type": "fixed", "layer": layer, "alpha": 1.0, "result": result}

    # Summary
    print(f"\n  Best config: {best_config}")
    print(f"  Baseline MC1={baseline['mc1']:.4f}")
    print(f"  Best DoLa MC1={best_mc1:.4f}")
    improvement = best_mc1 - baseline["mc1"]
    print(f"  Improvement: {improvement:+.4f}")


def cmd_ablation(args, model, tokenizer, device):
    """
    Run the full ablation study.

    1. Baseline
    2. DoLa only
    3. ENN only  (DoLa logits + ENN, alpha=0 effectively = baseline + ENN)
    4. DoLa + ENN
    """
    vh_weight = get_vocab_head_weight(model)

    print("\n" + "=" * 60)
    print("  ABLATION STUDY")
    print("=" * 60)

    results = {}

    # 1. Baseline
    print("\n[ablation] 1/4  Baseline...")
    results["Baseline"] = evaluate_baseline(
        model, tokenizer, device,
        n_questions=args.n_questions,
        start_idx=args.start_idx,
    )

    # 2. DoLa only
    print("\n[ablation] 2/4  DoLa only...")
    results["DoLa"] = evaluate_dola(
        model, tokenizer, device,
        n_questions=args.n_questions,
        alpha=args.alpha,
        start_idx=args.start_idx,
    )

    # Load ENN if available
    has_enn = os.path.exists(ENN_CHECKPOINT)
    if has_enn:
        prior_params = load_enn(ENN_CHECKPOINT, device=device)
        learnable_params = prior_params

        # 3. DoLa + ENN
        print("\n[ablation] 3/4  DoLa + ENN...")
        results["DoLa + ENN"] = evaluate_dola_enn(
            model, tokenizer, device,
            prior_params=prior_params,
            learnable_params=learnable_params,
            vocab_head_weight_np=vh_weight,
            n_questions=args.n_questions,
            alpha=args.alpha,
            enn_weight=args.enn_weight,
            start_idx=args.start_idx,
        )

        # 4. DoLa + ENN with different weights
        for w in [0.01, 0.05, 0.1, 0.2]:
            if w == args.enn_weight:
                continue
            label = f"DoLa + ENN (w={w})"
            print(f"\n[ablation] Extra: {label}...")
            results[label] = evaluate_dola_enn(
                model, tokenizer, device,
                prior_params=prior_params,
                learnable_params=learnable_params,
                vocab_head_weight_np=vh_weight,
                n_questions=args.n_questions,
                alpha=args.alpha,
                enn_weight=w,
                start_idx=args.start_idx,
            )
    else:
        print(
            "\n[ablation] ENN checkpoint not found — "
            "skipping ENN ablations."
        )

    # Summary table
    print("\n" + "=" * 60)
    print("  ABLATION RESULTS")
    print("=" * 60)
    print(f"  {'Configuration':<30} {'MC1':>8} {'MC2':>8}")
    print("  " + "-" * 48)
    for label, res in results.items():
        print(f"  {label:<30} {res['mc1']:>8.4f} {res['mc2']:>8.4f}")
    print()


def cmd_before_after(args, model, tokenizer, device):
    """
    Before vs After Training evaluation pipeline.

    1. Load user-provided evaluation questions.
    2. Evaluate with DoLa-only (before ENN training).
    3. Train the ENN (extract features + train).
    4. Evaluate with DoLa+ENN (after ENN training).
    5. Generate comparison tables, metrics, and graphs.
    """
    import os as _os

    # ---- Load questions ----
    eval_path = args.eval_questions
    if not _os.path.isabs(eval_path):
        eval_path = _os.path.join(_ROOT, eval_path)

    print(f"\n[before-after] Loading evaluation questions from {eval_path}")
    questions = load_eval_questions(eval_path)
    print(f"[before-after] {len(questions)} questions loaded.")

    output_dir = args.output_dir
    if not _os.path.isabs(output_dir):
        output_dir = _os.path.join(_ROOT, output_dir)
    _os.makedirs(output_dir, exist_ok=True)

    vh_weight = get_vocab_head_weight(model)

    # ==================================================================
    #  PHASE 1 — BEFORE TRAINING  (DoLa-only)
    # ==================================================================
    print("\n" + "=" * 70)
    print("  PHASE 1: BEFORE TRAINING  (DoLa-only scoring)")
    print("=" * 70)

    t0 = time.time()
    before_results = run_before_after_evaluation(
        model, tokenizer, device, questions,
        alpha=args.alpha,
        enn_weight=args.enn_weight,
        phase="before",
    )
    print(f"[before-after] Phase 1 complete — {time.time() - t0:.1f}s")

    # ==================================================================
    #  TRAINING — Feature extraction + ENN training
    # ==================================================================
    print("\n" + "=" * 70)
    print("  TRAINING PHASE: C4 Feature Extraction + ENN Training")
    print("=" * 70)

    print("\n[before-after] Step T.1: Feature extraction...")
    cmd_extract(args, model, tokenizer, device)

    print("\n[before-after] Step T.2: ENN training...")
    # Run training and capture losses
    (
        train_mat, train_prem, train_labels,
        val_mat, val_prem, val_labels,
    ) = prepare_enn_features(
        model, tokenizer, device,
        n_c4_samples=args.n_c4,
        cache_path=FEATURES_CACHE,
    )

    prior_params, learnable_params, train_losses, val_losses = train_enn(
        mature_features=train_mat,
        premature_features=train_prem,
        labels=train_labels,
        vocab_head_weight=vh_weight,
        alpha=args.alpha,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_z_samples=args.n_z_samples,
        checkpoint_path=ENN_CHECKPOINT,
        val_mature=val_mat,
        val_premature=val_prem,
        val_labels=val_labels,
    )
    print(f"[before-after] Training complete — final loss: {train_losses[-1]:.4f}")

    # ==================================================================
    #  PHASE 2 — AFTER TRAINING  (DoLa + ENN)
    # ==================================================================
    print("\n" + "=" * 70)
    print("  PHASE 2: AFTER TRAINING  (DoLa + ENN scoring)")
    print("=" * 70)

    t0 = time.time()
    after_results = run_before_after_evaluation(
        model, tokenizer, device, questions,
        alpha=args.alpha,
        enn_weight=args.enn_weight,
        prior_params=prior_params,
        learnable_params=learnable_params,
        vh_weight_np=vh_weight,
        phase="after",
    )
    print(f"[before-after] Phase 2 complete — {time.time() - t0:.1f}s")

    # ==================================================================
    #  RESULTS
    # ==================================================================
    before_agg = compute_aggregate_metrics(before_results)
    after_agg = compute_aggregate_metrics(after_results)
    comparison = build_comparison_table(before_results, after_results)

    # Print to console
    print_comparison_table(comparison, before_agg, after_agg)

    # Save JSON results
    args_dict = {
        "eval_questions": args.eval_questions,
        "alpha": args.alpha,
        "enn_weight": args.enn_weight,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "n_c4": args.n_c4,
        "n_z_samples": args.n_z_samples,
        "model_dir": MODEL_DIR,
        "enn_checkpoint": ENN_CHECKPOINT,
    }
    save_results(
        before_results, after_results,
        before_agg, after_agg, comparison,
        output_dir,
        train_losses=train_losses,
        val_losses=val_losses,
        args_dict=args_dict,
    )

    # Generate graphs
    saved_plots = generate_all_plots(
        comparison, before_agg, after_agg,
        train_losses=train_losses,
        val_losses=val_losses,
        output_dir=output_dir,
    )

    # Final summary
    print("\n" + "=" * 70)
    print("  BEFORE vs AFTER — PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Results  : {output_dir}/results.json")
    print(f"  Graphs   : {len(saved_plots)} plots in {output_dir}/")
    if before_agg.get('mc_accuracy') is not None:
        print(f"  Accuracy : {before_agg['mc_accuracy']*100:.1f}% → {after_agg['mc_accuracy']*100:.1f}%")
    print("=" * 70 + "\n")


def cmd_c4_eval(args, model, tokenizer, device):
    """
    C4 token-level evaluation: Before (DoLa-only) vs After (DoLa+ENN).

    Uses held-out C4 samples NOT seen during ENN training to evaluate
    perplexity, token accuracy, log-likelihood, and ENN uncertainty
    calibration.
    """
    import os as _os

    vh_weight = get_vocab_head_weight(model)

    # Load ENN checkpoint
    if not _os.path.exists(ENN_CHECKPOINT):
        print(
            f"\n[c4-eval] ENN checkpoint not found at {ENN_CHECKPOINT}.\n"
            f"         Run: python main.py --mode train\n"
            f"         (or: python main.py --mode before-after)\n"
        )
        return

    print(f"\n[c4-eval] Loading ENN checkpoint from {ENN_CHECKPOINT}...")
    prior_params = load_enn(ENN_CHECKPOINT, device=device)
    learnable_params = prior_params

    output_dir = args.c4_eval_output
    if not _os.path.isabs(output_dir):
        output_dir = _os.path.join(_ROOT, output_dir)
    _os.makedirs(output_dir, exist_ok=True)

    # ==================================================================
    #  Run C4 token-level evaluation
    # ==================================================================
    print("\n" + "=" * 70)
    print("  C4 TOKEN-LEVEL EVALUATION")
    print("  DoLa-only (Before) vs DoLa+ENN (After)")
    print("=" * 70)

    t0 = time.time()
    per_text_results, aggregate = run_c4_evaluation(
        model, tokenizer, device,
        n_eval=args.n_c4_eval,
        n_skip=args.n_c4,
        alpha=args.alpha,
        enn_weight=args.enn_weight,
        prior_params=prior_params,
        learnable_params=learnable_params,
        vh_weight_np=vh_weight,
    )
    elapsed = time.time() - t0

    # Print results
    print_c4_results(aggregate)

    # Save results
    args_dict = {
        "n_c4_eval": args.n_c4_eval,
        "n_c4_train": args.n_c4,
        "alpha": args.alpha,
        "enn_weight": args.enn_weight,
        "model_dir": MODEL_DIR,
        "enn_checkpoint": ENN_CHECKPOINT,
    }
    save_c4_results(per_text_results, aggregate, output_dir, args_dict)

    # Generate plots
    saved_plots = generate_c4_plots(aggregate, output_dir)

    # Final summary
    print("\n" + "=" * 70)
    print("  C4 EVALUATION — COMPLETE")
    print("=" * 70)
    print(f"  Time     : {elapsed:.1f}s")
    print(f"  Texts    : {aggregate.get('n_texts', 0)}")
    print(f"  Tokens   : {aggregate.get('n_tokens', 0)}")
    ppl_b = aggregate.get('before_perplexity', 0)
    ppl_a = aggregate.get('after_perplexity')
    if ppl_a is not None:
        print(f"  PPL      : {ppl_b:.2f} → {ppl_a:.2f}")
    else:
        print(f"  PPL      : {ppl_b:.2f}")
    t1_b = aggregate.get('before_top1_accuracy', 0) * 100
    t1_a = aggregate.get('after_top1_accuracy')
    if t1_a is not None:
        print(f"  Top-1 Acc: {t1_b:.1f}% → {t1_a*100:.1f}%")
    else:
        print(f"  Top-1 Acc: {t1_b:.1f}%")
    print(f"  Results  : {output_dir}/c4_eval_results.json")
    print(f"  Graphs   : {len(saved_plots)} plots in {output_dir}/")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    with open(os.path.join(_ROOT, "config.yaml"), "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    training = config.get("training", {})
    evaluation = config.get("evaluation", {})
    dola_config = config.get("dola", {})
    p = argparse.ArgumentParser(
        description="DoLa + ENN hallucination-reduction pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--mode",
        choices=["all", "extract", "train", "eval", "ask", "dola-sweep", "ablation", "before-after", "c4-eval"],
        default="all",
        help="Pipeline stage to run.",
    )

    # Data
    p.add_argument("--n-c4",        type=int,   default=training.get("c4_samples", 20),  help="C4 samples for extraction")
    p.add_argument("--n-questions", type=int,   default=evaluation.get("truthfulqa_questions", 10),  help="TruthfulQA questions")
    p.add_argument("--start-idx",   type=int,   default=0,    help="Start index in TruthfulQA (for val/test split)")

    # ENN training
    p.add_argument("--epochs",       type=int,   default=training.get("epochs", 2), help="ENN training epochs")
    p.add_argument("--batch-size",   type=int,   default=training.get("batch_size", 1), help="ENN mini-batch size")
    p.add_argument("--lr",           type=float, default=training.get("learning_rate", 1e-4), help="ENN learning rate")
    p.add_argument("--n-z-samples",  type=int,   default=training.get("num_epistemic_samples", 3), help="Epistemic z samples per step")
    p.add_argument("--enn-hidden-dim", type=int, default=training.get("enn_hidden_dim", 512), help="ENN MLP hidden width")
    p.add_argument("--epistemic-index-dim", type=int, default=training.get("epistemic_index_dim", 6), help="Epistemic index width")
    p.add_argument("--debug", action="store_true", help="Print model and tensor diagnostics")

    # DoLa
    p.add_argument("--alpha",        type=float, default=dola_config.get("alpha", 1.0), help="DoLa contrastive alpha")

    # ENN inference
    p.add_argument("--enn-weight",   type=float, default=1.0,  help="Weight of ENN logits added to DoLa (1.0 is the paper architecture)")

    # Flags
    p.add_argument("--force-extract", action="store_true", help="Re-extract features")
    p.add_argument("--no-baseline",   action="store_true", help="Skip baseline")
    p.add_argument("--no-dola",       action="store_true", help="Skip DoLa-only")
    p.add_argument("--no-enn",        action="store_true", help="Skip DoLa+ENN")

    # Before-after evaluation
    p.add_argument("--eval-questions", type=str, default="eval_questions.json",
                   help="Path to JSON file with evaluation questions")
    p.add_argument("--output-dir",     type=str, default="output/before_after",
                   help="Directory for before-after results and graphs")

    # C4 token-level evaluation
    p.add_argument("--n-c4-eval",      type=int, default=100,
                   help="Number of held-out C4 samples for token-level eval")
    p.add_argument("--c4-eval-output", type=str, default="output/c4_eval",
                   help="Directory for C4 evaluation results and graphs")

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()

    # ---- Load model ----
    device, dtype = get_device()
    print(f"[main] Device: {device} | dtype: {dtype}")
    print(f"[main] Loading TinyLlama from {MODEL_DIR}...")

    tokenizer = _load_tokenizer(MODEL_DIR)
    model     = _load_model(MODEL_DIR, device, dtype)
    model.eval()

    if args.debug:
        print(f"[debug] model={model.config._name_or_path or MODEL_DIR}")
        print(f"[debug] layers={model.config.num_hidden_layers} hidden={model.config.hidden_size} vocab={model.config.vocab_size}")
        print(f"[debug] frozen_parameters={sum(not p.requires_grad for p in model.parameters())}/{sum(1 for _ in model.parameters())}")

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"[main] Model loaded — {n_params:.2f}B parameters")

    # ---- Dispatch ----
    if args.mode == "extract":
        cmd_extract(args, model, tokenizer, device)

    elif args.mode == "train":
        cmd_train(args, model, tokenizer, device)

    elif args.mode == "eval":
        cmd_eval(args, model, tokenizer, device)

    elif args.mode == "ask":
        from inference import generate_answer
        print("Enter an empty question to exit.")
        while True:
            question = input("\nEnter your question: ").strip()
            if not question:
                break
            prompt = _load_tokenizer and __import__('load_llama2').build_prompt(tokenizer, question)
            ids = tokenizer(prompt, return_tensors="pt").to(device)["input_ids"]
            def show(name, result):
                print(f"\n{name}\nAnswer: {result['answer']}\nConfidence: {result['confidence']*100:.2f}%\nEntropy: {result['entropy']:.4f}")
                if result.get("selected_premature_layer") is not None:
                    print(f"Selected premature layer: {result['selected_premature_layer']}")
                if result.get("uncertainty"):
                    print(f"Epistemic variance: {result['uncertainty']['epistemic_variance']:.6f}")
                print("Top 10 tokens:")
                for token, probability in result["top_tokens"]:
                    print(f"  {token!r:<20} {probability:.4f}")
            base = generate_answer(model, tokenizer, ids, "base", alpha=args.alpha)
            dola = generate_answer(model, tokenizer, ids, "dola", alpha=args.alpha)
            show("BASE TINYLLAMA", base)
            show("DOLA", dola)
            if os.path.exists(ENN_CHECKPOINT):
                net = load_enn(ENN_CHECKPOINT, device=device)
                vh = get_vocab_head_weight(model)
                enn = generate_answer(model, tokenizer, ids, "dola_enn", alpha=args.alpha, epinet=net, vocab_head_weight=vh, n_z_samples=args.n_z_samples)
                show("DOLA + ENN", enn)
                print(f"\nCOMPARISON\nBase: {base['confidence']*100:.2f}%\nDoLa: {dola['confidence']*100:.2f}%\nDoLa + ENN: {enn['confidence']*100:.2f}%")
            else:
                print("\nDOLA + ENN\nNo new ENN checkpoint yet. Run --mode train first.")

    elif args.mode == "dola-sweep":
        cmd_dola_sweep(args, model, tokenizer, device)

    elif args.mode == "ablation":
        cmd_ablation(args, model, tokenizer, device)

    elif args.mode == "before-after":
        cmd_before_after(args, model, tokenizer, device)

    elif args.mode == "c4-eval":
        cmd_c4_eval(args, model, tokenizer, device)

    else:
        # ---- Full pipeline ----
        print("\n[main] === Step 1/3: Feature extraction ===")
        cmd_extract(args, model, tokenizer, device)

        print("\n[main] === Step 2/3: ENN training ===")
        cmd_train(args, model, tokenizer, device)

        print("\n[main] === Step 3/3: TruthfulQA evaluation ===")
        cmd_eval(args, model, tokenizer, device)


if __name__ == "__main__":
    main()
