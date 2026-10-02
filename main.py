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
import textwrap
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
                   "alpha": args.alpha, "enn_weight": args.enn_weight,
                   "enn_hidden_dim": args.enn_hidden_dim,
                   "epistemic_index_dim": args.epistemic_index_dim,
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


def cmd_evaluate_dataset(args, model, tokenizer, device):
    """Evaluate answer correctness and hallucination detection only against labels."""
    from inference import generate_answer
    from metrics import binary_metrics
    if not args.dataset:
        raise ValueError("--mode evaluate requires --dataset PATH")
    path = args.dataset if os.path.isabs(args.dataset) else os.path.join(_ROOT, args.dataset)
    with open(path, encoding="utf-8") as stream: dataset = json.load(stream)
    samples = dataset if isinstance(dataset, list) else dataset.get("data", dataset.get("examples", []))
    if not samples: raise ValueError("Dataset must be a non-empty JSON array or contain a data/examples array")
    if any(not row.get("question") or not (row.get("ground_truth") or row.get("answer")) for row in samples):
        raise ValueError("Each dataset row needs question and ground_truth (or answer) fields")
    if any("hallucination_label" not in row for row in samples):
        raise ValueError("Each row must include a labeled hallucination_label (true/false or 1/0)")
    has_enn = os.path.exists(ENN_CHECKPOINT)
    enn = load_enn(ENN_CHECKPOINT, device=device) if has_enn else None
    vh = get_vocab_head_weight(model) if has_enn else None
    labels = {name: [] for name in ("Base TinyLlama", "DoLa", "DoLa + SC", "DoLa + ENN", "Final System")}
    correct = {name: [] for name in labels}; confidence = {name: [] for name in labels}
    hallu_probs = {name: [] for name in labels}
    print(f"\nEvaluating {len(samples)} labeled examples from {path}")
    for row in samples:
        question = row["question"]
        truth = str(row.get("ground_truth", row.get("answer", ""))).strip().casefold()
        prompt = __import__('load_llama2').build_prompt(tokenizer, question)
        ids = tokenizer(prompt, return_tensors="pt").to(device)["input_ids"]
        generated = {
            "Base TinyLlama": generate_answer(model, tokenizer, ids, "base", max_new_tokens=args.max_new_tokens),
            "DoLa": generate_answer(model, tokenizer, ids, "dola", alpha=args.alpha, max_new_tokens=args.max_new_tokens),
        }
        sc_decision, sc_winner, clustering, reports, _ = run_interactive_self_consistency(
            args, model, tokenizer, device, question, ids, args.temperature)
        generated["DoLa + SC"] = sc_winner
        if has_enn:
            generated["DoLa + ENN"] = generate_answer(model, tokenizer, ids, "dola_enn", alpha=args.alpha,
                epinet=enn, vocab_head_weight=vh, n_z_samples=args.n_z_samples,
                enn_weight=args.enn_weight, max_new_tokens=args.max_new_tokens)
        else:
            generated["DoLa + ENN"] = generated["DoLa"]
        generated["Final System"] = sc_winner
        refs = row.get("reference_answers", [truth])
        refs = [str(value).strip().casefold() for value in refs]
        for name, item in generated.items():
            answer = str(item.get("answer", "")).strip().casefold()
            is_correct = int(answer in refs)
            conf = float(item.get("mean_token_confidence", item.get("confidence", 0.0)))
            if name in ("DoLa + SC", "Final System"):
                metric = item.get("metrics", {})
                risk = float(metric.get("hallucination_score", 1.0-conf))
                conf = float(metric.get("reliability_probability", conf))
            else:
                risk = 1.0-conf
            correct[name].append(is_correct); confidence[name].append(conf)
            label = row["hallucination_label"]
            labels[name].append(int(label if isinstance(label, bool) else str(label).strip().casefold() in ("1", "true", "yes", "hallucination")))
            hallu_probs[name].append(risk)
    print("\nMODEL EVALUATION — exact-match answer correctness against references")
    print(f"{'Method':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Hallu det.':>11} {'FPR':>8} {'FNR':>8} {'ECE':>8} {'Brier':>8}")
    print("-" * 108)
    for name in labels:
        answer_metrics = binary_metrics(correct[name], confidence[name])
        detection = binary_metrics(labels[name], hallu_probs[name])
        exact_accuracy = sum(correct[name]) / max(1, len(correct[name]))
        print(f"{name:<20} {exact_accuracy:>9.1%} {answer_metrics['precision']:>9.1%} {answer_metrics['recall']:>9.1%} {answer_metrics['f1']:>9.1%} {detection['hallucination_detection_rate']:>10.1%} {detection['false_positive_rate']:>7.1%} {detection['false_negative_rate']:>7.1%} {answer_metrics['ece']:>7.3f} {answer_metrics['brier_score']:>8.3f}")
    if not has_enn:
        print("ENN status: HEURISTIC / UNCALIBRATED; DoLa + ENN answer row falls back to DoLa (no token ENN checkpoint).")
    print("Accuracy uses exact reference matching. Hallucination detection uses provided labels; Base/DoLa risk is the proxy 1 − mean token confidence.")
    print("Reliability estimates are not reported as accuracy. Calibration metrics use answer confidence against exact-match correctness.")


def cmd_dola_sweep(args, model, tokenizer, device):
    """
    DoLa hyperparameter sweep:  test fixed premature layers and alpha values.
    Uses the first N questions as validation set.
    """
    from dola import DoLaStats

    fixed_layers = [0, 2, 4, 6, 8, 10, 14, 18]
    alphas       = [0.05, 0.1, 0.25, 0.5, 1.0]

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
    aggregate["after_label"] = (
        f"DoLa (ENN weight={args.enn_weight:g})"
        if args.enn_weight == 0
        else f"DoLa + ENN (weight={args.enn_weight:g})"
    )
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


def run_interactive_self_consistency(args, model, tokenizer, device, question, ids, temperature):
    """Generate, cluster, verify and score one candidate pool."""
    from inference import generate_candidate_batch
    from semantic_consistency import answer_embeddings, cluster_answers
    from factuality import verify_answer
    from claim_verifier import score_claims, split_claims
    from answer_selector import score_candidates, decide
    from reliability_enn import predict_reliability
    evidence, retrieval_note = ([], None)
    if args.use_rag:
        from rag import retrieve_evidence
        evidence, retrieval_note = retrieve_evidence(question)
        if retrieval_note: print(f"[RAG] {retrieval_note}")
    samples = generate_candidate_batch(model, tokenizer, ids, "dola", args.sc_samples,
                                      alpha=args.alpha, max_new_tokens=args.max_new_tokens,
                                      temperature=temperature, top_p=args.top_p)
    for index, item in enumerate(samples):
        for attempt in range(2):
            if item.get("answer", "").strip(): break
            item = generate_candidate_batch(model, tokenizer, ids, "dola", 1,
                                            alpha=args.alpha, max_new_tokens=args.max_new_tokens,
                                            temperature=temperature, top_p=args.top_p,
                                            seed=42000 + index * 101 + attempt)[0]
        samples[index] = item
        if not item.get("answer", "").strip():
            item["answer"] = "I could not produce a non-empty answer."
    answers = [item["answer"] for item in samples]
    clustering = cluster_answers(answers, answer_embeddings(model, tokenizer, answers, device),
                                 args.semantic_similarity_threshold)
    sim = np.asarray(clustering["similarity_matrix"], dtype=float)
    claims_by_candidate = [split_claims(answer) for answer in answers]
    flat_claims = [claim for group in claims_by_candidate for claim in group]
    claim_vectors = answer_embeddings(model, tokenizer, flat_claims, device) if flat_claims else np.empty((0, model.config.hidden_size))
    claim_vector_map = {claim: claim_vectors[k] for k, claim in enumerate(flat_claims)}
    def claim_similarity(left, right):
        a, b = claim_vector_map.get(left), claim_vector_map.get(right)
        if a is None or b is None: return 0.0
        return float(np.dot(a, b) / max(1e-12, np.linalg.norm(a)*np.linalg.norm(b)))
    reports = []
    for i, item in enumerate(samples):
        peers = [answers[j] for j in range(len(answers)) if j != i]
        factual = verify_answer(model, tokenizer, question, item["answer"], evidence=evidence,
                                max_new_tokens=96)
        peer_claims = [claim for j, group in enumerate(claims_by_candidate) if j != i for claim in group]
        claim = score_claims(item["answer"], peers,
                             lambda a, b: claim_similarity(a, b))
        # Candidate-level semantic support is measurable without pretending a sentence parser is an NLI model.
        semantic_support = float(np.mean([sim[i, j] for j in range(len(answers)) if j != i])) if len(answers) > 1 else 1.0
        claim["claim_support"] = semantic_support
        entropy_scale = max(1.0, float(np.log(max(2, len(tokenizer)))))
        mean_conf = item.get("mean_token_confidence", item.get("confidence", 0.0))
        mean_entropy = item.get("mean_token_entropy", 0.0)
        metrics = {
            "semantic_consistency": len(clustering["clusters"][clustering["cluster_ids"][i]]) / len(answers),
            "factuality_score": factual["factuality_score"],
            "hallucination_score": factual["hallucination_score"],
            "relevance_score": factual["relevance_score"],
            "evidence_support_score": factual["evidence_support_score"],
            "contradiction_score": max(factual["contradiction_score"], claim["claim_contradiction"]),
            "claim_support": claim["claim_support"],
            "token_confidence": mean_conf,
            "sequence_probability": item.get("normalized_sequence_probability", 0.0),
            "entropy_penalty": min(1.0, mean_entropy / entropy_scale),
            "verifier_parse_ok": factual["verifier_parse_ok"],
        }
        enn_path = os.path.join(_ROOT, "checkpoints", "reliability_enn.pt")
        enn_features = {
            "mean_token_confidence": mean_conf,
            "min_token_confidence": item.get("min_token_confidence", mean_conf),
            "mean_token_entropy": mean_entropy,
            "max_token_entropy": item.get("max_token_entropy", mean_entropy),
            "uncertain_token_ratio": item.get("uncertain_token_ratio", 0.0),
            "mean_topk_probability_mass": float(np.mean(item.get("topk_probability_mass", [0.0]))),
            "sequence_log_probability": item.get("sequence_log_probability", 0.0),
            "normalized_sequence_probability": item.get("normalized_sequence_probability", 0.0),
            "dola_confidence": mean_conf, "dola_layer_disagreement": len(set(x for x in item.get("dola_layers", []) if x is not None))/max(1, len(item.get("dola_layers", []))),
            "semantic_consistency": metrics["semantic_consistency"], "largest_cluster_ratio": clustering["largest_cluster_ratio"],
            "average_pairwise_similarity": clustering["average_pairwise_similarity"],
            "factuality_score": metrics["factuality_score"], "contradiction_score": metrics["contradiction_score"],
            "candidate_disagreement": 1.0 - clustering["largest_cluster_ratio"],
            "answer_length": len(tokenizer(item["answer"]).input_ids), "epistemic_uncertainty": 0.0,
            "claim_support": metrics["claim_support"], "relevance_score": metrics["relevance_score"],
            "evidence_support_score": metrics["evidence_support_score"]}
        if os.path.exists(enn_path):
            reliability = predict_reliability(enn_path, enn_features, device=device)
        else:
            # Explicitly untrained fallback; never calls next-token ENN variance correctness evidence.
            estimate = .50 * metrics["factuality_score"] + .25 * metrics["semantic_consistency"] + .25 * metrics["claim_support"]
            reliability = {"reliability_probability": estimate, "hallucination_probability": 1-estimate,
                           "epistemic_uncertainty": None, "source": "untrained heuristic (not calibrated)"}
        metrics["reliability_probability"] = reliability["reliability_probability"]
        judge_hallucination = (metrics["hallucination_score"]
                               if factual.get("verifier_status") != "FALLBACK" else 0.0)
        metrics["hallucination_score"] = max(judge_hallucination, reliability["hallucination_probability"])
        item.update({"cluster_id": clustering["cluster_ids"][i]+1, "metrics": metrics,
                     "factuality": factual, "claim_diagnostics": claim, "reliability": reliability})
        reports.append(item)
    score_weights = json.loads(args.score_weights) if args.score_weights else None
    penalty_weights = json.loads(args.penalty_weights) if args.penalty_weights else None
    score_candidates(reports, score_weights, penalty_weights)
    decision = decide(reports, args.reliability_threshold, args.consistency_threshold,
                      args.hallucination_threshold)
    ranked = sorted(reports, key=lambda item: item["final_score"], reverse=True)
    for rank, item in enumerate(ranked, 1): item["rank"] = rank
    if args.verbose:
        print("\n## SELF-CONSISTENCY CANDIDATES (VERBOSE)")
        for index, item in enumerate(reports, 1):
            print(f"Candidate {index}: {item['answer']}\n  token confidence={item.get('mean_token_confidence', 0):.3f}; entropy={item.get('mean_token_entropy', 0):.3f}; seq-prob={item.get('normalized_sequence_probability', 0):.3g}; DoLa layers={item.get('dola_layers', [])}; score={item['final_score']:.3f}; verifier={item['factuality'].get('verifier_status')}")
            print(f"  verifier details: {item['factuality'].get('verifier_raw', '')}")
            for claim_row in item["claim_diagnostics"]["claims"]:
                print(f"  Claim: {claim_row['claim']} | support={claim_row['claim_support']:.3f} contradiction={claim_row['claim_contradiction']:.3f}")
    print(f"\nSELF-CONSISTENCY  Candidates: {len(reports)} | clusters: {clustering['cluster_count']} | largest: {max(map(len, clustering['clusters']), default=0)}/{len(reports)} | support: {clustering['largest_cluster_ratio']:.1%} | avg similarity: {clustering['average_pairwise_similarity']:.1%}")
    print("Candidate ranking: " + " | ".join(f"{item['rank']}. Candidate {reports.index(item)+1} — {item['final_score']:.3f}" for item in ranked))
    winner_rank = next((i + 1 for i, item in enumerate(reports) if item is decision.get("selected")), "N/A")
    print(f"Selected candidate: Candidate {winner_rank}")
    if clustering["average_pairwise_similarity"] >= args.semantic_similarity_threshold:
        print("High consensus — candidates are semantically similar.")
    if any(not item["factuality"].get("verifier_parse_ok") for item in reports):
        print("Verifier status: FALLBACK (neutral score used; not treated as hallucination evidence)")
    winner = decision["selected"]
    return decision, winner, clustering, reports, evidence


def append_interactive_result(question, base, dola, sc_answer, enn_answer, winner, decision, clustering):
    import csv
    path = os.path.join(_ROOT, "results.csv")
    fields = ["question", "base_answer", "dola_answer", "sc_answer", "enn_answer",
              "final_answer", "base_confidence", "dola_confidence", "sc_consistency",
              "enn_reliability", "hallucination_probability", "factuality_score",
              "final_score", "decision"]
    row = {"question": question, "base_answer": base["answer"],
           "dola_answer": dola["answer"], "sc_answer": sc_answer,
           "enn_answer": enn_answer, "final_answer": winner.get("answer", ""),
           "base_confidence": base.get("mean_token_confidence", ""),
           "dola_confidence": dola.get("mean_token_confidence", ""),
           "sc_consistency": clustering.get("largest_cluster_ratio", ""),
           "factuality_score": winner.get("metrics", {}).get("factuality_score", ""),
           "enn_reliability": winner.get("metrics", {}).get("reliability_probability", ""),
           "hallucination_probability": winner.get("metrics", {}).get("hallucination_score", ""),
           "final_score": winner.get("final_score", ""), "decision": decision}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    existing, old_fields = [], []
    if os.path.exists(path) and os.path.getsize(path):
        with open(path, newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream); old_fields = reader.fieldnames or []; existing = list(reader)
    all_fields = list(dict.fromkeys(old_fields + fields))
    if old_fields != all_fields:
        with open(path, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=all_fields); writer.writeheader(); writer.writerows(existing)
    with open(path, "a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=all_fields); writer.writerow(row)


def _short(text, limit=250):
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[:limit - 3].rstrip() + "..."


def _pct(value):
    try: return f"{float(value):.1%}"
    except (TypeError, ValueError): return "N/A"


def print_interactive_report(question, base, dola, sc_answer, enn, winner, decision, clustering, reports, enn_status):
    print("\nHALLUCINATION ANALYSIS")
    print(f"Question: {question}\n")
    rows = [("BASE", base["answer"]), ("DOLA", dola["answer"]),
            ("DOLA + SC", sc_answer), ("DOLA + ENN", enn.get("answer", "Not available")),
            ("FINAL", winner.get("answer", ""))]
    print(f"{'Method':<16} Answer")
    print("-" * 96)
    for label, answer in rows: print(f"{label:<16} {_short(answer)}")
    m = winner.get("metrics", {})
    print("\nMETRICS COMPARISON")
    print(f"{'Method':<16} {'Conf.':>8} {'Entropy':>9} {'Consistency':>13} {'Reliability':>12} {'Halluc.':>9}")
    print("-" * 74)
    for label, item, consistency, rel in [
        ("Base", base, None, None), ("DoLa", dola, None, None),
        ("DoLa + SC", winner, clustering["largest_cluster_ratio"], None),
        ("DoLa + ENN", enn or {}, None, m.get("reliability_probability")),
        ("Final", winner, clustering["largest_cluster_ratio"], m.get("reliability_probability"))]:
        mm = item.get("metrics", {})
        conf = item.get("mean_token_confidence", item.get("confidence"))
        entropy = item.get("mean_token_entropy")
        reliability = rel if rel is not None else mm.get("reliability_probability")
        halluc = (1 - reliability) if reliability is not None else mm.get("hallucination_score")
        print(f"{label:<16} {_pct(conf):>8} {(_pct(entropy) if entropy is not None else 'N/A'):>9} {(_pct(consistency) if consistency is not None else '—'):>13} {(_pct(reliability) if reliability is not None else 'N/A'):>12} {(_pct(halluc) if halluc is not None else 'N/A'):>9}")
    print("\nMETHOD COMPARISON")
    for label, line in [("Base TinyLlama", base), ("DoLa", dola)]:
        print(f"{label}\nConfidence: {_pct(line.get('mean_token_confidence'))}\nAnswer: {_short(line['answer'])}\n")
    print(f"DoLa + Self-Consistency\nConsistency: {_pct(clustering['largest_cluster_ratio'])}\nSelected candidate: Candidate {winner.get('rank', 'N/A')}\nAnswer: {_short(sc_answer)}\n")
    token_enn_trained = os.path.exists(ENN_CHECKPOINT)
    print(f"DoLa + ENN\nToken ENN status: {'TRAINED' if token_enn_trained else 'NOT AVAILABLE'}\nReliability estimator: {'TRAINED' if enn_status else 'HEURISTIC / UNCALIBRATED'}\nReliability: {_pct(m.get('reliability_probability'))}\nHallucination risk: {_pct(m.get('hallucination_score'))}\nAnswer: {_short(enn.get('answer', 'Not available'))}\n")
    print(f"Final System\nReliability: {_pct(m.get('reliability_probability'))} | Consistency: {_pct(m.get('semantic_consistency'))} | Factuality: {_pct(m.get('factuality_score'))}\nDecision: {decision}\nAnswer: {_short(winner.get('answer'))}")
    print("\nIMPROVEMENT ANALYSIS (metric changes; not accuracy claims)")
    print(f"Base → DoLa confidence: {(dola.get('mean_token_confidence', 0)-base.get('mean_token_confidence', 0))*100:+.2f} percentage points")
    print(f"DoLa → Self-Consistency support: {_pct(clustering['largest_cluster_ratio'])}")
    print(f"ENN reliability estimate: {_pct(m.get('reliability_probability'))}")
    print(f"ENN / Reliability: {_pct(m.get('reliability_probability'))} | epistemic uncertainty: N/A | status: {'TRAINED' if os.path.exists(os.path.join(_ROOT, 'checkpoints', 'reliability_enn.pt')) else 'HEURISTIC / UNCALIBRATED'}")
    print(f"Verifier status: {winner.get('factuality', {}).get('verifier_status', 'FALLBACK')}\n")
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║ FINAL RESULT                                                         ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    print(f"║ Question: {_short(question, 60):<60} ║")
    print(f"║ Selected method: {'DoLa + SC + ENN':<52} ║")
    print(f"║ Decision: {decision:<62} ║")
    print(f"║ Confidence: {_pct(winner.get('mean_token_confidence')):<60} ║")
    print(f"║ Semantic consensus: {_pct(clustering['largest_cluster_ratio']):<51} ║")
    print(f"║ Factuality: {_pct(m.get('factuality_score')):<61} ║")
    print(f"║ Reliability: {_pct(m.get('reliability_probability')):<60} ║")
    print(f"║ Hallucination risk: {_pct(m.get('hallucination_score')):<52} ║")
    print("║ Answer:                                                                ║")
    for answer_line in textwrap.wrap(_short(winner.get("answer"), 250), width=68) or [""]:
        print(f"║ {answer_line:<68} ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    append_interactive_result(question, base, dola, sc_answer, enn.get("answer", ""), winner, decision, clustering)
    save_method_comparison(question, base, dola, winner, enn, clustering, decision)


def save_method_comparison(question, base, dola, winner, enn, clustering, decision):
    import csv
    path = os.path.join(_ROOT, "comparison_results.csv")
    fields = ["question", "method", "answer", "confidence", "consistency", "reliability", "hallucination_risk", "decision"]
    final_metrics = winner.get("metrics", {})
    rows = [("Base", base, None), ("DoLa", dola, None), ("DoLa + SC", winner, clustering.get("largest_cluster_ratio")),
            ("DoLa + ENN", enn, None), ("Final", winner, clustering.get("largest_cluster_ratio"))]
    with open(path, "a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if stream.tell() == 0: writer.writeheader()
        for label, item, consistency in rows:
            reliability = final_metrics.get("reliability_probability") if label in ("DoLa + ENN", "Final") else ""
            writer.writerow({"question": question, "method": label, "answer": item.get("answer", ""),
                             "confidence": item.get("mean_token_confidence", ""), "consistency": consistency or "",
                             "reliability": reliability, "hallucination_risk": final_metrics.get("hallucination_score", "") if reliability != "" else "",
                             "decision": decision if label == "Final" else ""})


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
        choices=["all", "extract", "train", "eval", "ask", "evaluate", "dola-sweep", "ablation", "before-after", "c4-eval"],
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
    p.add_argument("--verbose", action="store_true", help="Show all candidate answers and detailed diagnostics")
    p.add_argument("--quiet", action="store_true", help="Keep output to the compact research summary (default)")

    # DoLa
    p.add_argument("--alpha",        type=float, default=dola_config.get("alpha", 1.0), help="DoLa contrastive alpha")

    # ENN inference
    p.add_argument("--enn-weight",   type=float, default=evaluation.get("enn_weight", 0.0),  help="Weight of ENN logits added to DoLa")
    p.add_argument("--sc-samples", type=int, default=5, help="DoLa candidates per interactive self-consistency vote")
    p.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for interactive self-consistency")
    p.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling cutoff for interactive self-consistency")
    p.add_argument("--max-new-tokens", type=int, default=48, help="Maximum answer length")
    p.add_argument("--reliability-threshold", type=float, default=0.55)
    p.add_argument("--consistency-threshold", type=float, default=0.4)
    p.add_argument("--hallucination-threshold", type=float, default=0.55)
    p.add_argument("--semantic-similarity-threshold", type=float, default=0.78)
    p.add_argument("--score-weights", default=None, help="JSON object mapping score names to non-negative weights")
    p.add_argument("--penalty-weights", default=None, help="JSON object: contradiction, hallucination, entropy, unsupported_claim")
    p.add_argument("--use-rag", action="store_true", help="Retrieve Wikipedia evidence for optional factuality checking")

    # Flags
    p.add_argument("--force-extract", action="store_true", help="Re-extract features")
    p.add_argument("--no-baseline",   action="store_true", help="Skip baseline")
    p.add_argument("--no-dola",       action="store_true", help="Skip DoLa-only")
    p.add_argument("--no-enn",        action="store_true", help="Skip DoLa+ENN")

    # Before-after evaluation
    p.add_argument("--eval-questions", type=str, default="eval_questions.json",
                   help="Path to JSON file with evaluation questions")
    p.add_argument("--dataset", type=str, default=None, help="Labeled JSON evaluation dataset")
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
    if not args.quiet:
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
    if not args.quiet:
        print(f"[main] Model loaded — {n_params:.2f}B parameters")

    # ---- Dispatch ----
    if args.mode == "extract":
        cmd_extract(args, model, tokenizer, device)

    elif args.mode == "train":
        cmd_train(args, model, tokenizer, device)

    elif args.mode == "eval":
        cmd_eval(args, model, tokenizer, device)

    elif args.mode == "evaluate":
        cmd_evaluate_dataset(args, model, tokenizer, device)

    elif args.mode == "ask":
        from inference import generate_answer
        if args.sc_samples < 1 or args.max_new_tokens < 1:
            raise ValueError("--sc-samples and --max-new-tokens must be positive")
        print(f"Interactive TinyLlama comparison | device={device} | reliability estimator={'TRAINED' if os.path.exists(os.path.join(_ROOT, 'checkpoints', 'reliability_enn.pt')) else 'HEURISTIC'}")
        print("Enter a question (blank line exits).")
        while True:
            question = input("\nQuestion: ").strip()
            if not question:
                break
            prompt = __import__('load_llama2').build_prompt(tokenizer, question)
            ids = tokenizer(prompt, return_tensors="pt").to(device)["input_ids"]
            base = generate_answer(model, tokenizer, ids, "base", alpha=args.alpha,
                                   max_new_tokens=args.max_new_tokens)
            dola = generate_answer(model, tokenizer, ids, "dola", alpha=args.alpha,
                                   max_new_tokens=args.max_new_tokens)
            decision, winner, clustering, reports, evidence = run_interactive_self_consistency(
                args, model, tokenizer, device, question, ids, args.temperature)
            enn = {}
            if os.path.exists(ENN_CHECKPOINT):
                net = load_enn(ENN_CHECKPOINT, device=device)
                enn = generate_answer(model, tokenizer, ids, "dola_enn", alpha=args.alpha,
                                      epinet=net, vocab_head_weight=get_vocab_head_weight(model),
                                      n_z_samples=args.n_z_samples, enn_weight=args.enn_weight,
                                      max_new_tokens=args.max_new_tokens)
            winner = decision.get("selected") or max(reports, key=lambda item: item.get("final_score", 0))
            print_interactive_report(question, base, dola, winner.get("answer", ""), enn, winner,
                                     decision["decision"], clustering, reports,
                                     os.path.exists(os.path.join(_ROOT, "checkpoints", "reliability_enn.pt")))

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
