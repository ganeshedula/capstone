"""Run reproducible Base/DoLa/ENN/self-consistency experiments on eval_questions.json."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
NN = ROOT / "Neural Network"
sys.path.insert(0, str(ROOT / "Model_load"))
sys.path.insert(0, str(NN))
from load_llama2 import build_prompt, load_model, load_tokenizer
from enn_torch import load_enn
from inference import generate_candidate_batch
from self_consistency import confidence_weighted_vote, normalize_candidate, self_consistency_vote


def calibration_metrics(rows, bins=10):
    if not rows:
        return {"accuracy": None, "ece": None, "brier": None, "nll": None,
                "mean_confidence_correct": None, "mean_confidence_incorrect": None}
    conf = np.clip(np.asarray([r["confidence"] for r in rows], dtype=float), 1e-8, 1 - 1e-8)
    correct = np.asarray([r["correct"] for r in rows], dtype=float)
    ece = 0.0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        hi = lo + 1 / bins
        mask = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(conf[mask].mean() - correct[mask].mean())
    return {"accuracy": float(correct.mean()), "ece": float(ece),
            "brier": float(np.mean((conf - correct) ** 2)),
            "nll": float(-np.mean(correct * np.log(conf) + (1 - correct) * np.log(1 - conf))),
            "mean_confidence_correct": float(conf[correct == 1].mean()) if (correct == 1).any() else None,
            "mean_confidence_incorrect": float(conf[correct == 0].mean()) if (correct == 0).any() else None}


def candidate_enn_diagnostics(model, tokenizer, device, question, candidate, epinet,
                              vocab_head, alpha, n_z, seed):
    """Score candidate using existing ENN predictive entropy; confidence is an entropy heuristic."""
    from dola import dola_logits
    from enn_torch import enn_predict
    prompt = build_prompt(tokenizer, question)
    prefix_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    full_text = prompt + candidate
    full_ids = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(device)
    with torch.inference_mode():
        _, _, _, mature, premature = dola_logits(model, full_ids, alpha=alpha)
        mature_np = mature[0].float().cpu().numpy()
        premature_np = model.model.norm(premature)[0].float().cpu().numpy()
        _, epistemic, predictive, aleatoric = enn_predict(
            epinet, epinet, mature_np, premature_np, vocab_head, n_samples=n_z, seed=seed)
    answer_len = max(1, full_ids.shape[1] - prefix_ids.shape[1])
    start = max(0, len(predictive) - answer_len)
    pred = float(np.nanmean(predictive[start:]))
    epi = float(np.nanmean(epistemic[start:]))
    alea = float(np.nanmean(aleatoric[start:]))
    if not np.isfinite(pred):
        raise ValueError("ENN produced non-finite predictive uncertainty")
    normalized_uncertainty = min(1.0, max(0.0, pred / math.log(max(2, len(tokenizer)))))
    return {"uncertainty": pred, "epistemic_uncertainty": epi,
            "aleatoric_uncertainty": alea,
            "confidence": 1.0 - normalized_uncertainty,
            "confidence_definition": "1 - predictive_entropy/log(vocab_size); heuristic, not calibrated correctness probability"}


def exact_correct(answer, gold, distractors):
    normalized = normalize_candidate(answer)
    return bool(normalized and normalized == normalize_candidate(gold))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--questions", default=str(ROOT / "eval_questions.json"))
    p.add_argument("--output-dir", default=str(ROOT / "results"))
    p.add_argument("--samples", type=int, nargs="+", default=[1, 3, 5, 10])
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--enn-weight", type=float, default=0.0)
    p.add_argument("--n-z-samples", type=int, default=3)
    p.add_argument("--limit", type=int, default=None,
                   help="Optional leading subset for a run check; omitted means the full evaluation file")
    p.add_argument("--strategy", choices=["all", "dola", "dola_self_consistency", "dola_enn_self_consistency"],
                   default="all", help="Run all ablations or one of the three requested decision strategies")
    p.add_argument("--reliability-threshold", type=float, default=0.5)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    if not args.samples or any(n < 1 for n in args.samples):
        p.error("--samples values must be positive integers")
    args.samples = list(dict.fromkeys(args.samples))
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32
    with open(args.questions, encoding="utf-8") as f: questions = json.load(f)
    if args.limit is not None:
        if args.limit < 1: p.error("--limit must be positive")
        questions = questions[:args.limit]
    if not questions: raise ValueError("Evaluation dataset is empty")
    model_dir = ROOT / "models" / "TinyLlama-1.1B-Chat-v1.0"
    checkpoint = ROOT / "checkpoints" / "enn_best.pt"
    tokenizer = load_tokenizer(str(model_dir)); model = load_model(str(model_dir), device, dtype); model.eval()
    epinet = load_enn(str(checkpoint), device=device) if checkpoint.exists() else None
    if epinet is None: print("[experiment] ENN checkpoint missing; ENN experiments will be marked unavailable")
    vocab_head = model.lm_head.weight.float().detach().cpu().numpy()
    rows = []
    detailed = []
    max_n = max(args.samples)
    strategy_methods = {
        "dola": ["Base + DoLa"],
        "dola_self_consistency": ["Base + DoLa + Self-Consistency"],
        "dola_enn_self_consistency": ["Base + DoLa + ENN + Self-Consistency"],
    }
    all_methods = ["Base", "Base + DoLa", "Base + ENN", "Base + DoLa + ENN",
                   "Base + DoLa + Self-Consistency", "Base + DoLa + ENN + Self-Consistency"]
    methods_to_run = all_methods if args.strategy == "all" else strategy_methods[args.strategy]
    start_all = time.time()
    for qi, item in enumerate(questions):
        question, gold = str(item["question"]), str(item["answer"])
        prompt = build_prompt(tokenizer, question)
        ids = tokenizer(prompt, return_tensors="pt").to(device)["input_ids"]
        generated = {"base": [], "dola": []}
        generation_confidence = {"base": [], "dola": []}
        generation_times = {"base": [], "dola": []}
        required_decoders = list(dict.fromkeys("dola" if "DoLa" in method else "base" for method in methods_to_run))
        for base_method in required_decoders:
            t_candidate = time.perf_counter()
            batch_results = []
            microbatch = 2  # limits KV-cache memory on CPU and accelerator devices
            for offset in range(0, max_n, microbatch):
                count = min(microbatch, max_n - offset)
                try:
                    chunk = generate_candidate_batch(
                        model, tokenizer, ids, base_method, count,
                        alpha=args.alpha, max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature, top_p=args.top_p,
                        seed=args.seed + qi * 1009 + offset)
                except RuntimeError as exc:
                    if "out of memory" not in str(exc).lower() or count == 1:
                        raise RuntimeError(f"{base_method} generation failed at question {qi + 1}: {exc}") from exc
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    print(f"[experiment] OOM at batch={count}; retrying candidates individually")
                    chunk = []
                    for j in range(count):
                        chunk.extend(generate_candidate_batch(
                            model, tokenizer, ids, base_method, 1,
                            alpha=args.alpha, max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature, top_p=args.top_p,
                            seed=args.seed + qi * 1009 + offset + j))
                batch_results.extend(chunk)
            per_candidate_time = (time.perf_counter() - t_candidate) / max(1, max_n)
            for candidate_index, result in enumerate(batch_results):
                answer = result.get("answer", "").strip()
                attempts = 0
                retry_elapsed = 0.0
                while not answer and attempts < 3:
                    attempts += 1
                    t_retry = time.perf_counter()
                    retry = generate_candidate_batch(
                        model, tokenizer, ids, base_method, 1, alpha=args.alpha,
                        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                        top_p=args.top_p, seed=args.seed + qi * 1009 + candidate_index + attempts * 1000003)[0]
                    retry_elapsed += time.perf_counter() - t_retry
                    answer = retry.get("answer", "").strip()
                    if answer:
                        result = retry
                if not answer:
                    raise RuntimeError(f"Empty {base_method} generation for question {qi + 1}, candidate {candidate_index + 1}, after 3 retries")
                generated[base_method].append(answer)
                generation_confidence[base_method].append(float(result.get("confidence", 0.0)))
                generation_times[base_method].append(per_candidate_time + retry_elapsed)
            if not generated[base_method]:
                raise RuntimeError(f"All {base_method} generations were empty for question {qi + 1}")

        for method in methods_to_run:
            uses_dola = "DoLa" in method
            uses_enn = "ENN" in method
            is_sc = "Self-Consistency" in method
            method_candidates = generated["dola" if uses_dola else "base"]
            sample_counts = args.samples if is_sc else [1]
            for n in sample_counts:
                candidates = method_candidates[:n]
                candidate_diags = []
                enn_elapsed = 0.0
                if uses_enn:
                    if epinet is None:
                        continue
                    for ci, candidate in enumerate(candidates):
                        t_enn = time.perf_counter()
                        candidate_diags.append(candidate_enn_diagnostics(
                            model, tokenizer, device, question, candidate, epinet,
                            vocab_head, args.alpha, args.n_z_samples,
                            args.seed + qi * 1009 + ci))
                        enn_elapsed += time.perf_counter() - t_enn
                if is_sc:
                    decision = (confidence_weighted_vote(candidates, [d["confidence"] for d in candidate_diags])
                                if uses_enn else self_consistency_vote(candidates))
                    answer = decision["final_answer"]
                    confidence = (decision["normalized_confidence"] if uses_enn
                                  else decision["agreement_ratio"])
                else:
                    answer = candidates[0]
                    if uses_enn:
                        confidence = candidate_diags[0]["confidence"]
                    else:
                        confidence = generation_confidence["dola" if uses_dola else "base"][0]
                    decision = {"final_answer": answer}
                if not (0 <= float(confidence) <= 1):
                    raise ValueError(f"Invalid confidence {confidence} for {method}")
                correct = exact_correct(answer, gold, item.get("distractors", []))
                uncertainty = float(np.mean([d["uncertainty"] for d in candidate_diags])) if candidate_diags else None
                record = {"question_id": qi + 1, "method": method, "samples": n,
                          "question": question, "gold": gold, "prediction": answer,
                          "correct": bool(correct), "confidence": float(confidence),
                          "confidence_kind": ("ENN entropy heuristic" if uses_enn else "vote agreement" if is_sc else "first generated token probability"),
                          "uncertainty": uncertainty,
                          "reliable": bool(confidence >= args.reliability_threshold),
                          "inference_time_sec": sum(generation_times["dola" if uses_dola else "base"][:n]) + enn_elapsed,
                          "candidates": candidates, "candidate_diagnostics": candidate_diags,
                          "decision": decision}
                detailed.append(record)
        if (qi + 1) % 1 == 0:
            print(f"[experiment] evaluated {qi + 1}/{len(questions)} examples")

    output_dir = Path(args.output_dir); (output_dir / "plots").mkdir(parents=True, exist_ok=True)
    elapsed = time.time() - start_all
    keys = sorted({(r["method"], r["samples"]) for r in detailed})
    analysis_rows = []
    for method, n in keys:
        group = [r for r in detailed if r["method"] == method and r["samples"] == n]
        metrics = calibration_metrics(group)
        reliable = [r for r in group if r["reliable"]]
        counts = {"total": len(group), "correct": sum(r["correct"] for r in group),
                  "incorrect": sum(not r["correct"] for r in group), "reliable": len(reliable),
                  "unreliable": len(group) - len(reliable),
                  "correct_reliable": sum(r["correct"] and r["reliable"] for r in group),
                  "correct_unreliable": sum(r["correct"] and not r["reliable"] for r in group),
                  "incorrect_reliable": sum(not r["correct"] and r["reliable"] for r in group),
                  "incorrect_unreliable": sum(not r["correct"] and not r["reliable"] for r in group)}
        uncertainty = [r["uncertainty"] for r in group if r["uncertainty"] is not None]
        coverage = len(reliable) / max(len(group), 1)
        counts.update({"method": method, "samples": n, **metrics,
                       "reliability_accuracy": (counts["correct_reliable"] + counts["incorrect_unreliable"]) / max(len(group), 1),
                       "coverage": coverage,
                       "selective_accuracy": sum(r["correct"] for r in reliable) / len(reliable) if reliable else None,
                       "risk": sum(not r["correct"] for r in reliable) / len(reliable) if reliable else None,
                       "average_uncertainty": float(np.mean(uncertainty)) if uncertainty else None,
                       "average_confidence": float(np.mean([r["confidence"] for r in group])),
                       "inference_time_sec": float(np.mean([r["inference_time_sec"] for r in group]))})
        analysis_rows.append(counts)
    with open(output_dir / "ablation_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(analysis_rows[0])); writer.writeheader(); writer.writerows(analysis_rows)
    with open(output_dir / "calibration_results.csv", "w", newline="", encoding="utf-8") as f:
        cols = ["method", "samples", "accuracy", "ece", "brier", "nll", "average_confidence", "mean_confidence_correct", "mean_confidence_incorrect"]
        writer = csv.DictWriter(f, fieldnames=cols); writer.writeheader(); writer.writerows({k: r.get(k) for k in cols} for r in analysis_rows)
    with open(output_dir / "confidence_analysis.csv", "w", newline="", encoding="utf-8") as f:
        cols = list(analysis_rows[0]); writer = csv.DictWriter(f, fieldnames=cols); writer.writeheader(); writer.writerows(analysis_rows)
    by_case = {(r["question_id"], r["method"], r["samples"]): r for r in detailed}
    error_analysis = {"dola_wrong_sc_corrected": [], "standard_sc_wrong_enn_low_reliability": [],
                      "enn_high_confidence_incorrect": [], "dola_enn_sc_failed": [],
                      "strong_candidate_disagreement": []}
    for n in args.samples:
        for qi in range(1, len(questions) + 1):
            dola = by_case.get((qi, "Base + DoLa", 1))
            standard = by_case.get((qi, "Base + DoLa + Self-Consistency", n))
            weighted = by_case.get((qi, "Base + DoLa + ENN + Self-Consistency", n))
            if dola and standard and not dola["correct"] and standard["correct"]:
                error_analysis["dola_wrong_sc_corrected"].append({"question_id": qi, "samples": n, "dola": dola["prediction"], "self_consistency": standard["prediction"]})
            if standard and weighted and not standard["correct"] and not weighted["reliable"]:
                error_analysis["standard_sc_wrong_enn_low_reliability"].append({"question_id": qi, "samples": n, "standard_prediction": standard["prediction"], "enn_confidence": weighted["confidence"]})
            if weighted and not weighted["correct"] and weighted["reliable"]:
                error_analysis["enn_high_confidence_incorrect"].append({"question_id": qi, "samples": n, "prediction": weighted["prediction"], "confidence": weighted["confidence"]})
            if weighted and not weighted["correct"]:
                error_analysis["dola_enn_sc_failed"].append({"question_id": qi, "samples": n, "prediction": weighted["prediction"], "gold": weighted["gold"]})
            for group in (standard, weighted):
                if group and group["decision"].get("agreement_ratio", 1.0) <= 0.5:
                    error_analysis["strong_candidate_disagreement"].append({"question_id": qi, "method": group["method"], "samples": n, "agreement_ratio": group["decision"]["agreement_ratio"], "vote_counts": group["decision"].get("vote_counts", group["decision"].get("weighted_scores"))})
    with open(output_dir / "predictions.json", "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "temperature": args.temperature, "top_p": args.top_p,
                   "samples": args.samples, "reliability_threshold": args.reliability_threshold,
                   "confidence_semantics": "ENN confidence is 1-normalized predictive entropy, not a calibrated correctness probability; non-ENN SC confidence is vote agreement.",
                   "elapsed_sec": elapsed, "error_analysis": error_analysis,
                   "results": detailed}, f, indent=2)
    make_plots(analysis_rows, detailed, output_dir / "plots")
    print("\nMethod | Samples | Accuracy | ECE | Brier | Avg Confidence | Inference Time (s)")
    for r in analysis_rows:
        print(f"{r['method']} | {r['samples']} | {r['accuracy']:.3f} | {r['ece']:.3f} | {r['brier']:.3f} | {r['average_confidence']:.3f} | {r['inference_time_sec']:.1f}")
    print(f"Results saved under {output_dir}; elapsed {elapsed:.1f}s")


def make_plots(metrics, predictions, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    names = [f"{r['method']}\nN={r['samples']}" for r in metrics]
    fig, ax = plt.subplots(figsize=(max(9, len(names) * .7), 5)); ax.bar(range(len(metrics)), [r["accuracy"] for r in metrics]); ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=55, ha="right", fontsize=7); ax.set_ylabel("Exact-match accuracy"); fig.tight_layout(); fig.savefig(output / "accuracy_comparison.png", dpi=140); plt.close(fig)
    conf_correct = [r["confidence"] for r in predictions if r["correct"]]
    conf_wrong = [r["confidence"] for r in predictions if not r["correct"]]
    fig, ax = plt.subplots(); ax.hist([conf_correct, conf_wrong], bins=10, range=(0, 1), label=["Correct", "Incorrect"], alpha=.7); ax.set_xlabel("Confidence proxy"); ax.legend(); fig.tight_layout(); fig.savefig(output / "confidence_distribution.png", dpi=140); plt.close(fig)
    rows = [r for r in metrics if r["ece"] is not None]
    fig, ax = plt.subplots(); ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    for r in rows:
        group = [x for x in predictions if x["method"] == r["method"] and x["samples"] == r["samples"]]
        bins = np.linspace(0, 1, 6); xs=[]; ys=[]
        for lo, hi in zip(bins[:-1], bins[1:]):
            b = [x for x in group if lo <= x["confidence"] < hi or hi == 1 and x["confidence"] <= hi]
            if b: xs.append(float(np.mean([x["confidence"] for x in b]))); ys.append(float(np.mean([x["correct"] for x in b])))
        ax.plot(xs, ys, marker="o", label=f"{r['method']} N={r['samples']}")
    ax.set(xlabel="Mean confidence", ylabel="Empirical accuracy", xlim=(0,1), ylim=(0,1)); ax.legend(fontsize=6); fig.tight_layout(); fig.savefig(output / "calibration_curve.png", dpi=140); plt.close(fig)
    fig, ax = plt.subplots()
    for r in rows:
        group = sorted([x for x in predictions if x["method"] == r["method"] and x["samples"] == r["samples"]], key=lambda x: x["confidence"], reverse=True)
        xs=[]; ys=[]
        for k in range(1, len(group)+1): xs.append(k/len(group)); ys.append(1-np.mean([x["correct"] for x in group[:k]]))
        ax.plot(xs, ys, label=f"{r['method']} N={r['samples']}")
    ax.set(xlabel="Coverage", ylabel="Risk (1 - accuracy)", xlim=(0,1), ylim=(0,1)); ax.legend(fontsize=6); fig.tight_layout(); fig.savefig(output / "risk_coverage.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
