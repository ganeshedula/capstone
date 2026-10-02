"""Evaluate saved predictions against answer and hallucination labels.

Input rows must contain ground_truth, hallucination_label (1 means the expected
answer is hallucinated/unreliable), final_answer, and hallucination_probability.
Interactive runs have no ground truth unless the researcher annotates them.
"""
from __future__ import annotations
import argparse
import csv
import json
import re
from pathlib import Path
from metrics import binary_metrics


def norm(text): return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", default="results.csv")
    p.add_argument("--output", default="evaluation_results.csv")
    a=p.parse_args()
    with open(a.predictions, newline="", encoding="utf-8") as f: rows=list(csv.DictReader(f))
    if not rows: raise SystemExit("No prediction rows found.")
    required={"ground_truth","hallucination_label","final_answer","hallucination_probability"}
    if not required.issubset(rows[0]):
        raise SystemExit("Predictions need ground_truth and hallucination_label annotations before scoring.")
    labeled=[r for r in rows if r["ground_truth"].strip() and r["hallucination_label"].strip()]
    if not labeled: raise SystemExit("No labeled rows. Annotate ground_truth and hallucination_label first.")
    correct=[int(norm(r["final_answer"])==norm(r["ground_truth"])) for r in labeled]
    hallucinated=[int(float(r["hallucination_label"])) for r in labeled]
    # Detection confidence is calibrated against the provided hallucination labels.
    result=binary_metrics(hallucinated,[float(r["hallucination_probability"]) for r in labeled])
    result["exact_match_accuracy"] = sum(correct)/len(correct)
    result["correct_answers"] = sum(correct)
    result["incorrect_answers"] = len(correct)-sum(correct)
    result["correctly_detected_hallucinations"] = result.pop("tp")
    result["false_hallucination_detections"] = result.pop("fp")
    result["missed_hallucinations"] = result.pop("fn")
    result["correctly_accepted_non_hallucinations"] = result.pop("tn")
    consistency=[float(r["semantic_consistency"]) for r in labeled if r.get("semantic_consistency")]
    result["average_semantic_consistency"] = sum(consistency)/len(consistency) if consistency else None
    with open(a.output,"w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=["metric","value"]); writer.writeheader()
        for k,v in result.items(): writer.writerow({"metric":k,"value":v})
    print(json.dumps(result,indent=2)); print(f"Saved: {Path(a.output).resolve()}")


if __name__=="__main__": main()
