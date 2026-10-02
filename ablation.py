"""Summarize a labeled, method-wise ablation CSV into ablation_results.csv.

Required columns: method, correct, hallucination_label, hallucination_probability.
Use one row per question and method; keep the same question split across methods.
"""
import argparse
import csv
from collections import defaultdict
from metrics import binary_metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions",required=True)
    p.add_argument("--output",default="ablation_results.csv")
    args=p.parse_args(); grouped=defaultdict(list)
    with open(args.predictions,newline="",encoding="utf-8") as f:
        for row in csv.DictReader(f): grouped[row["method"]].append(row)
    fields=["method","accuracy","precision","recall","f1","hallucination_detection_rate","false_positive_rate","false_negative_rate","ece","brier_score","average_confidence","average_semantic_consistency"]
    with open(args.output,"w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for method,rows in grouped.items():
            if any(not r.get(k) for r in rows for k in ("correct","hallucination_label","hallucination_probability")):
                raise SystemExit(f"Missing labels or scores for {method}")
            m=binary_metrics([int(r["hallucination_label"]) for r in rows],[float(r["hallucination_probability"]) for r in rows])
            consistency=[float(r["semantic_consistency"]) for r in rows if r.get("semantic_consistency")]
            writer.writerow({"method":method,"accuracy":sum(int(r["correct"]) for r in rows)/len(rows),
                             **{k:m[k] for k in fields if k in m},
                             "average_semantic_consistency":sum(consistency)/len(consistency) if consistency else ""})
    print(f"Saved ablation summary to {args.output}")


if __name__=="__main__":main()
