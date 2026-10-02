"""Train the answer-level reliability ENN from a labeled JSONL file."""
import argparse
import json
import torch
from pathlib import Path

from importlib.util import spec_from_file_location, module_from_spec

ROOT = Path(__file__).resolve().parent
spec = spec_from_file_location("reliability_enn", ROOT / "Neural Network" / "reliability_enn.py")
module = module_from_spec(spec); spec.loader.exec_module(module)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="JSONL rows: {features: {...}, reliable: 0|1}")
    parser.add_argument("--output", default=str(ROOT / "checkpoints" / "reliability_enn.pt"))
    parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    device = "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    module.train_reliability_enn(rows, args.output, epochs=args.epochs, device=device)
    print(f"Saved answer-level reliability ENN: {args.output}")
    print("Train/evaluate with question-disjoint splits before interpreting probabilities as calibrated.")


if __name__ == "__main__": main()
