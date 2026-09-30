# verify_datasets.py
from datasets import load_dataset

print("=" * 50)
print("Datasets Verification")
print("=" * 50)

# Test loading TruthfulQA
print("Loading TruthfulQA multiple choice dataset...")
try:
    truthfulqa = load_dataset("truthful_qa", "multiple_choice")
    print(f"✓ TruthfulQA loaded successfully")
    print(f"  - Split 'validation' size: {len(truthfulqa['validation'])}")
    print(f"  - Sample: {truthfulqa['validation'][0]}")
except Exception as e:
    print(f"Warning: Could not load TruthfulQA: {e}")
    print("  You can load it manually later")

# Test loading a small C4 sample
print("\nLoading C4 dataset (small preview)...")
try:
    c4_sample = load_dataset("c4", "en", split="validation", streaming=True)
    sample = next(iter(c4_sample))
    print(f"✓ C4 dataset accessible")
    print(f"  - Sample keys: {sample.keys()}")
    print(f"  - Text length: {len(sample['text'])} characters")
except Exception as e:
    print(f"Warning: {e}")
    print("  You can load C4 when needed")