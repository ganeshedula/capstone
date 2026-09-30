# verify_all.py
# !/usr/bin/env python
"""
Comprehensive environment verification for the hallucination-reduction project.
"""

import sys


def verify_pytorch():
    try:
        import torch
        print("✓ PyTorch:", torch.__version__)
        print("  - CUDA available:", torch.cuda.is_available())
        print("  - MPS available:", torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False)
        return True
    except Exception as e:
        print("✗ PyTorch:", str(e))
        return False


def verify_jax():
    try:
        import jax
        print("✓ JAX:", jax.__version__)
        print("  - Devices:", jax.devices())
        return True
    except Exception as e:
        print("✗ JAX:", str(e))
        return False


def verify_transformers():
    try:
        import transformers
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        print("✓ Transformers:", transformers.__version__)
        print("  - Model loading: OK")
        return True
    except Exception as e:
        print("✗ Transformers:", str(e))
        return False


def verify_datasets():
    try:
        import datasets
        print("✓ Datasets:", datasets.__version__)
        return True
    except Exception as e:
        print("✗ Datasets:", str(e))
        return False


def verify_supporting():
    try:
        import numpy
        import scipy
        import sklearn
        import tqdm
        import sentencepiece
        print("✓ Supporting libraries: numpy, scipy, sklearn, tqdm, sentencepiece")
        return True
    except Exception as e:
        print("✗ Supporting libraries:", str(e))
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("ENVIRONMENT VERIFICATION")
    print("=" * 60)

    results = {
        "PyTorch": verify_pytorch(),
        "JAX": verify_jax(),
        "Transformers": verify_transformers(),
        "Datasets": verify_datasets(),
        "Supporting": verify_supporting(),
    }

    print("=" * 60)
    passed = sum(results.values())
    print(f"SUMMARY: {passed}/{len(results)} checks passed")

    if passed == len(results):
        print("✓ Environment is ready!")
        sys.exit(0)
    else:
        print("✗ Some checks failed. Fix errors above.")
        sys.exit(1)