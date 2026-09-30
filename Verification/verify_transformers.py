# verify_transformers.py
from transformers import AutoTokenizer

print("=" * 50)
print("Transformers Verification")
print("=" * 50)

# Test tokenizer loading (uses a small model for fast verification)
print("Loading GPT-2 tokenizer for quick verification...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Test tokenization
test_text = "Hello, how are you?"
tokens = tokenizer.encode(test_text)
decoded = tokenizer.decode(tokens)

print(f"Transformers Version: {__import__('transformers').__version__}")
print(f"Test text: {test_text}")
print(f"Tokens: {tokens}")
print(f"Decoded: {decoded}")
print("✓ Tokenization successful")
