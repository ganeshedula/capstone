import torch

print("=" * 50)
print("PyTorch Verification")
print("=" * 50)
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"MPS Available: {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False}")

if torch.cuda.is_available():
    device = "cuda"
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
    print("Using Apple Metal (MPS) GPU")
else:
    device = "cpu"
    print("Running on CPU")

x = torch.randn(3, 4, device=device)
print(f"Tensor creation successful on {device}: {x.shape}")