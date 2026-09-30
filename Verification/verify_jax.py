# verify_jax.py
import jax
import jax.numpy as jnp

print("=" * 50)
print("JAX Verification")
print("=" * 50)
print(f"JAX Version: {jax.__version__}")
print(f"Available Devices: {jax.devices()}")

# Test array creation and operations
arr = jnp.ones((3, 4))
result = jnp.mean(arr)
print(f"Array operation successful: result = {result}")