# load_llama.py
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "meta-llama/Llama-2-7b-hf"


def get_device_and_dtype() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_name: str, device: str, dtype: torch.dtype, load_in_8bit: bool = False):
    if load_in_8bit:
        if device != "cuda":
            raise ValueError("8-bit loading is only supported on CUDA GPUs.")
        try:
            import bitsandbytes  # noqa: F401
        except Exception as exc:
            raise RuntimeError("Install bitsandbytes to use 8-bit loading.") from exc

        return AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    return model.to(device)


def build_prompt(tokenizer, user_prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Give concise, direct answers."},
            {"role": "user", "content": user_prompt},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return user_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="Load Llama-2 with lower precision settings.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--load-in-8bit", action="store_true")
    args = parser.parse_args()

    device, dtype = get_device_and_dtype()

    print(f"Loading model: {args.model_name}")
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")
    if args.load_in_8bit:
        print("Using 8-bit quantization on CUDA")

    tokenizer = load_tokenizer(args.model_name)
    model = load_model(args.model_name, device, dtype, load_in_8bit=args.load_in_8bit)
    model.eval()

    print("✓ Model loaded successfully!")
    print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B parameters")

    input_text = "What is the capital of France?"
    prompt_text = build_prompt(tokenizer, input_text)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    print("\nTest inference:")
    print(f"Input: {input_text}")
    print(f"Output: {result}")


if __name__ == "__main__":
    main()

