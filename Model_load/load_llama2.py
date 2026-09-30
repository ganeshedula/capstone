# load_llama2.py
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Local model path (relative to project root)
LOCAL_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "TinyLlama-1.1B-Chat-v1.0")
LOCAL_MODEL_DIR = os.path.abspath(LOCAL_MODEL_DIR)

MODEL_PATH = LOCAL_MODEL_DIR


def load_tokenizer(model_path: str) -> object:
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_path: str, device: str, dtype: torch.dtype) -> torch.nn.Module:
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model = model.to(device)
        # TinyLlama is a fixed feature extractor.  This also prevents an
        # accidental optimizer from fine-tuning it during ENN training.
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
        return model
    except Exception as e:
        if device != "cpu":
            print(f"Primary load failed on {device}: {e}")
            print("Falling back to CPU...")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            ).to("cpu")
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            model.eval()
            return model
        raise


def build_prompt(tokenizer: object, user_prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Give concise, direct answers."},
            {"role": "user", "content": user_prompt},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return user_prompt


def main() -> None:
    if not os.path.isdir(MODEL_PATH):
        raise FileNotFoundError(
            f"Local model not found at: {MODEL_PATH}\n"
            "Run: python -c \"from huggingface_hub import snapshot_download; "
            "snapshot_download('TinyLlama/TinyLlama-1.1B-Chat-v1.0', local_dir='models/TinyLlama-1.1B-Chat-v1.0')\""
        )

    print(f"Loading model from: {MODEL_PATH}")

    device = "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32

    print(f"Device: {device} | dtype: {dtype}")

    tokenizer = load_tokenizer(MODEL_PATH)
    model = load_model(MODEL_PATH, device, dtype)
    model.eval()

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B parameters")

    input_text = "Hi"
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

    print(f"\nInput:  {input_text}")
    print(f"Output: {result}")


if __name__ == "__main__":
    main()
