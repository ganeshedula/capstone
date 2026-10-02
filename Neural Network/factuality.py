"""Prompted factuality review with optional evidence support.

Scores are model-judge estimates, not externally validated probabilities.
"""
from __future__ import annotations
import json
import re
import torch


@torch.inference_mode()
def verify_answer(model, tokenizer, question, answer, evidence=None, max_new_tokens=100):
    evidence_text = "\n".join(f"- {x}" for x in (evidence or [])) or "No external evidence supplied."
    def prompt_for(strict=False):
      schema = ('{"factuality_score": 0.0, "hallucination_score": 0.0, '
                '"contradiction_score": 0.0, "claim_support": 0.0, '
                '"relevance_score": 0.0, "evidence_support_score": 0.0, "reason": "..."}')
      instruction = (
        "Act as a skeptical factuality reviewer. Assess only the answer to the question. "
        "Check relevance, factual plausibility, causal reasoning, unsupported claims, and contradictions. "
        "When evidence is supplied, treat it as the only verified source and penalize unsupported claims. "
        "Return exactly one JSON object with numeric values from 0 to 1 for factuality_score, "
        "hallucination_score, relevance_score, evidence_support_score, contradiction_score, "
        "plus a short reason. Do not reward confidence or fluent wording.\n"
        f"Question: {question}\nAnswer: {answer}\nEvidence:\n{evidence_text}\n"
        f"Return this schema with numeric values in [0,1]: {schema}\nJSON:"
      )
      if strict:
        instruction += " Return JSON only. No markdown, preamble, or additional text."
      return tokenizer.apply_chat_template(
        [{"role": "system", "content": "You are a careful factuality evaluator."},
         {"role": "user", "content": instruction}], tokenize=False,
        add_generation_prompt=True) if hasattr(tokenizer, "apply_chat_template") else instruction
    raw = ""
    for attempt in range(2):
        prompt = prompt_for(strict=bool(attempt))
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(next(model.parameters()).device)
        generated = model.generate(ids, do_sample=False, max_new_tokens=max_new_tokens,
                                   pad_token_id=tokenizer.eos_token_id,
                                   eos_token_id=tokenizer.eos_token_id)
        raw = tokenizer.decode(generated[0, ids.shape[1]:], skip_special_tokens=True)
        match = re.search(r"\{.*?\}", raw, flags=re.S)
        try: data = json.loads(match.group(0)) if match else {}
        except (ValueError, TypeError): data = {}
        required = ("factuality_score", "hallucination_score", "contradiction_score", "claim_support")
        if all(k in data and _valid_unit(data[k]) for k in required):
            values = {key: _unit(data.get(key)) for key in
                      ("factuality_score", "hallucination_score", "relevance_score",
                       "evidence_support_score", "contradiction_score", "claim_support")}
            values.update({"reason": str(data.get("reason", "")), "verifier_raw": raw,
                           "verifier_parse_ok": True, "verifier_status": "OK" if not attempt else "RETRIED"})
            return values
    # Explicit neutral fallback: parser failure is not itself a hallucination signal.
    return {"factuality_score": .5, "hallucination_score": .5, "contradiction_score": 0.0,
            "claim_support": .5, "relevance_score": .5, "evidence_support_score": .5,
            "reason": "Verifier returned invalid JSON; neutral fallback used.",
            "verifier_raw": raw, "verifier_parse_ok": False, "verifier_status": "FALLBACK"}


def _valid_unit(value):
    try: return 0.0 <= float(value) <= 1.0
    except (TypeError, ValueError): return False


def _unit(value):
    try: return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError): return 0.5
