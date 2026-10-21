"""Local browser interface for the interactive ask pipeline."""
from __future__ import annotations

import argparse
import json
import os
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import torch

import main as pipeline


ROOT = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(ROOT, "web")
MODEL_LOCK = threading.Lock()


class AskHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def log_message(self, fmt, *args):
        print("[ask-web] " + fmt % args)

    def do_POST(self):
        if urlparse(self.path).path != "/api/chat":
            self.send_error(404)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size > 64_000:
                self._json(413, {"error": "Request is too large."})
                return
            body = json.loads(self.rfile.read(size) or b"{}")
            question = str(body.get("question", "")).strip()
            history = body.get("history", [])
            if not question:
                self._json(400, {"error": "Enter a question first."})
                return
            if not isinstance(history, list):
                history = []
            turns = []
            for turn in history[-10:]:
                if isinstance(turn, dict) and turn.get("role") in ("user", "assistant"):
                    content = str(turn.get("content", "")).strip()[:4000]
                    if content:
                        turns.append({"role": turn["role"], "content": content})
            messages = [{"role": "system", "content": "You are a helpful assistant. Give concise, direct answers. Use the conversation to answer follow-up questions."}]
            messages.extend(turns)
            messages.append({"role": "user", "content": question})
            if hasattr(self.server.tokenizer, "apply_chat_template"):
                prompt_text = self.server.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt_text = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
            ids = self.server.tokenizer(prompt_text, return_tensors="pt").to(self.server.device)["input_ids"]
            from inference import generate_answer
            with MODEL_LOCK, torch.inference_mode():
                base = generate_answer(self.server.model, self.server.tokenizer, ids, "base",
                                        alpha=self.server.args.alpha,
                                        max_new_tokens=self.server.args.max_new_tokens)
                dola = generate_answer(self.server.model, self.server.tokenizer, ids, "dola",
                                        alpha=self.server.args.alpha,
                                        max_new_tokens=self.server.args.max_new_tokens)
                decision, winner, clustering, reports, evidence = pipeline.run_interactive_self_consistency(
                    self.server.args, self.server.model, self.server.tokenizer,
                    self.server.device, question, ids, self.server.args.temperature)
                enn = {}
                if os.path.exists(pipeline.ENN_CHECKPOINT):
                    net = pipeline.load_enn(pipeline.ENN_CHECKPOINT, device=self.server.device)
                    enn = generate_answer(
                        self.server.model, self.server.tokenizer, ids, "dola_enn",
                        alpha=self.server.args.alpha, epinet=net,
                        vocab_head_weight=pipeline.get_vocab_head_weight(self.server.model),
                        n_z_samples=3, max_new_tokens=self.server.args.max_new_tokens)
            selected = decision.get("selected") or max(reports, key=lambda item: item.get("final_score", 0))
            self._json(200, {
                "answer": selected.get("answer", ""),
                "decision": decision.get("decision", "UNCERTAIN"),
                "score": selected.get("final_score"),
                "confidence": selected.get("mean_token_confidence"),
                "consensus": selected.get("metrics", {}).get("semantic_consistency", clustering.get("largest_cluster_ratio")),
                "reliability": selected.get("metrics", {}).get("reliability_probability"),
                "hallucination_risk": selected.get("metrics", {}).get("hallucination_score"),
                "factuality": selected.get("metrics", {}).get("factuality_score"),
                "verifier_status": selected.get("factuality", {}).get("verifier_status", "UNKNOWN"),
                "candidate_count": len(reports),
                "details": {
                    "base": self._answer_details(base),
                    "dola": self._answer_details(dola),
                    "self_consistency": {
                        "support": clustering.get("largest_cluster_ratio"),
                        "cluster_count": clustering.get("cluster_count"),
                        "average_similarity": clustering.get("average_pairwise_similarity"),
                        "candidates": [self._candidate_details(item) for item in reports],
                    },
                    "enn": self._answer_details(enn) if enn else None,
                    "final": self._candidate_details(selected),
                    "evidence": evidence,
                },
            })
        except Exception as exc:
            self._json(500, {"error": str(exc)})

    def _json(self, status, payload):
        data = json.dumps(payload, default=self._json_default).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    @staticmethod
    def _json_default(value):
        if hasattr(value, "item"):
            return value.item()
        if hasattr(value, "tolist"):
            return value.tolist()
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        return str(value)

    @staticmethod
    def _answer_details(item):
        return {"answer": item.get("answer", ""),
                "confidence": item.get("mean_token_confidence", item.get("confidence")),
                "entropy": item.get("mean_token_entropy"),
                "uncertain_token_ratio": item.get("uncertain_token_ratio"),
                "dola_layers": item.get("dola_layers", []),
                "token_probabilities": item.get("token_probabilities", []),
                "uncertainty_curve": item.get("uncertainty_curve", []),
                "uncertainty": item.get("uncertainty")}

    @staticmethod
    def _candidate_details(item):
        factuality = item.get("factuality", {})
        return {"answer": item.get("answer", ""), "rank": item.get("rank"),
                "score": item.get("final_score"), "confidence": item.get("mean_token_confidence"),
                "metrics": item.get("metrics", {}),
                "verifier_status": factuality.get("verifier_status", "UNKNOWN"),
                "verifier_details": factuality.get("verifier_raw", ""),
                "claims": item.get("claim_diagnostics", {}).get("claims", [])}


def main():
    parser = argparse.ArgumentParser(description="Start the local ask-mode chat page")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--sc-samples", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--enn-weight", type=float, default=0.0)
    parser.add_argument("--use-rag", action="store_true")
    parser.add_argument("--semantic-similarity-threshold", type=float, default=0.78)
    parser.add_argument("--reliability-threshold", type=float, default=0.55)
    parser.add_argument("--consistency-threshold", type=float, default=0.4)
    parser.add_argument("--hallucination-threshold", type=float, default=0.55)
    parser.add_argument("--score-weights", default=None)
    parser.add_argument("--penalty-weights", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.sc_samples < 1 or args.max_new_tokens < 1:
        parser.error("--sc-samples and --max-new-tokens must be positive")

    device, dtype = pipeline.get_device()
    print(f"[ask-web] Loading TinyLlama on {device}; this can take a moment.")
    tokenizer = pipeline._load_tokenizer(pipeline.MODEL_DIR)
    model = pipeline._load_model(pipeline.MODEL_DIR, device, dtype)
    model.eval()
    server = ThreadingHTTPServer((args.host, args.port), AskHandler)
    server.tokenizer, server.model, server.device, server.args = tokenizer, model, device, args
    print(f"[ask-web] Open http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[ask-web] Shutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
