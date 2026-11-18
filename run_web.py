#!/usr/bin/env python3
"""Launcher script for the Streamlit Neural Interactive Web Console."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Start the Neural Interactive Console (Streamlit)")
    parser.add_argument("--port", type=int, default=8501, help="Port to serve Streamlit on (default: 8501)")
    parser.add_argument("--host", default="127.0.0.1", help="Host address to bind to (default: 127.0.0.1)")
    parser.add_argument("--headless", action="store_true", help="Run Streamlit in headless mode")
    args, extra = parser.parse_known_args()

    root_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(root_dir, "web", "app.py")
    
    # Use python -m streamlit to ensure the exact environment is used
    venv_python = os.path.join(root_dir, "llm_hallucination_env", "bin", "python")
    python_cmd = venv_python if os.path.isfile(venv_python) else sys.executable

    dependency_check = subprocess.run(
        [
            python_cmd,
            "-c",
            "import streamlit, langchain_huggingface, qdrant_client, ddgs",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if dependency_check.returncode:
        print(f"[run_web] Required Streamlit/RAG packages are missing from {python_cmd}.")
        print(f"[run_web] Install them with: {python_cmd} -m pip install -r {os.path.join(root_dir, 'requirements.txt')}")
        if dependency_check.stderr:
            print(dependency_check.stderr.strip())
        return 1

    cmd = [
        python_cmd,
        "-m",
        "streamlit",
        "run",
        app_path,
        "--server.port",
        str(args.port),
        "--server.address",
        args.host,
    ]
    if args.headless:
        cmd.extend(["--server.headless", "true"])
    cmd.extend(extra)

    print(f"[run_web] Python: {python_cmd}")
    print(f"[run_web] Starting Neural Interactive Console on http://{args.host}:{args.port}")
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n[run_web] Stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
