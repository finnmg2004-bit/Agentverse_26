#!/usr/bin/env python3
"""
Download the local perception / agent models (gitignored) — run with `pixi run fetch-models`.

  assets/models/yolov8n.onnx          YOLOv8n person detector (Ultralytics release, AGPL-3.0)
  assets/models/yamnet.tflite         YAMNet AudioSet sound classifier (TF Hub / Google, Apache-2.0)
  assets/models/yamnet_class_map.csv  YAMNet class names (tensorflow/models, Apache-2.0)
  Ollama model (default gemma4:e2b)   local LLM for nudge-brain/agent.py (needs Ollama installed)

Every file is checked against a pinned SHA-256, so a corrupted or changed download is rejected.
Usage: python scripts/fetch_models.py [--no-llm] [--llm MODEL]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "assets" / "models"
DEFAULT_LLM = os.environ.get("NUDGE_LLM_MODEL", "gemma4:e2b")

FILES = [
    (
        "yolov8n.onnx",
        "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.onnx",
        "b2bc52f40e8e1c532427d5bde3575a5d5b571b739fab2c6df443733ed1589cbd",
    ),
    (
        "yamnet.tflite",
        "https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/"
        "audio_classification/android/lite-model_yamnet_classification_tflite_1.tflite",
        "10c95ea3eb9a7bb4cb8bddf6feb023250381008177ac162ce169694d05c317de",
    ),
    (
        "yamnet_class_map.csv",
        "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/"
        "yamnet_class_map.csv",
        "cdf24d193e196d9e95912a2667051ae203e92a2ba09449218ccb40ef787c6df2",
    ),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(name: str, url: str, digest: str) -> bool:
    dest = MODEL_DIR / name
    if dest.exists() and sha256(dest) == digest:
        print(f"[ok]   {name} already present")
        return True
    print(f"[get]  {name} <- {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "nudge-fetch-models"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as out:
            shutil.copyfileobj(resp, out)
    except Exception as exc:
        print(f"[FAIL] {name}: {exc}")
        tmp.unlink(missing_ok=True)
        return False
    got = sha256(tmp)
    if got != digest:
        print(f"[FAIL] {name}: SHA-256 mismatch (got {got})")
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(dest)
    print(f"[ok]   {name} ({dest.stat().st_size / 1e6:.1f} MB)")
    return True


def find_ollama() -> str | None:
    exe = shutil.which("ollama")
    if exe:
        return exe
    if sys.platform == "win32":  # winget/installer default, often not on PATH in fresh shells
        cand = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
        if cand.exists():
            return str(cand)
    return None


def pull_llm(model: str) -> bool:
    exe = find_ollama()
    if exe is None:
        print(f"[skip] Ollama not installed — install it (https://ollama.com/download, or "
              f"`winget install Ollama.Ollama` on Windows), then rerun or `ollama pull {model}`.")
        return False
    print(f"[get]  Ollama model {model} (this can take a few minutes the first time)")
    ok = subprocess.run([exe, "pull", model]).returncode == 0
    print(f"[{'ok' if ok else 'FAIL'}]   Ollama model {model}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-llm", action="store_true", help="skip pulling the Ollama model")
    ap.add_argument("--llm", default=DEFAULT_LLM, help=f"Ollama model to pull (default {DEFAULT_LLM})")
    args = ap.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ok = all([fetch(*f) for f in FILES])
    if not args.no_llm:
        pull_llm(args.llm)  # optional: the brain falls back to deterministic rules without it
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
