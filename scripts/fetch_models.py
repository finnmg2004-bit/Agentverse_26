#!/usr/bin/env python3
"""
Download the local perception / agent models (gitignored) — run with `pixi run fetch-models`.

  assets/models/yolov8n.onnx          YOLOv8n person detector (Ultralytics release, AGPL-3.0)
  assets/models/yamnet.tflite         YAMNet AudioSet sound classifier (TF Hub / Google, Apache-2.0)
  assets/models/yamnet_class_map.csv  YAMNet class names (tensorflow/models, Apache-2.0)
  assets/models/whisper-base.en/      Whisper base.en (CTranslate2 int8-ready, Systran, MIT) — owner voice requests
  assets/models/en_US-ljspeech-medium.onnx(.json)  Piper TTS voice for speech/tts.py (rhasspy/piper-voices,
                                      MIT; trained on the public-domain LJ Speech dataset)
  Ollama model (default gemma4:e2b)   local LLM for nudge-brain/agent.py (needs Ollama installed)

Every file is checked against a pinned SHA-256, so a corrupted or changed download is rejected.
Usage: python scripts/fetch_models.py [--no-llm] [--llm MODEL] [--stt base.en|tiny.en]
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

# faster-whisper speech-to-text models, pinned to a Hugging Face revision: name -> (revision, {file: sha256})
WHISPER_REPO = "https://huggingface.co/Systran/faster-whisper-{name}/resolve/{rev}/{file}"
WHISPER_TOKENIZER_SHA = "929c5252409436dce1b38a75d1abbcb5e132d170d8e324e4e04ed915fa2d22df"
WHISPER_VOCAB_SHA = "ff77588746d3a2595d32ab5b69ffd7b95ce2441ac57533cb66fc3eb575a115cf"
WHISPER = {
    "base.en": ("3d3d5dee26484f91867d81cb899cfcf72b96be6c", {
        "model.bin": "2a166925539a16005f14ff328359f9b9adb9dc4fb631bb3b227526862e93e2ef",
        "config.json": "f3bc3821e9fc76a27bae538e11ae5b677dcdd352b4600429ce7951d398569aeb",
        "tokenizer.json": WHISPER_TOKENIZER_SHA,
        "vocabulary.txt": WHISPER_VOCAB_SHA,
    }),
    "tiny.en": ("0d3d19a32d3338f10357c0889762bd8d64bbdeba", {
        "model.bin": "1a5afae06a4db91c975c9a9d78be5cc110ee4ea022ad57d55492e4550e936b2a",
        "config.json": "14b1b421a90349bc551b881461426b561a874049cb9e4c4864f2ca384f6a7cc5",
        "tokenizer.json": WHISPER_TOKENIZER_SHA,
        "vocabulary.txt": WHISPER_VOCAB_SHA,
    }),
}
DEFAULT_STT = os.environ.get("NUDGE_STT_MODEL", "base.en")


def whisper_files(name: str):
    rev, files = WHISPER[name]
    return [(f"whisper-{name}/{f}", WHISPER_REPO.format(name=name, rev=rev, file=f), d) for f, d in files.items()]
# Piper text-to-speech voice for speech/tts.py (pinned to a rhasspy/piper-voices commit).
_PIPER = ("https://huggingface.co/rhasspy/piper-voices/resolve/e7229f7d33a10e145e109b886d661984201ac3c4/"
          "en/en_US/ljspeech/medium/en_US-ljspeech-medium.onnx")
FILES += [
    ("en_US-ljspeech-medium.onnx", _PIPER,
     "6f52a751e2349abe7a76735eb09dc1875298c77ea2342ffd2fef79ff81b87f22"),
    ("en_US-ljspeech-medium.onnx.json", _PIPER + ".json",
     "141d612cc0a95ed7efc1ca936b845c2364967f2e9217c5dbfcf69fc4d6c65860"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(name: str, url: str, digest: str) -> bool:
    dest = MODEL_DIR / name
    dest.parent.mkdir(parents=True, exist_ok=True)
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
    ap.add_argument("--stt", default=DEFAULT_STT if DEFAULT_STT in WHISPER else "base.en", choices=sorted(WHISPER),
                    help="Whisper speech-to-text model for voice requests (default base.en)")
    args = ap.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ok = all([fetch(*f) for f in FILES + whisper_files(args.stt)])
    if not args.no_llm:
        pull_llm(args.llm)  # optional: the brain falls back to deterministic rules without it
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
