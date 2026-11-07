#!/usr/bin/env python3
"""
Local text-to-speech for Nudge's spoken alerts (Piper, CPU via onnxruntime, ~0.1-0.3 s per sentence).

    from speech.tts import speak, synthesize
    speak("The doorbell is ringing. Take my handle.")   # returns at once, plays in the background
    pcm, sr = synthesize("Hello")                        # int16 mono numpy array + sample rate

Voice: assets/models/<NUDGE_TTS_VOICE>.onnx (default en_US-ljspeech-medium, LJ Speech
dataset = public domain; get it with `pixi run fetch-models`). NUDGE_TTS_VOICE may also be a
path to any Piper .onnx voice (with its .onnx.json next to it).

Fail soft: no model / no piper package / no audio device -> one log line, then the calls just
return (synthesize -> empty array). Set NUDGE_TTS_MUTE=1 to synthesize without playing (headless).
Playback backends, tried in order: sounddevice (PortAudio, only if installed), winsound
(Windows, built in), paplay / pw-play / aplay (Linux). One worker thread speaks utterances in order, so the caller (e.g. the 500 Hz
physics loop) never blocks unless it passes block=True.

CLI: python speech/tts.py "text to say" [--out file.wav]
"""

from __future__ import annotations

import io
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VOICE_DIR = PROJECT_ROOT / "assets" / "models"
DEFAULT_VOICE = "en_US-ljspeech-medium"  # keep in sync with scripts/fetch_models.py
DEFAULT_SR = 22050

_lock = threading.Lock()
_voice = None              # piper.PiperVoice, loaded once
_voice_failed = False
_warned: set[str] = set()
_cache: "OrderedDict[str, Tuple[np.ndarray, int]]" = OrderedDict()
_CACHE_MAX = 32            # alerts repeat ("Take my handle."), so keep recent clips

_queue: "queue.Queue[Tuple[str, threading.Event]]" = queue.Queue()
_worker: Optional[threading.Thread] = None
_speaking = threading.Event()
_stop_now = threading.Event()
_player_proc: Optional[subprocess.Popen] = None
_backend: Optional[str] = None  # playback backend that worked last


def _log_once(key: str, msg: str) -> None:
    if key not in _warned:
        _warned.add(key)
        print(f"[tts] {msg}", flush=True)


def voice_path() -> Path:
    name = os.environ.get("NUDGE_TTS_VOICE", DEFAULT_VOICE)
    p = Path(name)
    if p.suffix == ".onnx":
        return p if p.is_absolute() else PROJECT_ROOT / p
    return VOICE_DIR / f"{name}.onnx"


def _load_voice():
    """Load the Piper voice once (thread-safe). Returns None if unavailable."""
    global _voice, _voice_failed
    if _voice is not None or _voice_failed:
        return _voice
    with _lock:
        if _voice is not None or _voice_failed:
            return _voice
        path = voice_path()
        if not path.exists():
            _voice_failed = True
            _log_once("model", f"voice model {path} missing - run `pixi run fetch-models`; speech disabled")
            return None
        try:
            from piper import PiperVoice  # type: ignore

            t0 = time.perf_counter()
            _voice = PiperVoice.load(str(path))
            _log_once("loaded", f"Piper voice {path.stem} loaded in {time.perf_counter() - t0:.2f} s")
        except Exception as exc:  # missing package, bad file, onnxruntime error
            _voice_failed = True
            _log_once("model", f"cannot load Piper voice ({exc}); speech disabled")
    return _voice


def available() -> bool:
    """True if a voice is installed and loads (does not check the audio device)."""
    return _load_voice() is not None


def warmup(background: bool = True) -> None:
    """Load the voice (~1-5 s) and run one tiny synthesis so the first real alert is fast.
    Call it at startup; with background=True it returns at once."""
    if background:
        threading.Thread(target=synthesize, args=("Ready.",), name="nudge-tts-warmup", daemon=True).start()
    else:
        synthesize("Ready.")


def synthesize(text: str) -> Tuple[np.ndarray, int]:
    """Text -> (int16 mono PCM, sample rate). Empty array if TTS is unavailable."""
    text = (text or "").strip()
    voice = _load_voice()
    if not text or voice is None:
        return np.zeros(0, dtype=np.int16), DEFAULT_SR
    with _lock:
        hit = _cache.get(text)
        if hit is not None:
            _cache.move_to_end(text)
            return hit
    try:
        chunks = list(voice.synthesize(text))
    except Exception as exc:
        _log_once("synth", f"synthesis failed ({exc})")
        return np.zeros(0, dtype=np.int16), DEFAULT_SR
    if not chunks:
        return np.zeros(0, dtype=np.int16), DEFAULT_SR
    sr = int(chunks[0].sample_rate)
    pcm = np.concatenate([c.audio_int16_array.reshape(-1) for c in chunks]).astype(np.int16)
    with _lock:
        _cache[text] = (pcm, sr)
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)
    return pcm, sr


def wav_bytes(pcm: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(np.asarray(pcm, dtype=np.int16).tobytes())
    return buf.getvalue()


def save_wav(text: str, path: str | Path) -> bool:
    pcm, sr = synthesize(text)
    if pcm.size == 0:
        return False
    Path(path).write_bytes(wav_bytes(pcm, sr))
    return True


# --------------------------------------------------------------------------- playback

def _play_sounddevice(pcm: np.ndarray, sr: int) -> None:
    import sounddevice as sd  # type: ignore

    sd.play(pcm, sr)
    dur = len(pcm) / sr
    t_end = time.monotonic() + dur + 0.5
    while time.monotonic() < t_end:  # sd.wait() can't be interrupted; poll so stop() works
        if _stop_now.is_set():
            sd.stop()
            return
        stream = sd.get_stream()
        if not stream.active:
            return
        time.sleep(0.02)
    sd.stop()


def _play_winsound(pcm: np.ndarray, sr: int) -> None:
    import winsound  # type: ignore

    fd, tmp = tempfile.mkstemp(suffix=".wav", prefix="nudge_tts_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(wav_bytes(pcm, sr))
        winsound.PlaySound(tmp, winsound.SND_FILENAME | winsound.SND_NODEFAULT)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _play_cli(pcm: np.ndarray, sr: int) -> None:
    global _player_proc
    for exe, args in (("paplay", []), ("pw-play", ["-"]), ("aplay", ["-q", "-"])):
        path = shutil.which(exe)
        if path is None:
            continue
        proc = subprocess.Popen([path, *args], stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        _player_proc = proc
        try:
            proc.stdin.write(wav_bytes(pcm, sr))  # type: ignore[union-attr]
            proc.stdin.close()  # type: ignore[union-attr]
            while proc.poll() is None:
                if _stop_now.is_set():
                    proc.kill()
                    return
                time.sleep(0.02)
        finally:
            _player_proc = None
        if proc.returncode == 0:
            return
    raise RuntimeError("no working paplay / pw-play / aplay")


def _backends():
    yield "sounddevice", _play_sounddevice
    if sys.platform == "win32":
        yield "winsound", _play_winsound
    else:
        yield "cli", _play_cli


def _play(pcm: np.ndarray, sr: int) -> bool:
    global _backend
    if pcm.size == 0:
        return False
    if os.environ.get("NUDGE_TTS_MUTE", "0") == "1":
        return False
    errors = []
    for name, fn in _backends():
        if _backend is not None and name != _backend and _backend != "none":
            continue  # stick with the backend that worked
        try:
            fn(pcm, sr)
            _backend = name
            return True
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    _backend = None
    _log_once("audio", "no audio output (" + "; ".join(errors) + "); speech is text-only")
    return False


def _worker_loop() -> None:
    while True:
        text, done = _queue.get()
        try:
            if _stop_now.is_set():
                continue
            pcm, sr = synthesize(text)
            _speaking.set()
            _play(pcm, sr)
        except Exception as exc:  # never let the worker die
            _log_once("worker", f"speak failed ({exc})")
        finally:
            _speaking.clear()
            done.set()
            _queue.task_done()


def _ensure_worker() -> None:
    global _worker
    with _lock:
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_worker_loop, name="nudge-tts", daemon=True)
            _worker.start()


def speak(text: str, block: bool = False, interrupt: bool = False) -> bool:
    """Say `text` on the local speaker. Non-blocking by default (queued, spoken in order).

    interrupt=True cuts off the current utterance and drops queued ones first (use it for an
    emergency message). block=True waits until this text has been spoken.
    Returns False if nothing will be said (empty text or TTS unavailable), else True.
    """
    text = (text or "").strip()
    if not text or _voice_failed:
        return False
    if _voice is None and not voice_path().exists():
        _load_voice()  # logs the missing model once, never blocks
        return False
    if interrupt:
        stop()
    _stop_now.clear()
    _ensure_worker()
    done = threading.Event()
    _queue.put((text, done))
    if block:
        done.wait()
    return True


def stop() -> None:
    """Stop the current utterance and drop everything queued."""
    _stop_now.set()
    try:
        while True:
            _, done = _queue.get_nowait()
            done.set()
            _queue.task_done()
    except queue.Empty:
        pass
    proc = _player_proc
    if proc is not None and proc.poll() is None:
        proc.kill()
    if sys.platform == "win32" and _backend == "winsound":
        try:
            import winsound  # type: ignore

            winsound.PlaySound(None, 0)
        except Exception:
            pass


def is_speaking() -> bool:
    return _speaking.is_set() or not _queue.empty()


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Say a sentence with Nudge's local Piper voice.")
    ap.add_argument("text", nargs="*", default=["Hello, I am Nudge. The doorbell is ringing. Take my handle."])
    ap.add_argument("--out", help="write a WAV file instead of playing")
    args = ap.parse_args()
    text = " ".join(args.text)
    t0 = time.perf_counter()
    pcm, sr = synthesize(text)
    if pcm.size == 0:
        print("[tts] nothing synthesized")
        return 1
    print(f"[tts] {len(pcm) / sr:.2f} s of audio in {time.perf_counter() - t0:.2f} s (incl. load)")
    if args.out:
        Path(args.out).write_bytes(wav_bytes(pcm, sr))
        print(f"[tts] wrote {args.out}")
    else:
        speak(text, block=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
