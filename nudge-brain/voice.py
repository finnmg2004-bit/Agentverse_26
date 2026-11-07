#!/usr/bin/env python3
"""
Owner voice input (`nudge-brain/voice.py`): real laptop microphone -> local speech-to-text.

Tap-to-talk: press [T] in the dashboard (or call `VoiceInput.listen()`), speak, and stop talking:
recording ends ~0.8 s after Silero VAD stops hearing speech (max 8 s; gives up after 5 s of no speech). The clip is transcribed by Whisper (faster-whisper,
CTranslate2 int8 on the CPU, ~0.3-0.5 s per phrase) from `assets/models/whisper-<model>/`, fully
offline (`pixi run fetch-models`). The text then goes through `intents.py` like a typed request.

Microphone backends, first that works (override with NUDGE_MIC=pipewire|pulse|alsa|sounddevice|off):
  Linux:   pw-record (PipeWire) -> parecord (PulseAudio) -> arecord (ALSA default device)
  Any OS:  sounddevice (PortAudio), recorded at the device rate and resampled to 16 kHz

`inject_wav(path)` runs a recorded clip through the same pipeline (headless tests, no mic needed).
The voice only says WHAT the owner wants; it is not used to locate or navigate to the owner.
"""

from __future__ import annotations

import math
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "assets" / "models"
SR = 16000
FRAME = 480  # 30 ms

# Endpointing (Silero VAD; the mic is only open after [T], so no wake word is needed)
WAIT_FOR_SPEECH_S = 5.0
END_SILENCE_S = 0.8
MAX_UTTERANCE_S = 8.0
MIN_SPEECH_S = 0.25

# Biases Whisper toward the words Nudge understands (rooms, commands).
STT_PROMPT = ("Nudge, come here. Take me to the kitchen. Stop. Front door, back door, living room, "
              "bedroom, dining room, family room, hallway, garden.")


def _resample(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == SR:
        return x.astype(np.float32)
    from scipy.signal import resample_poly

    g = math.gcd(SR, int(sr))
    return resample_poly(x, SR // g, int(sr) // g).astype(np.float32)


def load_wav(path: str) -> np.ndarray:
    """Any PCM WAV -> mono float32 at 16 kHz."""
    with wave.open(str(path)) as w:
        sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"{path}: only 16-bit PCM WAV is supported")
    a = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return _resample(a, sr)


# ---------------------------------------------------------------------- microphone
class _ProcessMic:
    """Raw s16le mono 16 kHz from a recorder CLI on stdout."""

    def __init__(self, name: str, cmd: List[str]) -> None:
        self.name, self.cmd = name, cmd
        self.proc: Optional[subprocess.Popen] = None

    def open(self) -> None:
        self.proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)

    def read(self, n: int) -> Optional[np.ndarray]:
        want, buf = n * 2, b""
        while len(buf) < want:
            chunk = self.proc.stdout.read(want - len(buf))
            if not chunk:
                return None
            buf += chunk
        return np.frombuffer(buf, np.int16).astype(np.float32) / 32768.0

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None


class _SoundDeviceMic:
    name = "sounddevice"

    def __init__(self) -> None:
        import sounddevice as sd  # type: ignore

        self.sd = sd
        self.rate = int(sd.query_devices(kind="input")["default_samplerate"])
        self.stream = None
        self.pending = np.zeros(0, np.float32)

    def open(self) -> None:
        self.stream = self.sd.InputStream(samplerate=self.rate, channels=1, dtype="float32")
        self.stream.start()
        self.pending = np.zeros(0, np.float32)

    def read(self, n: int) -> Optional[np.ndarray]:
        while self.pending.size < n:
            data, _ = self.stream.read(int(math.ceil(n * self.rate / SR)) + 1)
            self.pending = np.concatenate([self.pending, _resample(data[:, 0], self.rate)])
        out, self.pending = self.pending[:n], self.pending[n:]
        return out

    def close(self) -> None:
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
        self.stream = None


def make_mic(choice: str = ""):
    choice = (choice or os.environ.get("NUDGE_MIC", "")).lower()
    if choice == "off":
        return None
    cli = {
        "pipewire": ("pw-record", ["pw-record", "--raw", "--rate", str(SR), "--channels", "1", "--format", "s16", "-"]),
        "pulse": ("parecord", ["parecord", "--raw", f"--rate={SR}", "--channels=1", "--format=s16le"]),
        "alsa": ("arecord", ["arecord", "-q", "-D", "default", "-f", "S16_LE", "-r", str(SR), "-c", "1", "-t", "raw"]),
    }
    order = [choice] if choice else (["pipewire", "pulse", "alsa", "sounddevice"] if sys.platform.startswith("linux")
                                     else ["sounddevice"])
    for name in order:
        if name in cli and shutil.which(cli[name][0]):
            return _ProcessMic(cli[name][0], cli[name][1])
        if name == "sounddevice":
            try:
                return _SoundDeviceMic()
            except Exception as exc:
                print(f"[voice] sounddevice unavailable: {exc}")
    return None


# ---------------------------------------------------------------------- speech to text
class SpeechToText:
    def __init__(self, model: str = "") -> None:
        self.name = model or os.environ.get("NUDGE_STT_MODEL", "base.en")
        self.model = None
        self.error: Optional[str] = None
        self.ready = threading.Event()
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self) -> None:
        path = MODEL_DIR / f"whisper-{self.name}"
        try:
            if not (path / "model.bin").exists():
                raise FileNotFoundError(f"{path} missing - run `pixi run fetch-models`")
            from faster_whisper import WhisperModel  # type: ignore

            self.model = WhisperModel(str(path), device="cpu", compute_type="int8", cpu_threads=4,
                                      local_files_only=True)
            # Warm-up so the first real request isn't slower.
            list(self.model.transcribe(np.zeros(SR // 2, np.float32), language="en", beam_size=1)[0])
        except Exception as exc:
            self.error = str(exc)
            print(f"[voice] speech-to-text unavailable: {exc}")
        self.ready.set()

    @property
    def backend(self) -> str:
        if not self.ready.is_set():
            return f"whisper-{self.name} (loading)"
        return f"whisper-{self.name}" if self.model is not None else "off"

    def transcribe(self, audio: np.ndarray) -> str:
        self.ready.wait(30.0)
        if self.model is None:
            raise RuntimeError(self.error or "speech-to-text model not loaded")
        segs, _ = self.model.transcribe(audio.astype(np.float32), language="en", beam_size=1,
                                        initial_prompt=STT_PROMPT, condition_on_previous_text=False,
                                        vad_filter=False)
        segs = list(segs)
        # Whisper invents text for noise ("Thank you."); drop segments it thinks are not speech.
        return " ".join(s.text for s in segs if s.no_speech_prob < 0.6).strip()


# ---------------------------------------------------------------------- facade
class VoiceInput:
    """Tap-to-talk microphone + Whisper. Thread-based; the 20 Hz brain loop only polls."""

    def __init__(self, enabled: bool = True, mic: str = "") -> None:
        self.enabled = enabled
        self.stt = SpeechToText() if enabled else None
        self.mic = make_mic(mic) if enabled else None
        self.state = "idle"  # idle | listening | transcribing
        self.level = 0.0     # live mic RMS for the dashboard meter
        self.results: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._busy = threading.Lock()
        self._stop = threading.Event()
        if enabled:
            print(f"[voice] mic: {self.mic.name if self.mic else 'none'} | stt: {self.stt.backend}")

    @property
    def backend(self) -> str:
        if not self.enabled:
            return "off"
        return f"{self.mic.name if self.mic else 'no mic'} + {self.stt.backend}"

    def listen(self) -> bool:
        """Start one tap-to-talk capture (ignored if already busy or there's no mic)."""
        if not self.enabled or self.mic is None or not self._busy.acquire(blocking=False):
            return False
        self._stop.clear()
        threading.Thread(target=self._capture, daemon=True).start()
        return True

    def inject_wav(self, path: str) -> bool:
        """Transcribe a recorded clip as if the owner had just said it."""
        if not self.enabled or not self._busy.acquire(blocking=False):
            return False
        threading.Thread(target=self._from_wav, args=(str(path),), daemon=True).start()
        return True

    def poll(self) -> Optional[Dict[str, Any]]:
        try:
            return self.results.get_nowait()
        except queue.Empty:
            return None

    def close(self) -> None:
        self._stop.set()

    # -------------------------------------------------------------- workers
    def _from_wav(self, path: str) -> None:
        try:
            self._finish(load_wav(path), "voice-file")
        except Exception as exc:
            self.results.put({"text": "", "error": str(exc), "channel": "voice-file"})
        finally:
            self.state = "idle"
            self._busy.release()

    def _capture(self) -> None:
        """Record until the owner stops talking. Speech start/end come from Silero VAD (bundled with
        faster-whisper), which ignores steady fan / hum noise that fools a loudness threshold."""
        self.state = "listening"
        frames: List[np.ndarray] = []
        try:
            from faster_whisper.vad import VadOptions, get_speech_timestamps  # type: ignore

            vad_opts = VadOptions(threshold=0.5, min_speech_duration_ms=200, min_silence_duration_ms=300,
                                  speech_pad_ms=200)
            self.mic.open()
            started = time.monotonic()
            first_speech: Optional[float] = None
            speech: List[dict] = []
            while not self._stop.is_set():
                f = self.mic.read(FRAME)
                if f is None:
                    raise RuntimeError(f"{self.mic.name} stopped delivering audio")
                frames.append(f)
                self.level = float(np.sqrt(np.mean(f * f)))
                if len(frames) % 8:  # run the VAD every ~0.25 s
                    continue
                audio = np.concatenate(frames)
                speech = get_speech_timestamps(audio, vad_opts)
                now = time.monotonic()
                if not speech:
                    if now - started > WAIT_FOR_SPEECH_S:
                        self.results.put({"text": "", "error": "heard nothing" + self._mic_hint(audio),
                                          "channel": "voice"})
                        return
                    continue
                first_speech = first_speech or now
                silent_after = (audio.size - speech[-1]["end"]) / SR
                if silent_after >= END_SILENCE_S or now - first_speech >= MAX_UTTERANCE_S:
                    break
            self.mic.close()
            audio = np.concatenate(frames)
            if not speech:
                return
            talk = sum(seg["end"] - seg["start"] for seg in speech) / SR
            if talk < MIN_SPEECH_S:
                self.results.put({"text": "", "error": "too short", "channel": "voice"})
                return
            clip = audio[max(0, speech[0]["start"] - SR // 5): speech[-1]["end"] + SR // 5]
            self._finish(clip, "voice", warning=self._mic_hint(audio).strip(" ()") or None)
        except Exception as exc:
            self.results.put({"text": "", "error": str(exc), "channel": "voice"})
        finally:
            self.mic.close()
            self.level = 0.0
            self.state = "idle"
            self._busy.release()

    @staticmethod
    def _mic_hint(audio: np.ndarray) -> str:
        if audio.size == 0 or float(np.max(np.abs(audio))) < 1e-4:
            return " (mic muted or silent?)"
        if float(np.mean(np.abs(audio) > 0.98)) > 0.01:
            return " (mic is clipping: lower the mic boost / input volume)"
        return ""

    def _finish(self, audio: np.ndarray, channel: str, warning: Optional[str] = None) -> None:
        self.state = "transcribing"
        t0 = time.monotonic()
        text = self.stt.transcribe(audio)
        res = {"text": text, "channel": channel, "stt_s": round(time.monotonic() - t0, 2),
               "audio_s": round(audio.size / SR, 2)}
        if not text:
            res["error"] = "no words" + (f" ({warning})" if warning else "")
        elif warning:
            res["warning"] = warning
        self.results.put(res)
