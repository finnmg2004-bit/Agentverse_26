#!/usr/bin/env python3
"""
Owner voice input (`nudge-brain/voice.py`, run with `pixi run voice`): real laptop microphone ->
local speech-to-text -> `user_command` for the brain (AGENTS.md 9.3 / 9.4). 100 % local, no cloud.

    pixi run voice                      # listen all the time; act on sentences that start with "Nudge"
    pixi run voice --no-wake            # every sentence is a command (e.g. push-to-talk style testing)
    pixi run voice --file assets/audio/voice/come_here.wav   # publish a recorded clip (no mic needed)
    pixi run voice --text "Nudge, take me to the kitchen"    # publish typed text through the same path
    pixi run voice --selftest           # run the test clips through Whisper + wake word, print, exit
    pixi run voice --mic-test           # before recording: noise floor, mute / clipping, one spoken sentence

Pipeline: mic (PipeWire `pw-record` -> PulseAudio -> ALSA -> sounddevice; NUDGE_MIC to force one)
  -> Silero VAD (bundled with faster-whisper) cuts the stream into sentences: a sentence ends after
     0.8 s without speech (max 8 s); steady fan / hum noise is ignored
  -> Whisper base.en (faster-whisper, CTranslate2 int8 on the CPU, ~0.5 s per sentence; NUDGE_STT_MODEL)
  -> wake word: "Nudge, <command>" publishes <command>; "Nudge." alone arms the next sentence (6 s)
  -> ZMQ PUB bound on tcp://127.0.0.1:5558, topic `user_command` (the brain SUB-connects):
       {"text": "take me to the kitchen", "source": "voice", "confidence": 0.0-1.0,
        "raw": "Nudge, take me to the kitchen.", "t": unix_time}
     plus topic `voice_status` (2 Hz): {"state": "listening"|"hearing"|"transcribing"|"off",
        "level": mic RMS, "last_heard": last transcript (with or without wake word), "mic", "stt"}

Voice only says WHAT the owner wants; it is never used to locate the owner or to navigate.
Sentences starting with "Nudge says" are ignored (the robot's own text-to-speech through the laptop
speaker would otherwise give it commands).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "assets" / "models"
VOICE_CLIPS = PROJECT_ROOT / "assets" / "audio" / "voice"
# Same variable as the brain (rules.py VOICE_PUB); NUDGE_VOICE_ENDPOINT kept as an alias.
VOICE_ENDPOINT = os.environ.get("NUDGE_VOICE_PUB") or os.environ.get("NUDGE_VOICE_ENDPOINT", "tcp://127.0.0.1:5558")
SR = 16000
FRAME = 480  # 30 ms

# Sentence segmentation (Silero VAD)
END_SILENCE_S = 0.8
MAX_UTTERANCE_S = 8.0
MIN_SPEECH_S = 0.25
PRE_ROLL_S = 1.0
WAKE_ARM_S = 6.0  # "Nudge." alone -> the next sentence within 6 s is the command

# Biases Whisper toward the words Nudge understands (wake word, rooms, commands).
STT_PROMPT = ("Nudge, come here. Nudge, take me to the kitchen. Nudge, stop. Front door, back door, "
              "living room, bedroom, dining room, family room, hallway, garden, carpet.")
# Whisper sometimes spells the wake word differently.
WAKE_RE = re.compile(r"^\W*(?:(?:hey|ok|okay)\W+)?(?:nudge|nudges|nudge's|nudj|nuj|budge|nadge|nodge)\b\W*(.*)$",
                     re.IGNORECASE | re.DOTALL)
SELF_SPEECH_RE = re.compile(r"^\W*nudge\W+says\b", re.IGNORECASE)


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
    """Whisper (faster-whisper int8, CPU). Loads in the background; `transcribe` waits for it."""

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
            list(self.model.transcribe(np.zeros(SR // 2, np.float32), language="en", beam_size=1)[0])  # warm-up
        except Exception as exc:
            self.error = str(exc)
            print(f"[voice] speech-to-text unavailable: {exc}")
        self.ready.set()

    @property
    def backend(self) -> str:
        if not self.ready.is_set():
            return f"whisper-{self.name} (loading)"
        return f"whisper-{self.name}" if self.model is not None else "off"

    def transcribe(self, audio: np.ndarray) -> Tuple[str, float]:
        """-> (text, confidence 0..1 = exp(mean token log-prob))."""
        self.ready.wait(30.0)
        if self.model is None:
            raise RuntimeError(self.error or "speech-to-text model not loaded")
        segs, _ = self.model.transcribe(audio.astype(np.float32), language="en", beam_size=1,
                                        initial_prompt=STT_PROMPT, condition_on_previous_text=False,
                                        vad_filter=False)
        # Whisper invents text for noise ("Thank you."); drop segments it thinks are not speech.
        segs = [s for s in segs if s.no_speech_prob < 0.6]
        if not segs:
            return "", 0.0
        conf = float(np.exp(np.mean([s.avg_logprob for s in segs])))
        return " ".join(s.text.strip() for s in segs).strip(), round(min(1.0, conf), 2)


# ---------------------------------------------------------------------- sentence segmentation
class Segmenter:
    """Feed 30 ms frames; returns a finished sentence (float32 16 kHz) when the speaker pauses."""

    def __init__(self) -> None:
        from faster_whisper.vad import VadOptions  # type: ignore

        self.opts = VadOptions(threshold=0.5, min_speech_duration_ms=200, min_silence_duration_ms=300,
                               speech_pad_ms=200)
        self.frames: List[np.ndarray] = []
        self.speaking = False

    def feed(self, frame: np.ndarray) -> Optional[np.ndarray]:
        from faster_whisper.vad import get_speech_timestamps  # type: ignore

        self.frames.append(frame)
        if len(self.frames) % 8:  # run the VAD every ~0.25 s
            return None
        audio = np.concatenate(self.frames)
        speech = get_speech_timestamps(audio, self.opts)
        if not speech:
            self.speaking = False
            keep = int(PRE_ROLL_S * SR / FRAME)
            self.frames = self.frames[-keep:]
            return None
        self.speaking = True
        silent_after = (audio.size - speech[-1]["end"]) / SR
        too_long = (audio.size - speech[0]["start"]) / SR >= MAX_UTTERANCE_S
        if silent_after < END_SILENCE_S and not too_long:
            return None
        self.frames, self.speaking = [], False
        if sum(s["end"] - s["start"] for s in speech) / SR < MIN_SPEECH_S:
            return None
        return audio[max(0, speech[0]["start"] - SR // 5): speech[-1]["end"] + SR // 5]


def mic_problem(audio: np.ndarray) -> Optional[str]:
    if audio.size == 0 or float(np.max(np.abs(audio))) < 1e-4:
        return "mic muted or silent?"
    if float(np.mean(np.abs(audio) > 0.98)) > 0.01:
        return "mic is clipping: lower the mic boost / input volume"
    return None


# ---------------------------------------------------------------------- wake word
class WakeWord:
    """'Nudge, take me to the kitchen' -> 'take me to the kitchen'; 'Nudge.' arms the next sentence."""

    def __init__(self, required: bool = True) -> None:
        self.required = required
        self.armed_until = 0.0

    def command(self, text: str, now: Optional[float] = None) -> Optional[str]:
        now = time.monotonic() if now is None else now
        if not text or SELF_SPEECH_RE.match(text):
            return None
        m = WAKE_RE.match(text)
        if m:
            rest = m.group(1).strip()
            if re.sub(r"\W", "", rest):
                self.armed_until = 0.0
                return rest
            self.armed_until = now + WAKE_ARM_S  # just "Nudge." -> wait for the command
            return None
        if not self.required or now <= self.armed_until:
            self.armed_until = 0.0
            return text.strip()
        return None


# ---------------------------------------------------------------------- the voice process
class VoiceCommander:
    """Mic -> sentences -> Whisper -> wake word -> on_command(dict). Also keeps a status dict."""

    def __init__(self, on_command: Callable[[Dict[str, Any]], None], wake: bool = True, mic: str = "") -> None:
        self.on_command = on_command
        self.stt = SpeechToText()
        self.wake = WakeWord(required=wake)
        self.mic = make_mic(mic)
        self.status: Dict[str, Any] = {"state": "off", "level": 0.0, "last_heard": "",
                                       "mic": self.mic.name if self.mic else "none", "stt": self.stt.backend}
        self.sentences: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=4)
        self._stop = threading.Event()

    def start(self) -> None:
        threading.Thread(target=self._transcribe_loop, daemon=True).start()
        if self.mic is not None:
            threading.Thread(target=self._mic_loop, daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def handle_text(self, raw: str, confidence: float) -> Optional[Dict[str, Any]]:
        self.status["last_heard"] = raw
        cmd = self.wake.command(raw)
        if cmd is None:
            print(f"[voice] heard (no wake word): {raw!r}" if raw else "[voice] heard nothing intelligible")
            return None
        msg = {"text": cmd, "source": "voice", "confidence": confidence, "raw": raw, "t": time.time()}
        self.on_command(msg)
        return msg

    def _mic_loop(self) -> None:
        seg = Segmenter()
        warned_at = 0.0
        while not self._stop.is_set():
            try:
                self.mic.open()
                self.status["state"] = "listening"
                recent: List[np.ndarray] = []
                while not self._stop.is_set():
                    f = self.mic.read(FRAME)
                    if f is None:
                        raise RuntimeError(f"{self.mic.name} stopped delivering audio")
                    self.status["level"] = round(float(np.sqrt(np.mean(f * f))), 4)
                    recent = (recent + [f])[-100:]  # last 3 s, for the mic health check
                    if len(recent) == 100 and time.monotonic() - warned_at > 30:
                        problem = mic_problem(np.concatenate(recent))
                        if problem:
                            print(f"[voice] WARNING: {problem}")
                            warned_at = time.monotonic()
                    sentence = seg.feed(f)
                    if self.status["state"] != "transcribing":
                        self.status["state"] = "hearing" if seg.speaking else "listening"
                    if sentence is not None:
                        try:
                            self.sentences.put_nowait(sentence)
                        except queue.Full:
                            print("[voice] busy, dropped a sentence")
            except Exception as exc:
                print(f"[voice] mic error: {exc}; retrying in 2 s")
                self.status["state"] = "off"
                time.sleep(2.0)
            finally:
                self.mic.close()

    def _transcribe_loop(self) -> None:
        while not self._stop.is_set():
            try:
                audio = self.sentences.get(timeout=0.5)
            except queue.Empty:
                continue
            self.status["state"] = "transcribing"
            t0 = time.monotonic()
            try:
                text, conf = self.stt.transcribe(audio)
            except Exception as exc:
                print(f"[voice] speech-to-text failed: {exc}")
                continue
            finally:
                self.status["state"] = "listening"
                self.status["stt"] = self.stt.backend
            print(f"[voice] {audio.size / SR:.1f}s of speech -> {text!r} (conf {conf:.2f}, "
                  f"{time.monotonic() - t0:.2f}s)")
            self.handle_text(text, conf)

    def transcribe_file(self, path: str) -> Tuple[str, float]:
        return self.stt.transcribe(load_wav(path))


# ---------------------------------------------------------------------- ZMQ + CLI
class Publisher:
    def __init__(self, endpoint: str = VOICE_ENDPOINT) -> None:
        import zmq

        self.sock = zmq.Context.instance().socket(zmq.PUB)
        self.sock.setsockopt(zmq.LINGER, 1000)
        self.sock.bind(endpoint)
        self.lock = threading.Lock()

    def send(self, topic: str, msg: Dict[str, Any]) -> None:
        with self.lock:
            self.sock.send_multipart([topic.encode(), json.dumps(msg).encode()])

    def close(self) -> None:
        self.sock.close()


def selftest() -> int:
    """Test clips -> Whisper -> wake word; no mic, no ZMQ."""
    stt = SpeechToText()
    ok = True
    for clip in sorted(VOICE_CLIPS.glob("*.wav")):
        t0 = time.monotonic()
        text, conf = stt.transcribe(load_wav(str(clip)))
        dt = time.monotonic() - t0
        cmd = WakeWord(required=True).command(text)
        print(f"{clip.name:22} {dt:4.2f}s conf {conf:.2f}  {text!r:48} -> wake-word command: {cmd!r}")
        ok &= bool(text)
    for text, want in [("Nudge, take me to the kitchen.", "take me to the kitchen."), ("Take me to the kitchen.", None),
                       ("Nudge says: the doorbell is ringing.", None), ("Hey Nudge come here", "come here")]:
        got = WakeWord().command(text)
        print(f"wake word: {text!r:42} -> {got!r} {'ok' if got == want else 'WRONG'}")
        ok &= got == want
    w = WakeWord()
    two_step = (w.command("Nudge.", now=0.0), w.command("Stop.", now=2.0), w.command("Stop.", now=3.0))
    print(f"wake word: 'Nudge.' then 'Stop.' -> {two_step} {'ok' if two_step == (None, 'Stop.', None) else 'WRONG'}")
    ok &= two_step == (None, "Stop.", None)
    print("[voice] selftest", "passed" if ok else "FAILED")
    return 0 if ok else 1


def mic_test(mic_choice: str = "", say: str = "Nudge, take me to the kitchen.") -> int:
    """Pre-recording check (docs/DEMO_STORYBOARD.md "Mic tested"): noise floor, mute / clipping,
    then one spoken sentence -> Whisper -> wake word. Prints a verdict and the fix for each problem."""
    mic = make_mic(mic_choice)
    if mic is None:
        print("[mic-test] FAIL: no microphone backend (Linux: install pipewire / pulseaudio tools; or sounddevice)")
        return 1
    stt = SpeechToText()
    from scipy.signal import butter, sosfilt

    hp = butter(4, 100.0, "highpass", fs=SR, output="sos")  # speech band: ignore DC / laptop rumble

    def speech_rms(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(sosfilt(hp, x - np.mean(x))[SR // 10:] ** 2)))

    print(f"[mic-test] mic backend: {mic.name}  |  speech-to-text: whisper-{stt.name}")
    problems: List[str] = []
    try:
        mic.open()
        print("[mic-test] 1/2  stay QUIET for 2 seconds ...", flush=True)
        quiet = np.concatenate([mic.read(FRAME) for _ in range(int(2.0 * SR / FRAME))])
        floor = speech_rms(quiet)
        print(f"[mic-test]      noise floor RMS {floor:.4f} in the speech band (>100 Hz; raw RMS "
              f"{float(np.sqrt(np.mean(quiet ** 2))):.4f}, peak {float(np.max(np.abs(quiet))):.3f})")
        hint = mic_problem(quiet)
        if hint:
            problems.append(hint)
        if floor > 0.05:
            problems.append("very noisy room / mic gain too high (noise floor > 0.05)")
        print(f'[mic-test] 2/2  now SAY: "{say}"  (up to 8 s)', flush=True)
        seg, sentence, heard = Segmenter(), None, []
        t0 = time.monotonic()
        while sentence is None and time.monotonic() - t0 < 10.0:
            f = mic.read(FRAME)
            if f is None:
                problems.append(f"{mic.name} stopped delivering audio")
                break
            heard.append(f)
            sentence = seg.feed(f)
    finally:
        mic.close()
    if heard:
        hint = mic_problem(np.concatenate(heard))
        if hint and hint not in problems:
            problems.append(hint)
    if sentence is None:
        problems.append("no speech detected (too quiet, muted, or the wrong input device)")
    else:
        text, conf = stt.transcribe(sentence)
        cmd = WakeWord().command(text)
        level = speech_rms(sentence)
        print(f'[mic-test]      heard {sentence.size / SR:.1f}s, speech RMS {level:.3f}: "{text}" (confidence {conf:.2f})')
        print(f"[mic-test]      wake word -> command: {cmd!r}")
        if cmd is None:
            problems.append('wake word "Nudge" not recognised: start the sentence with "Nudge," and speak clearly')
        if conf < 0.5:
            problems.append(f"low confidence {conf:.2f} (< 0.5: the brain would ask to repeat) - move closer / reduce noise")
        if level < 2.5 * max(floor, 1e-4):
            problems.append("speech is barely above the noise floor - move closer to the mic")
    if problems:
        print("[mic-test] PROBLEMS:")
        for prob in problems:
            print(f"  - {prob}")
        print("[mic-test] fixes: unmute `wpctl set-mute @DEFAULT_AUDIO_SOURCE@ 0`; clipping -> "
              "`amixer -c <card> sset 'Internal Mic Boost' 0` (card from `arecord -l`); pick a device with "
              "--mic pipewire|pulse|alsa|sounddevice; Windows: Settings > Sound > Input volume")
        return 1
    print("[mic-test] OK - the mic is ready for `pixi run voice`")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-wake", action="store_true", help="treat every sentence as a command")
    ap.add_argument("--file", action="append", help="publish a recorded WAV clip instead of using the mic")
    ap.add_argument("--text", help="publish this text as a voice command (no mic / Whisper)")
    ap.add_argument("--mic", default="", help="pipewire | pulse | alsa | sounddevice (default: first that works)")
    ap.add_argument("--endpoint", default=VOICE_ENDPOINT)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mic-test", action="store_true", help="check the microphone + Whisper before recording a demo")
    args = ap.parse_args()
    if args.mic_test:
        return mic_test(args.mic)
    if args.selftest:
        return selftest()

    pub = Publisher(args.endpoint)

    def publish(msg: Dict[str, Any]) -> None:
        pub.send("user_command", msg)
        print(f"[voice] -> user_command {json.dumps(msg)}", flush=True)

    if args.text or args.file:
        time.sleep(0.5)  # PUB/SUB slow joiner: let subscribers connect before the one-shot message
        vc = VoiceCommander(publish, wake=not args.no_wake, mic="off")
        if args.text:
            vc.handle_text(args.text, 1.0)
        for f in args.file or []:
            vc.handle_text(*vc.transcribe_file(f))
        time.sleep(0.3)
        pub.close()
        return 0

    vc = VoiceCommander(publish, wake=not args.no_wake, mic=args.mic)
    if vc.mic is None:
        print("[voice] no microphone backend found (install PipeWire/PulseAudio tools or use --file/--text)")
        return 1
    vc.start()
    print(f"[voice] publishing user_command on {args.endpoint} | mic {vc.mic.name} | "
          + ("say 'Nudge, ...'" if not args.no_wake else "every sentence is a command"), flush=True)
    try:
        while True:
            vc.status["stt"] = vc.stt.backend
            pub.send("voice_status", vc.status)
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        vc.stop()
        pub.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
