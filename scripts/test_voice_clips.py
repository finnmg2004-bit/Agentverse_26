#!/usr/bin/env python3
"""
Offline check of the owner-voice path for TEACHING sentences (overnight R3): each test clip in
`assets/audio/voice_tests/` goes through the same steps as the mic in `nudge-brain/voice.py`
(Whisper with STT_PROMPT -> wake word -> normalize_times) and is compared with the text the brain
should receive. Prints word accuracy + transcription time per clip and the number of exact matches.

    pixi run python scripts/test_voice_clips.py            # transcribe + score the clips
    pixi run python scripts/test_voice_clips.py --make     # (re)make the clips with Piper TTS first

Clips: Piper TTS (`speech.tts`, LJSpeech voice, public domain). Needs `pixi run fetch-models`.
Pass: >= 4 of 5 teach sentences exact (after lower-casing and stripping punctuation).
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "nudge-brain")]

CLIP_DIR = ROOT / "assets" / "audio" / "voice_tests"
# (clip name, what the owner says (TTS text), what the brain must receive as user_command.text, kind)
CLIPS = [
    ("teach_bedtime", "Nudge, every night at ten pm, take me to bed.",
     "every night at 10 pm take me to bed", "teach"),
    ("teach_kettle", "Nudge, when the kettle boils, take me to the kitchen.",
     "when the kettle boils take me to the kitchen", "teach"),
    ("teach_pills", "Nudge, remind me to take my pills at half past nine.",
     "remind me to take my pills at 9:30", "teach"),
    ("teach_morning", "Nudge, every morning at seven am, come and find me.",
     "every morning at 7 am come and find me", "teach"),
    ("teach_oven", "Nudge, when the oven timer goes, remind me to switch it off.",
     "when the oven timer goes remind me to switch it off", "teach"),
    ("cmd_kitchen", "Nudge, take me to the kitchen.", "take me to the kitchen", "command"),
    ("cmd_come", "Nudge, come here.", "come here", "command"),
    ("cmd_forget", "Nudge, forget the bedtime task.", "forget the bedtime task", "command"),
]


def norm(text: str) -> list:
    """Lower-case words, punctuation stripped (keeps the ':' of times)."""
    return re.sub(r"[^\w:' ]+", " ", text.lower()).split()


def word_accuracy(hyp: list, ref: list) -> float:
    """1 - word edit distance / len(ref)."""
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return max(0.0, 1.0 - d[len(hyp)] / max(1, len(ref)))


def make_clips() -> None:
    from speech.tts import save_wav

    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    for name, say, _, _ in CLIPS:
        ok = save_wav(say, CLIP_DIR / f"{name}.wav")
        print(f"[make] {name}.wav {'ok' if ok else 'FAILED'}: {say}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--make", action="store_true", help="(re)make the clips with Piper TTS first")
    args = ap.parse_args()
    if args.make or not all((CLIP_DIR / f"{c[0]}.wav").exists() for c in CLIPS):
        make_clips()

    import voice

    stt = voice.SpeechToText()
    wake = voice.WakeWord(required=True)
    exact = {"teach": 0, "command": 0}
    total = {"teach": 0, "command": 0}
    accs, times = [], []
    for name, _, expected, kind in CLIPS:
        audio = voice.load_wav(str(CLIP_DIR / f"{name}.wav"))
        t0 = time.monotonic()
        raw, conf = stt.transcribe(audio)
        dt = time.monotonic() - t0
        cmd = wake.command(raw)
        text = voice.normalize_times(cmd) if cmd is not None else ""
        acc = word_accuracy(norm(text), norm(expected))
        ok = norm(text) == norm(expected)
        total[kind] += 1
        exact[kind] += ok
        accs.append(acc)
        times.append(dt)
        print(f"{'OK ' if ok else 'BAD'} {name:14s} acc {acc:4.0%}  {dt:4.2f}s  conf {conf:.2f}  "
              f"raw {raw!r} -> {text!r}" + ("" if ok else f"  (want {expected!r})"))
    print(f"[voice-clips] teach exact {exact['teach']}/{total['teach']}, commands exact "
          f"{exact['command']}/{total['command']}, mean word accuracy {sum(accs) / len(accs):.0%}, "
          f"mean {sum(times) / len(times):.2f}s per clip ({stt.backend})")
    return 0 if exact["teach"] >= 4 and exact["command"] == total["command"] else 1


if __name__ == "__main__":
    sys.exit(main())
