#!/usr/bin/env python3
"""Synthesize the test clips for UNKNOWN sounds (P4, `nudge-brain/unknown_events.py`).

Writes 16 kHz mono WAVs to assets/audio/unknown/: glass_break, baby_cry, crash, water_running.
`dog_bark()` is kept but NOT written: YAMNet does not hear it as a dog, and its harmonics fire the
spectral doorbell rule (0.75) = a false doorbell mission (finding for hearing.py).
All synthesized here from numpy (no recordings, no licence issues; ESC-50 stays out of git).
Deterministic (fixed seeds). `--check` prints YAMNet's top-3 classes per clip (assets/models/yamnet.tflite).

    pixi run python scripts/make_unknown_clips.py [--check]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "audio" / "unknown"
SR = 16000


def _env(n: int, attack: float, decay: float) -> np.ndarray:
    t = np.arange(n) / SR
    return np.minimum(1.0, t / max(attack, 1e-4)) * np.exp(-t / decay)


def glass_break(rng) -> np.ndarray:
    """Impact crack + many inharmonic high 'shard' rings falling over ~1.5 s."""
    n = int(2.5 * SR)
    y = np.zeros(n)
    crack = rng.standard_normal(int(0.08 * SR)) * _env(int(0.08 * SR), 0.0005, 0.02)
    crack = signal.sosfilt(signal.butter(4, 1500, "hp", fs=SR, output="sos"), crack)
    y[:len(crack)] += 1.2 * crack
    for _ in range(70):  # shards hitting the floor
        t0 = int(rng.uniform(0.0, 1.6) ** 1.5 * SR)
        m = int(rng.uniform(0.05, 0.35) * SR)
        tt = np.arange(m) / SR
        ring = np.zeros(m)
        for _k in range(3):
            # skip 4.6-5.8 kHz: the oven-timer spectral rule (5.18-5.26 kHz tone) fired on glass shards there
            f = rng.uniform(2500, 4600) if rng.random() < 0.6 else rng.uniform(5800, 7500)
            ring += np.sin(2 * np.pi * f * tt + rng.uniform(0, 6.3))
        ring *= _env(m, 0.0003, rng.uniform(0.02, 0.09)) * rng.uniform(0.1, 0.5)
        tick = rng.standard_normal(m) * _env(m, 0.0002, 0.004) * 0.6
        end = min(n, t0 + m)
        y[t0:end] += (ring + tick)[: end - t0]
    return y


def dog_bark(rng) -> np.ndarray:
    """'Woof' = 0.2-0.3 s rough sawtooth bursts (f0 ~ 450 -> 300 Hz, jitter) + breath noise through
    two broad vocal-tract resonances (~700 Hz, ~1.6 kHz), in 'wuf-wuf' pairs."""
    n = int(3.0 * SR)
    y = np.zeros(n)
    for s in (0.05, 0.38, 1.25, 1.55, 2.4):
        m = int(rng.uniform(0.22, 0.3) * SR)
        tt = np.arange(m) / SR
        f0 = np.interp(tt, [0, 0.04, tt[-1]], [380, rng.uniform(480, 540), 300])
        f0 *= 1 + 0.08 * signal.sosfilt(signal.butter(1, 120, "lp", fs=SR, output="sos"), rng.standard_normal(m)) * 5
        ph = np.cumsum(f0) / SR
        src = 2 * (ph % 1.0) - 1.0 + 0.8 * rng.standard_normal(m)
        out = np.zeros(m)
        for fc, q, g in ((700, 2.0, 1.0), (1600, 3.0, 0.7), (2800, 4.0, 0.3)):
            bb, aa = signal.iirpeak(fc, q, fs=SR)
            out += g * signal.lfilter(bb, aa, src)
        env = np.minimum(1.0, tt / 0.015) * np.exp(-np.maximum(0, tt - 0.05) / 0.08)
        t0 = int(s * SR)
        y[t0:t0 + m] += out * env
    return y


def baby_cry(rng) -> np.ndarray:
    """'Waah' segments: f0 ~ 420-520 Hz with vibrato, rich harmonics, formants ~1.1 / 2.8 kHz."""
    n = int(3.5 * SR)
    y = np.zeros(n)
    for s, d in ((0.0, 1.0), (1.25, 0.9), (2.4, 1.0)):
        m = int(d * SR)
        tt = np.arange(m) / SR
        f0 = 420 + 160 * np.sin(np.pi * tt / d) ** 0.7 + 25 * np.sin(2 * np.pi * 9 * tt)
        f0 = f0 * (1 + 0.02 * signal.sosfilt(signal.butter(1, 40, "lp", fs=SR, output="sos"), rng.standard_normal(m)) * 8)
        ph = 2 * np.pi * np.cumsum(f0) / SR
        v = sum((0.8 ** k) * np.sin(k * ph + 0.3 * k) for k in range(1, 18))
        v = v * (1 + 0.5 * np.sin(ph / 2))  # subharmonic roughness (screamy)
        v += 0.4 * signal.sosfilt(signal.butter(2, [1500, 5000], "bp", fs=SR, output="sos"), rng.standard_normal(m))
        v = (signal.sosfilt(signal.butter(2, [900, 1400], "bp", fs=SR, output="sos"), v)
             + 0.6 * signal.sosfilt(signal.butter(2, [2500, 3200], "bp", fs=SR, output="sos"), v) + 0.3 * v)
        env = np.minimum(1.0, tt / 0.06) * np.minimum(1.0, (d - tt) / 0.12)
        t0 = int(s * SR)
        y[t0:t0 + m] += v * env
    return y


def crash(rng) -> np.ndarray:
    """Something heavy falling: a low thud + a broadband clatter tail."""
    n = int(2.0 * SR)
    tt = np.arange(n) / SR
    thud = np.sin(2 * np.pi * 70 * tt) * np.exp(-tt / 0.12) * 1.5
    noise = rng.standard_normal(n) * np.exp(-tt / 0.35)
    noise = signal.sosfilt(signal.butter(2, [150, 5000], "bp", fs=SR, output="sos"), noise)
    y = thud + 1.2 * noise
    for _ in range(12):  # bits bouncing
        t0 = int(rng.uniform(0.2, 1.4) * SR)
        m = int(0.04 * SR)
        y[t0:t0 + m] += rng.standard_normal(m) * _env(m, 0.0005, 0.01) * rng.uniform(0.3, 0.8)
    return y


def water_running(rng) -> np.ndarray:
    """Tap left running: steady band-limited noise with bubbly amplitude flicker."""
    n = int(4.0 * SR)
    x = rng.standard_normal(n)
    x = signal.sosfilt(signal.butter(2, [400, 6000], "bp", fs=SR, output="sos"), x)
    flick = 1.0 + 0.35 * signal.sosfilt(signal.butter(2, 30, "lp", fs=SR, output="sos"), rng.standard_normal(n)) * 4
    return x * np.clip(flick, 0.3, 1.8)


CLIPS = {"glass_break": glass_break, "baby_cry": baby_cry, "crash": crash,
         "water_running": water_running}


def write_all() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for i, (name, fn) in enumerate(CLIPS.items()):
        y = fn(np.random.default_rng(100 + i))
        y = 0.6 * y / (np.max(np.abs(y)) + 1e-9)
        wavfile.write(str(OUT / f"{name}.wav"), SR, (y * 32767).astype(np.int16))
        print(f"wrote {OUT / (name + '.wav')} ({len(y) / SR:.1f} s)")


def check() -> None:
    sys.path.insert(0, str(ROOT / "nudge-brain"))
    sys.path.insert(0, str(ROOT))
    from hearing import MODEL_DIR, YAMNET_SAMPLES, _load_tflite_interpreter

    interp = _load_tflite_interpreter(MODEL_DIR / "yamnet.tflite")
    i_in, i_out = interp.get_input_details()[0]["index"], interp.get_output_details()[0]["index"]
    with open(MODEL_DIR / "yamnet_class_map.csv", newline="", encoding="utf-8") as f:
        names = [row[2] for row in list(csv.reader(f))[1:]]
    for name in CLIPS:
        _, d = wavfile.read(str(OUT / f"{name}.wav"))
        x = d.astype(np.float32) / 32768.0
        best = {}
        for s in range(0, max(1, len(x) - YAMNET_SAMPLES), SR // 2):
            w = np.zeros(YAMNET_SAMPLES, np.float32)
            seg = x[s:s + YAMNET_SAMPLES]
            w[:len(seg)] = seg
            interp.set_tensor(i_in, w)
            interp.invoke()
            sc = interp.get_tensor(i_out).reshape(-1)
            for j in np.argsort(sc)[-3:]:
                best[names[j]] = max(best.get(names[j], 0.0), float(sc[j]))
        top = sorted(best.items(), key=lambda kv: -kv[1])[:5]
        print(f"{name:14s} " + ", ".join(f"{k} {v:.2f}" for k, v in top))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only run YAMNet on the existing clips")
    a = ap.parse_args()
    if not a.check:
        write_all()
    check()
