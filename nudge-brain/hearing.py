#!/usr/bin/env python3
"""
Milestone 6a: Local Acoustic Perception (`nudge-brain/hearing.py`).
Performs:
  1. 16 kHz audio classification of every sound in the registry `SOUND_SOURCES`
     (`doorbell`, `knock`, `smoke_alarm`, `kettle`) or `silence`
     using local YAMNet TFLite (`assets/models/yamnet.tflite`, get it with `pixi run fetch-models`)
     fused with a multi-band spectral classifier fallback. YAMNet is multi-label, so
     simultaneous sounds (e.g. `smoke_alarm` pre-empting `doorbell`) are both scored.
  2. Stereo ear localization via Interaural Level Difference (ILD) and GCC-PHAT
     Interaural Time Difference (ITD) to estimate relative & world azimuth of the sound.
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from scipy import signal

from sim.house_layout import SOUND_SOURCES


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "assets" / "models"
SAMPLE_RATE = 16000
YAMNET_SAMPLES = 15600  # 0.975 s input window of the YAMNet TFLite model

# Nudge sound -> YAMNet (AudioSet) display names from yamnet_class_map.csv (registry field `yamnet`).
# Score = max over the list. There is no "kettle" class: Whistle / Steam whistle are the closest.
YAMNET_CLASSES: Dict[str, Tuple[str, ...]] = {
    name: tuple(spec.get("yamnet", ())) for name, spec in SOUND_SOURCES.items() if spec.get("yamnet")
}
SOUNDS: Tuple[str, ...] = tuple(SOUND_SOURCES)
KETTLE_TONAL_GATE = 8.0
# Knock = short wooden bursts. Burstiness = p95 / median energy of 10 ms frames in 110-600 Hz.
# Party music (sustained chords at 175-392 Hz, inside the knock band) used to fire the knock rule
# in 64 % of windows with nothing knocking; real knocks are >= 279 in a quiet room. Gate at 4:
# 0 false knocks under every noise profile; knocks still heard from the entry / front door /
# dining room under fan, TV, vacuum and party noise (only far rooms lose them under fan noise).
KNOCK_BURST_GATE = 4.0
# YAMNet hears a loud kettle whistle as "Alarm" / "Fire alarm" (up to 0.85 next to the stove). When the
# spectral kettle rule fires and the spectral smoke rule does NOT confirm a 3.1 kHz alarm, YAMNet's
# smoke score is capped below the 0.6 pre-emption threshold. A real smoke alarm (also with the kettle
# on) is still caught by the spectral smoke rule.
YAMNET_SMOKE_CAP_WITH_KETTLE = 0.5


def zero_scores() -> Dict[str, float]:
    return {name: 0.0 for name in SOUNDS}


def _load_tflite_interpreter(model_path: Path):
    """LiteRT (ai-edge-litert, Windows + Linux wheels) first, then legacy tflite_runtime / TensorFlow."""
    try:
        from ai_edge_litert.interpreter import Interpreter  # type: ignore
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter  # type: ignore
        except ImportError:
            from tensorflow.lite import Interpreter  # type: ignore
    interp = Interpreter(model_path=str(model_path), num_threads=2)  # leave cores for sim / Whisper / YOLO
    interp.allocate_tensors()
    return interp


class StereoHearing:
    """Local YAMNet (TFLite) sound classifier + spectral fallback + Stereo ILD/ITD localizer.

    YAMNet runs on a rolling ~1 s mono buffer (two 0.5 s audio windows). The hand-tuned log-mel
    band classifier is kept as a fallback for sounds YAMNet does not recognise (e.g. synthesized
    placeholder clips); each sound's score is the max of the two and `sources` says which one won.
    Set NUDGE_HEARING_FALLBACK=0 to run YAMNet-only.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate
        self._knock_sos = signal.butter(4, [110.0, 600.0], btype="bandpass", fs=sample_rate, output="sos")
        self.backend_name = "LogMel-spectral"
        self.allow_spectral = os.environ.get("NUDGE_HEARING_FALLBACK", "1") != "0"

        self.tflite_interpreter = None
        self.yamnet_names: list = []
        self.yamnet_index: Dict[str, list] = {}
        self._buf = np.zeros(YAMNET_SAMPLES, dtype=np.float32)
        yamnet_path = MODEL_DIR / "yamnet.tflite"
        class_map = MODEL_DIR / "yamnet_class_map.csv"
        if yamnet_path.exists() and class_map.exists():
            try:
                self.tflite_interpreter = _load_tflite_interpreter(yamnet_path)
                self._in = self.tflite_interpreter.get_input_details()[0]["index"]
                self._out = self.tflite_interpreter.get_output_details()[0]["index"]
                with open(class_map, newline="", encoding="utf-8") as f:
                    self.yamnet_names = [row[2] for row in list(csv.reader(f))[1:]]
                for sound, names in YAMNET_CLASSES.items():
                    self.yamnet_index[sound] = [self.yamnet_names.index(n) for n in names]
                self.backend_name = "YAMNet-TFLite" + ("+spectral" if self.allow_spectral else "")
            except Exception as exc:
                print(f"[hearing] YAMNet unavailable ({exc}); using spectral classifier")
                self.tflite_interpreter = None
        else:
            print("[hearing] assets/models/yamnet.tflite missing (run `pixi run fetch-models`); "
                  "using spectral classifier")
        print(f"[hearing] Sound classifier backend: {self.backend_name}")

    def _yamnet(self) -> Tuple[Dict[str, float], str, float]:
        """Run YAMNet on the rolling buffer -> (mapped scores, top-1 class name, top-1 score)."""
        self.tflite_interpreter.set_tensor(self._in, self._buf)
        self.tflite_interpreter.invoke()
        s = self.tflite_interpreter.get_tensor(self._out).reshape(-1)
        mapped = {sound: float(max(s[i] for i in idx)) for sound, idx in self.yamnet_index.items()}
        top = int(np.argmax(s))
        return mapped, self.yamnet_names[top], float(s[top])

    def _burstiness(self, mono: np.ndarray, frame: int = 160) -> float:
        """p95 / median energy of 10 ms frames in the knock band (110-600 Hz)."""
        y = signal.sosfilt(self._knock_sos, mono)
        n = (len(y) // frame) * frame
        if n < frame * 8:
            return 0.0
        e = np.mean(y[:n].reshape(-1, frame) ** 2, axis=1)
        return float(np.percentile(e, 95) / (np.median(e) + 1e-14))

    def _band_energy(
        self, freqs: np.ndarray, psd: np.ndarray, f_low: float, f_high: float
    ) -> float:
        mask = (freqs >= f_low) & (freqs <= f_high)
        if not np.any(mask):
            return 0.0
        return float(np.sum(psd[mask]))

    def estimate_azimuth(
        self, stereo_pcm: np.ndarray, robot_yaw: float
    ) -> Tuple[float, float, float]:
        """
        Estimate sound azimuth from stereo ear channels using ILD + ITD.
        Returns `(rel_azimuth_deg, world_azimuth_deg, ild_db)`.
        """
        left = stereo_pcm[:, 0].astype(np.float64)
        right = stereo_pcm[:, 1].astype(np.float64)

        rms_l = math.sqrt(float(np.mean(left * left)) + 1e-12)
        rms_r = math.sqrt(float(np.mean(right * right)) + 1e-12)
        ild_db = 20.0 * math.log10((rms_l + 1e-9) / (rms_r + 1e-9))

        # From synthesis: gain_l = G*(1 + 0.38*sin(az)), gain_r = G*(1 - 0.38*sin(az))
        # -> (rms_l - rms_r) / (rms_l + rms_r) = 0.38 * sin(az)
        ild_norm = (rms_l - rms_r) / (rms_l + rms_r + 1e-9)
        sin_az_ild = float(np.clip(ild_norm / 0.38, -1.0, 1.0))

        # Cross-correlation ITD over lags [-8, +8] samples
        max_lag = 8
        cc = signal.correlate(left, right, mode="full")
        mid = len(left) - 1
        window = cc[mid - max_lag : mid + max_lag + 1]
        best_lag = int(np.argmax(window)) - max_lag
        # Positive lag means left leads right -> sound on left (+azimuth)
        sin_az_itd = float(np.clip(-best_lag / 6.0, -1.0, 1.0))

        # Fuse ILD (80% weight, smooth & robust to phase aliasing) + ITD (20% weight)
        sin_az = float(np.clip(0.80 * sin_az_ild + 0.20 * sin_az_itd, -1.0, 1.0))
        rel_az_rad = math.asin(sin_az)
        world_az_rad = math.atan2(
            math.sin(robot_yaw + rel_az_rad), math.cos(robot_yaw + rel_az_rad)
        )
        return (
            round(math.degrees(rel_az_rad), 1),
            round(math.degrees(world_az_rad), 1),
            round(ild_db, 2),
        )

    def process_audio(
        self,
        stereo_pcm: np.ndarray,
        robot_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> Dict[str, Any]:
        """
        Classify a 0.5s 16 kHz stereo window and localize the active sound source.
        Returns:
          {
            "label": <a SOUND_SOURCES key> | "silence",
            "confidence": float,
            "scores": {sound: float for every SOUND_SOURCES key},
            "rel_azimuth_deg": float,
            "world_azimuth_deg": float,
            "estimated_room": str | None,
            "rms": float,
            "backend": str,
          }
        """
        if stereo_pcm.ndim == 1:
            stereo_pcm = np.stack([stereo_pcm, stereo_pcm], axis=-1)

        mono = 0.5 * (stereo_pcm[:, 0] + stereo_pcm[:, 1])
        rms = float(np.sqrt(np.mean(mono * mono)))
        peak = float(np.max(np.abs(mono))) + 1e-9

        # Keep the last ~1 s of audio for YAMNet (always updated, so silence flushes old sounds).
        chunk = mono[-YAMNET_SAMPLES:].astype(np.float32)
        self._buf = np.concatenate([self._buf[len(chunk):], chunk])

        if rms < 0.006:
            return {
                "label": "silence",
                "confidence": 0.0,
                "scores": zero_scores(),
                "rel_azimuth_deg": 0.0,
                "world_azimuth_deg": 0.0,
                "estimated_room": None,
                "rms": round(rms, 5),
                "backend": self.backend_name,
            }

        # Compute Welch PSD / FFT spectrum on the 16 kHz mono signal
        freqs, psd = signal.welch(mono, fs=self.sample_rate, nperseg=1024)
        total_energy = float(np.sum(psd)) + 1e-12

        # 1. Smoke Alarm signature: strong narrowband energy at 3000-3400 Hz (+ 6100-6500 Hz)
        e_smoke = self._band_energy(freqs, psd, 2950.0, 3400.0) + self._band_energy(
            freqs, psd, 6100.0, 6500.0
        )
        # 2. Doorbell signature: C5 (523 Hz) & E5 (659 Hz) fundamentals + 2nd/3rd harmonics
        e_doorbell_fund = self._band_energy(freqs, psd, 480.0, 700.0)
        e_doorbell_harm = self._band_energy(freqs, psd, 1000.0, 1400.0) + self._band_energy(
            freqs, psd, 1500.0, 2050.0
        )
        e_doorbell = e_doorbell_fund + 0.8 * e_doorbell_harm

        # 3. Knock signature: low-mid wooden resonance (120-360 Hz) + impulsive crest factor
        e_knock_low = self._band_energy(freqs, psd, 110.0, 360.0)
        crest_factor = peak / (rms + 1e-9)
        impulse_bonus = min(1.5, max(0.5, (crest_factor - 2.2) / 2.0))
        e_knock = e_knock_low * impulse_bonus

        # Convert band energies into calibrated [0, 1] confidence scores.
        # Importantly, we score each band against both total energy AND absolute power
        # so that if `smoke_alarm` starts while `doorbell` is playing, both get high scores!
        smoke_ratio = e_smoke / total_energy
        smoke_abs = min(1.0, math.sqrt(e_smoke * 3000.0))  # calibrated for whole-house range (nerv house)
        # Narrowband prominence: the alarm tone towers over its neighbouring bands even when a
        # louder, closer sound (e.g. the doorbell right next to the robot) dominates total energy.
        e_smoke_nb = self._band_energy(freqs, psd, 2450.0, 2900.0) + self._band_energy(freqs, psd, 3450.0, 3900.0)
        smoke_prominent = e_smoke > 8.0 * e_smoke_nb + 1e-9
        gate = smoke_ratio > 0.18 or (smoke_prominent and smoke_ratio > 0.02)
        score_smoke = float(np.clip(max(smoke_ratio * 1.25, 0.85 * smoke_abs if gate else 0.0), 0.0, 0.99))

        doorbell_ratio = e_doorbell / total_energy
        score_doorbell = float(np.clip(doorbell_ratio * 1.20, 0.0, 0.98))
        if e_doorbell_fund < 0.08 * total_energy:
            score_doorbell *= 0.25

        knock_ratio = e_knock / total_energy
        score_knock = float(np.clip(knock_ratio * 1.15, 0.0, 0.97))
        if crest_factor < 2.6:
            score_knock *= 0.30
        knock_burst = self._burstiness(mono)
        if knock_burst <= KNOCK_BURST_GATE:
            score_knock *= 0.30

        # 4. Kettle signature: a pure whistle ~2.25 kHz. Tonality = strongest PSD bin in 2150-2400 Hz
        #    over the median bin of the neighbouring 1.7-2.9 kHz: >= 19 with the kettle on anywhere in
        #    the house under every noise profile (tv / vacuum / party included), <= 3.3 without it
        #    (doorbell, knock, smoke alarm, noise). Gate at 8; confidence grows with log tonality.
        k_band = (freqs >= 2150.0) & (freqs <= 2400.0)
        k_floor = ((freqs >= 1700.0) & (freqs < 2100.0)) | ((freqs > 2450.0) & (freqs <= 2900.0))
        kettle_tonal = float(np.max(psd[k_band]) / (np.median(psd[k_floor]) + 1e-14))
        kettle_gate = kettle_tonal > KETTLE_TONAL_GATE
        kettle_ratio = self._band_energy(freqs, psd, 2150.0, 2400.0) / total_energy
        score_kettle = 0.0
        if kettle_gate:
            score_kettle = float(np.clip(max(kettle_ratio * 1.1,
                                             0.55 + 0.15 * math.log10(kettle_tonal / KETTLE_TONAL_GATE)), 0.0, 0.97))

        spectral = {
            "doorbell": round(score_doorbell, 3),
            "knock": round(score_knock, 3),
            "smoke_alarm": round(score_smoke, 3),
            "kettle": round(score_kettle, 3),
        }
        for name in SOUNDS:  # a registry sound without a spectral rule relies on YAMNet alone
            spectral.setdefault(name, 0.0)

        # Fuse with YAMNet: per sound, take the stronger of YAMNet and (if allowed) the spectral rules.
        yamnet_scores: Dict[str, float] = {}
        yamnet_top: Tuple[str, float] = ("", 0.0)
        sources: Dict[str, str] = {}
        if self.tflite_interpreter is not None:
            yamnet_scores, top_name, top_score = self._yamnet()
            yamnet_top = (top_name, round(top_score, 3))
            scores = {}
            if kettle_gate and spectral["smoke_alarm"] < 0.3 and "smoke_alarm" in yamnet_scores:
                yamnet_scores["smoke_alarm"] = min(yamnet_scores["smoke_alarm"], YAMNET_SMOKE_CAP_WITH_KETTLE)
            for sound in SOUNDS:
                y = yamnet_scores.get(sound, 0.0)
                sp = spectral[sound] if self.allow_spectral else 0.0
                scores[sound] = round(max(y, sp), 3)
                sources[sound] = "yamnet" if y >= sp else "spectral"
            yamnet_scores = {k: round(v, 3) for k, v in yamnet_scores.items()}
        else:
            scores = dict(spectral)
            sources = {k: "spectral" for k in scores}

        # Emergency priority: if smoke_alarm >= 0.60, select smoke_alarm immediately
        if scores["smoke_alarm"] >= 0.60:
            best_label = "smoke_alarm"
            best_conf = scores["smoke_alarm"]
        else:
            best_label = max(scores, key=scores.get)
            best_conf = scores[best_label]
            if best_conf < 0.35:
                best_label = "silence"
                best_conf = 0.0

        rel_az, world_az, ild_db = self.estimate_azimuth(stereo_pcm, robot_pose[2])
        estimated_room = None
        if best_label in SOUND_SOURCES:
            estimated_room = str(SOUND_SOURCES[best_label]["room"])

        return {
            "label": best_label,
            "confidence": round(float(best_conf), 3),
            "scores": scores,
            "sources": sources,
            "yamnet_scores": yamnet_scores,
            "spectral_scores": spectral,
            "yamnet_top": yamnet_top,
            "rel_azimuth_deg": rel_az,
            "world_azimuth_deg": world_az,
            "ild_db": ild_db,
            "estimated_room": estimated_room,
            "rms": round(rms, 4),
            "backend": self.backend_name,
        }
