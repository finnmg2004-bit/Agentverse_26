#!/usr/bin/env python3
"""
Hearing evaluation: does Nudge alert on the right household sounds and ignore everything else?

Random, seeded trials. Each trial picks
  * a clip: one of the 5 registry sounds (our synthesized demo clips, the real Wikimedia recordings
    in assets/audio/real/, real door knocks from ESC-50) or one of 49 other ESC-50 classes
    (dog, footsteps, vacuum, glass breaking, baby crying, alarm clock, siren, ...) or background only,
  * where it plays: registry sounds at their installed spot (doorbell at the front door, smoke alarm
    on the kitchen ceiling, ...), other sounds at a random point in a random ground-floor room,
  * where the robot is: random point + heading anywhere on the ground floor,
  * a background-noise profile (quiet / fan / tv / vacuum / party),
then plays it through the sim's own ear model (`sim.sensors.SimSensors.synthesize_stereo_audio`:
distance + wall attenuation, ILD/ITD, ambient noise, ear AGC, int16 PCM) and feeds every 0.5 s
window to the brain's `hearing.StereoHearing.process_audio` exactly like `nudge-brain/main.py`.

What counts as an alert (the same tests `main.py` uses to start a mission):
  * smoke alarm: `safety.smoke_detected` (score >= 0.6, one window) -> pre-emption + evacuation
  * other registry sounds: label == sound and confidence >= NEW_SOUND_CONF (0.5)
`SoundEvents` starts (the debounced trigger for taught tasks) are recorded too.

Also: (d) YAMNet top-1 on the clean ESC-50 clips (all 2,000), mapped to AudioSet class names, to
show the classifier knows far more than our 5 sounds.

    pixi run fetch-esc50                       # once: ESC-50 (CC BY-NC 3.0) -> assets/datasets/
    pixi run hearing-eval                      # 300 trials, seed 0 -> docs/hearing-eval.md + chart
    pixi run hearing-eval --n 600 --seed 1     # more trials
    pixi run hearing-eval --no-esc50           # registry clips + background only (no download needed)

Raw per-trial results: assets/eval/hearing_trials.csv (gitignored).
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import signal as sps
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "nudge-brain")]

import sim.sensors as sensors  # noqa: E402
from sim.house_layout import ROOMS, SOUND_SOURCES, ALERT_SOUNDS, get_room_at  # noqa: E402
from sim.sensors import AmbientNoise, NOISE_NAMES, SimSensors, AUDIO_WINDOW_SAMPLES  # noqa: E402

with contextlib.redirect_stdout(io.StringIO()):
    from hearing import SoundEvents, StereoHearing  # noqa: E402
    import safety  # noqa: E402

NEW_SOUND_CONF = 0.5  # nudge-brain/main.py (importing main.py would start a brain)
SR = 16000
WIN_S = 0.5
DATASETS = Path(os.environ.get("NUDGE_DATASETS", ROOT / "assets" / "datasets"))
ESC50 = DATASETS / "ESC-50"
OUT_DIR = ROOT / "assets" / "eval"
REGISTRY = tuple(SOUND_SOURCES)
EVAL_KEY = "__eval__"

# ESC-50 class -> AudioSet display names (yamnet_class_map.csv) that count as a correct YAMNet top-1:
# the matching class(es) plus their direct AudioSet parent, never generic roots like "Animal".
ESC_TO_AUDIOSET = {
    "dog": ("Dog", "Bark", "Bow-wow", "Yip", "Howl", "Growling", "Whimper (dog)", "Canidae, dogs, wolves"),
    "rooster": ("Chicken, rooster", "Crowing, cock-a-doodle-doo", "Fowl"),
    "pig": ("Pig", "Oink"),
    "cow": ("Cattle, bovinae", "Moo"),
    "frog": ("Frog", "Croak"),
    "cat": ("Cat", "Meow", "Purr", "Caterwaul", "Hiss"),
    "hen": ("Chicken, rooster", "Cluck", "Fowl"),
    "insects": ("Insect", "Fly, housefly", "Bee, wasp, etc.", "Mosquito", "Buzz"),
    "sheep": ("Sheep", "Bleat"),
    "crow": ("Crow", "Caw"),
    "rain": ("Rain", "Raindrop", "Rain on surface"),
    "sea_waves": ("Waves, surf", "Ocean"),
    "crackling_fire": ("Fire", "Crackle"),
    "crickets": ("Cricket", "Insect"),
    "chirping_birds": ("Bird", "Bird vocalization, bird call, bird song", "Chirp, tweet"),
    "water_drops": ("Drip", "Raindrop", "Plop"),
    "wind": ("Wind", "Wind noise (microphone)", "Rustling leaves"),
    "pouring_water": ("Pour", "Water tap, faucet", "Fill (with liquid)", "Trickle, dribble", "Liquid", "Water",
                      "Sink (filling or washing)"),
    "toilet_flush": ("Toilet flush",),
    "thunderstorm": ("Thunderstorm", "Thunder", "Rain"),
    "crying_baby": ("Baby cry, infant cry", "Crying, sobbing"),
    "sneezing": ("Sneeze",),
    "clapping": ("Clapping", "Applause", "Hands"),
    "breathing": ("Breathing", "Pant", "Gasp", "Wheeze", "Snort", "Sigh"),
    "coughing": ("Cough", "Throat clearing"),
    "footsteps": ("Walk, footsteps", "Run", "Shuffle"),
    "laughing": ("Laughter", "Giggle", "Snicker", "Belly laugh", "Chuckle, chortle", "Baby laughter"),
    "brushing_teeth": ("Toothbrush", "Electric toothbrush"),
    "snoring": ("Snoring",),
    "drinking_sipping": ("Gargling", "Liquid", "Slosh", "Chewing, mastication"),
    "door_wood_knock": ("Knock", "Door", "Tap", "Wood"),
    "mouse_click": ("Clicking", "Tick"),
    "keyboard_typing": ("Typing", "Computer keyboard", "Typewriter"),
    "door_wood_creaks": ("Creak", "Door", "Squeak"),
    "can_opening": ("Burst, pop", "Hiss", "Clicking"),
    "washing_machine": ("Mechanisms", "Whir", "Rumble", "Vibration", "Hum"),
    "vacuum_cleaner": ("Vacuum cleaner",),
    "clock_alarm": ("Alarm clock", "Alarm", "Beep, bleep", "Buzzer", "Ringtone", "Telephone bell ringing"),
    "clock_tick": ("Tick", "Tick-tock", "Clock"),
    "glass_breaking": ("Shatter", "Glass", "Chink, clink", "Breaking", "Smash, crash"),
    "helicopter": ("Helicopter",),
    "chainsaw": ("Chainsaw",),
    "siren": ("Siren", "Civil defense siren", "Police car (siren)", "Ambulance (siren)",
              "Fire engine, fire truck (siren)", "Emergency vehicle"),
    "car_horn": ("Vehicle horn, car horn, honking", "Toot", "Air horn, truck horn"),
    "engine": ("Engine", "Idling", "Medium engine (mid frequency)", "Heavy engine (low frequency)",
               "Light engine (high frequency)", "Engine starting", "Accelerating, revving, vroom", "Vehicle",
               "Car", "Motor vehicle (road)"),
    "train": ("Train", "Rail transport", "Train wheels squealing", "Railroad car, train wagon", "Train horn",
              "Train whistle", "Clickety-clack"),
    "church_bells": ("Church bell", "Bell", "Change ringing (campanology)"),
    "airplane": ("Aircraft", "Aircraft engine", "Fixed-wing aircraft, airplane", "Jet engine", "Propeller, airscrew"),
    "fireworks": ("Fireworks", "Firecracker", "Explosion", "Burst, pop", "Boom"),
    "hand_saw": ("Sawing", "Tools", "Filing (rasp)"),
}
# ESC-50 classes that are a registry sound (played at that sound's installed spot, must alert as it).
ESC_IS_REGISTRY = {"door_wood_knock": "knock"}
# Indoor / household ESC-50 classes (the report groups false alerts by these vs outdoor / animals).
HOUSEHOLD = {"crying_baby", "sneezing", "clapping", "breathing", "coughing", "footsteps", "laughing",
             "brushing_teeth", "snoring", "drinking_sipping", "mouse_click", "keyboard_typing",
             "door_wood_creaks", "can_opening", "washing_machine", "vacuum_cleaner", "clock_alarm",
             "clock_tick", "glass_breaking", "pouring_water", "toilet_flush", "water_drops", "dog", "cat"}


# --------------------------------------------------------------------------- clips
def load_wav16(path: Path) -> np.ndarray:
    """Any WAV -> 16 kHz mono float32, peak-normalised to 0.9 (like the registry clips)."""
    sr, d = wavfile.read(str(path))
    if d.ndim > 1:
        d = d.mean(axis=1)
    x = d.astype(np.float32)
    if d.dtype == np.int16:
        x /= 32768.0
    elif d.dtype == np.int32:
        x /= 2147483648.0
    elif d.dtype == np.uint8:
        x = (x - 128.0) / 128.0
    if sr != SR:
        g = math.gcd(SR, sr)
        x = sps.resample_poly(x, SR // g, sr // g).astype(np.float32)
    peak = float(np.max(np.abs(x))) + 1e-9
    return (x / peak * 0.9).astype(np.float32)


def build_pool(use_esc50: bool):
    """-> (registry pool {sound: [(variant, name, clip)]}, distractor pool {esc_class: [(name, clip)]})."""
    reg = defaultdict(list)
    for name, spec in SOUND_SOURCES.items():
        reg[name].append(("synth", f"{spec.get('file', name + '.wav')}", load_wav16(ROOT / "assets/audio" / str(spec.get("file", f"{name}.wav")))))
        real = ROOT / "assets/audio/real" / str(spec.get("file", f"{name}.wav"))
        if real.exists():
            reg[name].append(("real", f"real/{real.name}", load_wav16(real)))
    dis = defaultdict(list)
    if use_esc50:
        meta = ESC50 / "meta" / "esc50.csv"
        if not meta.exists():
            sys.exit(f"[hearing-eval] {meta} missing: run `pixi run fetch-esc50` (or pass --no-esc50)")
        with open(meta, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            cat = r["category"]
            item = (r["filename"], ESC50 / "audio" / r["filename"])  # loaded lazily
            if cat in ESC_IS_REGISTRY:
                reg[ESC_IS_REGISTRY[cat]].append(("real-esc50", r["filename"], item[1]))
            else:
                dis[cat].append(item)
    return reg, dis


# --------------------------------------------------------------------------- the house
_ROOM_KEYS = [k for k in ROOMS]


def random_point(rng: np.random.Generator, inset: float = 0.5):
    while True:
        room = _ROOM_KEYS[rng.integers(len(_ROOM_KEYS))]
        x0, y0, x1, y1 = ROOMS[room]["rect"]
        x = float(rng.uniform(x0 + inset, x1 - inset))
        y = float(rng.uniform(y0 + inset, y1 - inset))
        if get_room_at(x, y) == room:
            return room, x, y


class Ears:
    """The sim's ear pipeline without a MuJoCo model: `SimSensors.synthesize_stereo_audio` only uses
    the clip table, the ambient noise generator, the clip cursor and the AGC gain."""

    def __init__(self, seed: int, noise: str) -> None:
        self.s = SimSensors.__new__(SimSensors)
        self.s.audio_clips = {}
        self.s.audio_cursor = 0
        self.s.ear_gain = 1.0
        self.s.rng = np.random.default_rng(seed)
        self.s.noise = AmbientNoise(self.s.rng)
        self.s.noise.set(noise)

    def window(self, pose, sound: str | None, clip: np.ndarray | None, cursor: int | None = None) -> np.ndarray:
        active = {}
        if sound is not None:
            self.s.audio_clips[sound] = clip
            active = {sound: True}
            if cursor is not None:      # ESC-50 clips play once from start to end (5 s), not as a 2 s loop
                self.s.audio_cursor = cursor
        pcm, _ = self.s.synthesize_stereo_audio(pose, active)
        # What nudge-brain/bridge.py hands to hearing: float32 stereo in [-1, 1].
        return np.frombuffer(pcm, np.int16).reshape(-1, 2).astype(np.float32) / 32768.0


def alerts_in(res) -> list:
    """Missions main.py would start on this window (smoke pre-emption + non-danger registry sounds)."""
    out = []
    if safety.smoke_detected(res):
        out.append("smoke_alarm")
    label, conf = res.get("label"), float(res.get("confidence", 0.0))
    if label in ALERT_SOUNDS and conf >= NEW_SOUND_CONF:
        out.append(label)
    return out


# --------------------------------------------------------------------------- one trial
def run_trial(hearing: StereoHearing, trial: dict, clip: np.ndarray | None) -> dict:
    ears = Ears(trial["seed"], trial["noise"])
    hearing._buf = np.zeros_like(hearing._buf)       # fresh rolling YAMNet buffer per trial
    ev = SoundEvents()
    pose = trial["pose"]
    expected = trial["expected"]                      # registry sound or None
    key = expected if expected else EVAL_KEY
    if not expected and clip is not None:
        sensors.SOUND_SOURCES[EVAL_KEY].update(pos=trial["src"], level=trial["level"])

    pre_windows = 4
    if expected and trial["variant"] != "real-esc50":
        on_windows, cursor_mode = int(trial["play_s"] / WIN_S), False   # loops like the sim (rings on)
        ears.s.audio_cursor = int(trial["seed"] % 4) * AUDIO_WINDOW_SAMPLES
    else:
        on_windows, cursor_mode = (int(math.ceil(len(clip) / AUDIO_WINDOW_SAMPLES)) if clip is not None else 10), True
    post_windows = 4

    alerts, ev_starts, tops, false_via = [], [], Counter(), Counter()
    first_alert_t, first_event_t, first_wrong = None, None, None
    max_score = 0.0
    t = 0.0
    n_total = pre_windows + on_windows + post_windows
    for w in range(n_total):
        on = pre_windows <= w < pre_windows + on_windows and clip is not None
        k = w - pre_windows
        x = ears.window(pose, key if on else None, clip,
                        cursor=(k * AUDIO_WINDOW_SAMPLES if cursor_mode and on else None))
        res = hearing.process_audio(x, pose)
        t_rel = (k + 1) * WIN_S                     # end of this window, seconds after the onset
        if on and res.get("yamnet_top", ("", 0))[0]:
            tops[res["yamnet_top"][0]] += 1
        if expected:
            max_score = max(max_score, float(res["scores"].get(expected, 0.0)))
        for a in alerts_in(res):
            alerts.append((a, round(t_rel, 1)))
            if a == expected and first_alert_t is None and w >= pre_windows:
                first_alert_t = t_rel
            elif a != expected:
                first_wrong = first_wrong or a
                false_via[res.get("sources", {}).get(a, "spectral")] += 1
        for e in ev.update(res, t):
            if e["phase"] == "start":
                ev_starts.append(e["sound"])
                if e["sound"] == expected and first_event_t is None and w >= pre_windows:
                    first_event_t = t_rel
        t += WIN_S
    wrong = sorted({a for a, _ in alerts if a != expected})
    return {
        "detected": first_alert_t is not None,
        "t_detect": first_alert_t,
        "event_detected": first_event_t is not None,
        "t_event": first_event_t,
        "wrong_alerts": wrong,
        "wrong_events": sorted({s for s in ev_starts if s != expected}),
        "false_via": ";".join(sorted(false_via)),
        "max_score": round(max_score, 3),
        "yamnet_top": tops.most_common(1)[0][0] if tops else "",
    }


# --------------------------------------------------------------------------- (d) YAMNet on clean clips
def yamnet_clean_accuracy(hearing: StereoHearing, dis_and_knock: dict) -> dict:
    """Top-1 / top-3 of YAMNet (scores averaged over 0.975 s frames, hop 0.5 s) per ESC-50 class."""
    names = hearing.yamnet_names
    idx = {c: {names.index(n) for n in ns} for c, ns in ESC_TO_AUDIOSET.items()}
    interp, i_in, i_out = hearing.tflite_interpreter, hearing._in, hearing._out
    per = {}
    for cat, items in sorted(dis_and_knock.items()):
        top1 = top3 = 0
        guesses = Counter()
        for _name, path in items:
            x = load_wav16(path)
            frames = []
            for s in range(0, max(1, len(x) - 15600 + 1), 8000):
                seg = x[s:s + 15600]
                if len(seg) < 15600:
                    seg = np.pad(seg, (0, 15600 - len(seg)))
                interp.set_tensor(i_in, seg.astype(np.float32))
                interp.invoke()
                frames.append(interp.get_tensor(i_out).reshape(-1).copy())
            mean = np.mean(frames, axis=0)
            order = np.argsort(mean)[::-1]
            top1 += int(order[0] in idx[cat])
            top3 += int(any(o in idx[cat] for o in order[:3]))
            guesses[names[order[0]]] += 1
        per[cat] = {"n": len(items), "top1": top1, "top3": top3, "guess": guesses.most_common(1)[0]}
    return per


# --------------------------------------------------------------------------- report
def pct(a, b):
    return f"{100.0 * a / b:.0f} %" if b else "-"


def draw_chart(path: Path, reg_rows, dis_rows, title: str) -> None:
    """Horizontal grouped bars: alert rate per sound, for the two triggers (series = trigger):
    blue = mission trigger in main.py today (one 0.5 s window), orange = SoundEvents (2 windows)."""
    import cv2

    rows = [("Registry sounds: Nudge must alert", None)] + reg_rows +            [("Other sounds: Nudge must stay quiet", None)] + dis_rows
    W, row_h, top, left, right = 1400, 46, 150, 430, 120
    H = top + sum(34 if v is None else row_h for _, v in rows) + 60
    img = np.full((H, W, 3), (251, 252, 252), np.uint8)      # surface #fcfcfb (BGR)
    ink, ink2, grid = (11, 11, 11), (78, 81, 82), (225, 225, 222)
    s1, s2 = (214, 120, 42), (52, 104, 235)                    # categorical slot 1 #2a78d6, slot 2 #eb6834
    font = cv2.FONT_HERSHEY_DUPLEX

    def text(s, x, y, scale=0.6, color=ink, thick=1):
        cv2.putText(img, s, (x, y), font, scale, color, thick, cv2.LINE_AA)

    text(title, 30, 48, 0.95, ink, 1)
    text("share of random trials that raised an alert (robot anywhere in the house, 5 noise profiles)", 30, 82, 0.55, ink2)
    # legend
    cv2.rectangle(img, (30, 104), (48, 118), s1, -1)
    text("alert trigger in main.py today (1 window)", 56, 117, 0.52, ink2)
    cv2.rectangle(img, (470, 104), (488, 118), s2, -1)
    text("debounced SoundEvents trigger (2 windows; smoke: 1)", 496, 117, 0.52, ink2)
    x0, x1 = left, W - right
    for p in (0, 25, 50, 75, 100):
        gx = int(x0 + (x1 - x0) * p / 100)
        cv2.line(img, (gx, top - 6), (gx, H - 56), grid, 1)
        text(f"{p}%", gx - 16, H - 32, 0.5, ink2)
    y = top
    for label, val in rows:
        if val is None:
            text(label, 30, y + 24, 0.62, ink, 1)
            y += 34
            continue
        r1, r2, n = val
        text(label, 46, y + 26, 0.55, ink2)
        text(f"n={n}", x0 - 70, y + 26, 0.45, ink2)
        for j, (rate, col) in enumerate(((r1, s1), (r2, s2))):
            yy = y + 6 + j * 18
            bw = int((x1 - x0) * rate)
            if bw > 0:
                cv2.rectangle(img, (x0, yy), (x0 + bw, yy + 14), col, -1)
            text(f"{100 * rate:.0f}%", x0 + bw + 8, yy + 13, 0.45, ink)
        y += row_h
    cv2.imwrite(str(path), img)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=300, help="number of random trials (default 300)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--registry-share", type=float, default=0.45, help="share of trials that play a registry sound")
    ap.add_argument("--background-share", type=float, default=0.05, help="share of background-only trials")
    ap.add_argument("--play-s", type=float, default=10.0, help="how long a registry sound rings (s)")
    ap.add_argument("--no-esc50", action="store_true", help="registry clips + background only")
    ap.add_argument("--no-clean", action="store_true", help="skip (d) YAMNet on the clean ESC-50 clips")
    ap.add_argument("--csv", default=str(OUT_DIR / "hearing_trials.csv"), help="per-trial rows (gitignored)")
    ap.add_argument("--report", default=str(ROOT / "docs" / "hearing-eval.md"))
    ap.add_argument("--chart", default=str(ROOT / "docs" / "img" / "hearing-eval.png"))
    args = ap.parse_args()

    with contextlib.redirect_stdout(io.StringIO()):
        hearing = StereoHearing()
    if hearing.tflite_interpreter is None:
        print("[hearing-eval] WARNING: YAMNet missing (pixi run fetch-models); spectral rules only")
    reg, dis = build_pool(not args.no_esc50)
    sensors.SOUND_SOURCES = dict(SOUND_SOURCES, **{EVAL_KEY: {"pos": (0.0, 0.0, 1.0), "level": 1.0}})

    rng = np.random.default_rng(args.seed)
    trials = []
    dis_classes = sorted(dis)
    for i in range(args.n):
        u = rng.random()
        _room, rx, ry = random_point(rng, inset=0.35)
        pose = (rx, ry, float(rng.uniform(-math.pi, math.pi)))
        noise = NOISE_NAMES[rng.integers(len(NOISE_NAMES))]
        tr = {"i": i, "seed": int(rng.integers(1 << 30)), "noise": noise, "pose": pose, "robot_room": _room}
        if u < args.registry_share or not dis_classes:
            sound = REGISTRY[rng.integers(len(REGISTRY))]
            variants = sorted({v for v, _, _ in reg[sound]})
            variant = variants[rng.integers(len(variants))]
            items = [it for it in reg[sound] if it[0] == variant]
            _v, name, clip = items[rng.integers(len(items))]
            src = SOUND_SOURCES[sound]["pos"]
            tr.update(kind="registry", expected=sound, variant=variant, clip=name, cls=sound,
                      src=src, src_room=SOUND_SOURCES[sound]["room"], level=float(SOUND_SOURCES[sound].get("level", 1.0)),
                      play_s=args.play_s, _clip=clip)
        elif u < args.registry_share + args.background_share:
            tr.update(kind="background", expected=None, variant="-", clip="-", cls="background only",
                      src=None, src_room="-", level=0.0, _clip=None)
        else:
            cls = dis_classes[rng.integers(len(dis_classes))]
            name, path = dis[cls][rng.integers(len(dis[cls]))]
            sroom, sx, sy = random_point(rng)
            level = float(np.exp(rng.uniform(math.log(0.5), math.log(2.0))))
            tr.update(kind="other", expected=None, variant="esc50", clip=name, cls=cls,
                      src=(sx, sy, float(rng.uniform(0.3, 2.0))), src_room=sroom, level=round(level, 2), _clip=path)
        trials.append(tr)

    t0 = time.monotonic()
    results = []
    for k, tr in enumerate(trials):
        clip = tr["_clip"]
        if isinstance(clip, Path):
            clip = load_wav16(clip)
        r = run_trial(hearing, tr, clip)
        results.append({**{k2: v for k2, v in tr.items() if not k2.startswith("_")}, **r})
        if (k + 1) % 50 == 0:
            print(f"[hearing-eval] {k + 1}/{len(trials)} trials ({time.monotonic() - t0:.0f} s)", flush=True)
    elapsed = time.monotonic() - t0

    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        cols = ["i", "kind", "cls", "variant", "clip", "expected", "noise", "robot_room", "src_room", "level",
                "detected", "t_detect", "event_detected", "t_event", "wrong_alerts", "wrong_events", "false_via", "max_score",
                "yamnet_top", "pose"]
        wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        for r in results:
            wr.writerow({**r, "wrong_alerts": ";".join(r["wrong_alerts"]), "wrong_events": ";".join(r["wrong_events"]),
                         "pose": "(%.2f, %.2f, %.2f)" % r["pose"]})

    clean = None
    if not args.no_esc50 and not args.no_clean and hearing.tflite_interpreter is not None:
        print("[hearing-eval] (d) YAMNet on the 2,000 clean ESC-50 clips ...", flush=True)
        esc_all = dict(dis)
        esc_all["door_wood_knock"] = [(n, p) for v, n, p in reg["knock"] if v == "real-esc50"]
        clean = yamnet_clean_accuracy(hearing, esc_all)

    write_report(args, results, clean, elapsed, hearing.backend_name)
    return 0


def write_report(args, results, clean, elapsed, backend) -> None:
    reg = [r for r in results if r["kind"] == "registry"]
    oth = [r for r in results if r["kind"] != "registry"]
    L = []
    L.append("# Hearing evaluation: right sounds in, everything else out\n")
    L.append(f"Generated by `pixi run hearing-eval --n {args.n} --seed {args.seed}` "
             f"({len(results)} random trials, {elapsed:.0f} s on CPU, backend `{backend}`, "
             f"env NUDGE_REAL_AUDIO={os.environ.get('NUDGE_REAL_AUDIO', '0')}, "
             f"NUDGE_SMOKE_TONAL={os.environ.get('NUDGE_SMOKE_TONAL', '(default = NUDGE_REAL_AUDIO)')}).\n")
    L.append("Each trial: a random clip, a random robot pose anywhere on the ground floor, one of the 5 "
             "background-noise profiles; the clip goes through the sim's ear model (distance + wall attenuation, "
             "ILD/ITD, noise, AGC) and every 0.5 s window through `hearing.process_audio`, as in the running "
             "robot. **Alert** = what `main.py` acts on: smoke score >= 0.6 (safety pre-emption) or a non-danger "
             "registry sound as the top label with confidence >= 0.5. Registry sounds ring for "
             f"{args.play_s:.0f} s at their installed spot; ESC-50 clips play once (5 s) at a random spot in a random "
             "room, 0.5-2x loudness. Dataset: **ESC-50** (K. J. Piczak, CC BY-NC 3.0, 2,000 clips, 50 classes; "
             "downloaded by `pixi run fetch-esc50`, not in git). Real registry recordings: Wikimedia Commons "
             "(CC0 / public domain, see README).\n")

    # headline
    syn = [r for r in reg if r["variant"] == "synth"]
    smoke = [r for r in reg if r["expected"] == "smoke_alarm"]
    smoke_syn = [r for r in smoke if r["variant"] == "synth"]
    false_oth = [r for r in oth if r["wrong_alerts"]]
    false_reg = [r for r in reg if r["wrong_alerts"]]
    L.append("## Headline\n")
    L.append("| | result |\n|---|---|")
    L.append(f"| (a) Registry sounds, demo clips (synthesized): alerted | **{pct(sum(r['detected'] for r in syn), len(syn))}** "
             f"({sum(r['detected'] for r in syn)}/{len(syn)}) |")
    L.append(f"| (a) Registry sounds, all clips incl. real recordings | {pct(sum(r['detected'] for r in reg), len(reg))} "
             f"({sum(r['detected'] for r in reg)}/{len(reg)}) |")
    L.append(f"| (b) Other sounds + background: false alert | **{pct(len(false_oth), len(oth))}** ({len(false_oth)}/{len(oth)}) |")
    ev_oth = [r for r in oth if r["wrong_events"]]
    L.append(f"| (b) same, with the 2-window debounce of `SoundEvents` | {pct(len(ev_oth), len(oth))} ({len(ev_oth)}/{len(oth)}); "
             f"registry demo clips then detected {pct(sum(r['event_detected'] for r in syn), len(syn))} |")
    L.append(f"| (b) Registry trials: alert for the wrong sound | {pct(len(false_reg), len(reg))} ({len(false_reg)}/{len(reg)}) |")
    L.append(f"| (c) Smoke alarm missed, demo clip | **{pct(sum(not r['detected'] for r in smoke_syn), len(smoke_syn))}** "
             f"({sum(not r['detected'] for r in smoke_syn)}/{len(smoke_syn)}) |")
    L.append(f"| (c) Smoke alarm missed, all clips | {pct(sum(not r['detected'] for r in smoke), len(smoke))} "
             f"({sum(not r['detected'] for r in smoke)}/{len(smoke)}) |")
    if clean:
        n = sum(v["n"] for v in clean.values())
        t1 = sum(v["top1"] for v in clean.values())
        t3 = sum(v["top3"] for v in clean.values())
        L.append(f"| (d) YAMNet top-1 on 2,000 clean ESC-50 clips (50 classes) | **{pct(t1, n)}** (top-3 {pct(t3, n)}) |")
    L.append("")

    # (a)
    L.append("## (a) Registry sounds: detection rate and time to alert\n")
    L.append("| sound | clip | trials | alerted | median time to alert | slowest | SoundEvents start |")
    L.append("|---|---|---|---|---|---|---|")
    chart_reg = []
    for s in REGISTRY:
        for v in ("synth", "real", "real-esc50"):
            rs = [r for r in reg if r["expected"] == s and r["variant"] == v]
            if not rs:
                continue
            ts = [r["t_detect"] for r in rs if r["detected"]]
            vname = {"synth": "synthesized (demo)", "real": "real (Wikimedia)", "real-esc50": "real (ESC-50 knocks)"}[v]
            L.append(f"| {SOUND_SOURCES[s]['label']} | {vname} | {len(rs)} | {pct(len(ts), len(rs))} | "
                     f"{(f'{np.median(ts):.1f} s' if ts else '-')} | {(f'{max(ts):.1f} s' if ts else '-')} | "
                     f"{pct(sum(r['event_detected'] for r in rs), len(rs))} |")
            short = {"synth": "demo clip", "real": "real recording", "real-esc50": "real, ESC-50"}[v]
            chart_reg.append((f"{SOUND_SOURCES[s]['label']} ({short})",
                              (len(ts) / len(rs), sum(r["event_detected"] for r in rs) / len(rs), len(rs))))
    L.append("")
    L.append("By noise profile (all registry trials, alerted / trials):\n")
    L.append("| sound | " + " | ".join(NOISE_NAMES) + " |")
    L.append("|---|" + "---|" * len(NOISE_NAMES))
    for s in REGISTRY:
        for v in ("synth", "real"):
            cells = []
            for nz in NOISE_NAMES:
                rs = [r for r in reg if r["expected"] == s and r["noise"] == nz
                      and (r["variant"] == v or (v == "real" and r["variant"] == "real-esc50"))]
                cells.append(f"{sum(r['detected'] for r in rs)}/{len(rs)}" if rs else "-")
            if any(c != "-" for c in cells):
                L.append(f"| {SOUND_SOURCES[s]['label']} ({'demo' if v == 'synth' else 'real'}) | " + " | ".join(cells) + " |")
    L.append("")

    # (b)
    L.append("## (b) Everything else: false alerts\n")
    by_cls = defaultdict(list)
    for r in oth:
        by_cls[r["cls"]].append(r)
    L.append(f"{len(by_cls)} sound classes, {len(oth)} trials. Classes that triggered an alert:\n")
    bad = [(c, rs) for c, rs in sorted(by_cls.items()) if any(r["wrong_alerts"] for r in rs)]
    if bad:
        L.append("| sound | trials | false alerts | alerted as | noise of the false ones |")
        L.append("|---|---|---|---|---|")
        for c, rs in bad:
            fa = [r for r in rs if r["wrong_alerts"]]
            L.append(f"| {c} | {len(rs)} | {len(fa)} | {', '.join(sorted(Counter(a for r in fa for a in r['wrong_alerts'])))} | "
                     f"{', '.join(sorted({r['noise'] for r in fa}))} |")
    else:
        L.append("**None.**")
    L.append("")
    via = Counter(v for r in oth for v in r["false_via"].split(";") if v)
    if via:
        L.append("Which detector fired the false alerts (trials; a trial can count for both): "
                 + ", ".join(f"{k} {n}" for k, n in via.most_common()) + ".\n")
    quiet = sorted(c for c, rs in by_cls.items() if not any(r["wrong_alerts"] for r in rs))
    L.append(f"No alert ({len(quiet)} classes): " + ", ".join(quiet) + ".\n")
    ev_false = [r for r in oth if r["wrong_events"]]
    L.append(f"`SoundEvents` (taught-task trigger) false starts on these trials: {len(ev_false)}/{len(oth)}"
             + (": " + ", ".join(f"{r['cls']} -> {'/'.join(r['wrong_events'])}" for r in ev_false[:12]) if ev_false else "") + ".\n")
    if false_reg:
        L.append("Registry trials that also alerted for another sound: "
                 + ", ".join(f"{r['expected']} ({r['variant']}, {r['noise']}) -> {'/'.join(r['wrong_alerts'])}" for r in false_reg[:12]) + ".\n")

    # (c)
    L.append("## (c) Smoke alarm (safety-critical)\n")
    for v in ("synth", "real"):
        rs = [r for r in smoke if r["variant"] == v]
        if not rs:
            continue
        miss = [r for r in rs if not r["detected"]]
        ts = [r["t_detect"] for r in rs if r["detected"]]
        L.append(f"- **{'Demo clip (synthesized 3.15 kHz)' if v == 'synth' else 'Real recording (2.8 kHz)'}**: "
                 f"missed {len(miss)}/{len(rs)} ({pct(len(miss), len(rs))}); pre-emption after "
                 f"{(f'median {np.median(ts):.1f} s, at most {max(ts):.1f} s' if ts else '-')}."
                 + (" Missed in: " + "; ".join(f"robot in {r['robot_room']}, {r['noise']} noise (max score {r['max_score']:.2f})"
                                                for r in miss[:8]) if miss else ""))
    L.append("")

    # (d)
    if clean:
        L.append("## (d) YAMNet recognises far more than our 5 sounds\n")
        L.append("YAMNet (521 AudioSet classes) on every clean ESC-50 clip, scores averaged over the clip. A hit = "
                 "the top class is one of the AudioSet classes mapped to that ESC-50 class (incl. its direct parent; "
                 "mapping in `scripts/test_hearing_eval.py`). Household classes first.\n")
        L.append("| ESC-50 class | top-1 | top-3 | YAMNet's most common answer |")
        L.append("|---|---|---|---|")
        order = sorted(clean, key=lambda c: (c not in HOUSEHOLD and c != "door_wood_knock", -clean[c]["top1"], c))
        for c in order:
            v = clean[c]
            L.append(f"| {c}{' (household)' if c in HOUSEHOLD or c == 'door_wood_knock' else ''} | "
                     f"{pct(v['top1'], v['n'])} | {pct(v['top3'], v['n'])} | {v['guess'][0]} ({v['guess'][1]}/{v['n']}) |")
        hh = [c for c in clean if c in HOUSEHOLD or c == "door_wood_knock"]
        L.append(f"\nHousehold classes: top-1 {pct(sum(clean[c]['top1'] for c in hh), sum(clean[c]['n'] for c in hh))}. "
                 "Through the robot's ears (house acoustics + noise) the per-trial YAMNet top class is in "
                 "`assets/eval/hearing_trials.csv`.\n")

    # confusion
    L.append("## Confusion table (trials -> what Nudge alerted)\n")
    cols = list(REGISTRY) + ["no alert"]
    L.append("| played \\ alerted | " + " | ".join(cols) + " |")
    L.append("|---|" + "---|" * len(cols))
    groups = [(f"{SOUND_SOURCES[s]['label']}", [r for r in reg if r["expected"] == s]) for s in REGISTRY]
    groups += [("other household sounds", [r for r in oth if r["cls"] in HOUSEHOLD]),
               ("other outdoor / animal sounds", [r for r in oth if r["kind"] == "other" and r["cls"] not in HOUSEHOLD]),
               ("background only", [r for r in oth if r["kind"] == "background"])]
    for g, rs in groups:
        if not rs:
            continue
        cnt = Counter()
        for r in rs:
            alerted = set(r["wrong_alerts"]) | ({r["expected"]} if r["detected"] else set())
            for a in alerted:
                cnt[a] += 1
            if not alerted:
                cnt["no alert"] += 1
        L.append(f"| {g} (n={len(rs)}) | " + " | ".join(str(cnt[c]) if cnt[c] else "." for c in cols) + " |")
    L.append("\n(A trial can alert more than one sound, so a row can add up to more than n.)\n")

    # chart
    chart_dis = []
    hh = [r for r in oth if r["cls"] in HOUSEHOLD]
    od = [r for r in oth if r["kind"] == "other" and r["cls"] not in HOUSEHOLD]
    bg = [r for r in oth if r["kind"] == "background"]
    for name, rs in (("Household (24 ESC-50 classes)", hh), ("Outdoor / animals (25 classes)", od),
                     ("Background noise only", bg)):
        if rs:
            chart_dis.append((name, (sum(bool(r["wrong_alerts"]) for r in rs) / len(rs),
                                     sum(bool(r["wrong_events"]) for r in rs) / len(rs), len(rs))))
    chart = Path(args.chart)
    chart.parent.mkdir(parents=True, exist_ok=True)
    draw_chart(chart, chart_reg, chart_dis, "Nudge hearing: how often each kind of sound raises an alert")
    try:
        rel = os.path.relpath(chart, Path(args.report).parent).replace("\\", "/")
    except ValueError:
        rel = str(chart)
    L.insert(3, f"![alert rate per sound]({rel})\n")
    L.append("## Reproduce\n")
    L.append("```bash\npixi run fetch-esc50\npixi run hearing-eval --n %d --seed %d\n```\n" % (args.n, args.seed))
    L.append("Per-trial rows: `assets/eval/hearing_trials.csv` (gitignored).")
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:20]))
    print(f"[hearing-eval] report -> {args.report}, chart -> {chart}")


if __name__ == "__main__":
    sys.exit(main())
