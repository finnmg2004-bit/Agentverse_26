#!/usr/bin/env python3
"""
Build the demo MONTAGE (one mp4, <= 60 s) from short recorded clips: trim, speed up, caption, title cards.
OpenCV only (no ffmpeg, no audio). Agent K, 2026-11-08.

    # all clips recorded by scripts/record_clips.py, in the order of its CLIPS list (trim/speed/caption from there)
    pixi run python scripts/make_montage.py
    pixi run python scripts/make_montage.py --clips voice,tv,smoke --out ~/nudge-overnight/videos/montage.mp4

    # any videos: "path|in_s|out_s|speed|caption"   (out_s empty = to the end, negative = from the end)
    pixi run python scripts/make_montage.py --seg "take_a.mp4|0|20|4|The doorbell rings" --seg "take_b.mp4|50||3|Smoke!"

Options: --title / --subtitle (opening card), --title-secs, --card-secs (0 = no cards between scenes),
--end "text" (closing card), --caption-pos top|bottom, --max 60 (refuses longer unless --force),
--fit 58 (raise all speeds evenly until it fits), --dry-run (print the plan only), --stills 3 (save PNG stills from the finished file to check captions).
Captions are ASCII (OpenCV Hershey font): em dashes / quotes are converted.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VIDEOS = Path.home() / "nudge-overnight" / "videos"
CLIPS_DIR = VIDEOS / "clips"

FONT = cv2.FONT_HERSHEY_DUPLEX
WHITE, BLACK, YELLOW = (255, 255, 255), (0, 0, 0), (0, 215, 255)  # BGR
BG = (32, 22, 14)  # title-card background (dark navy, BGR)
ACCENT = (230, 200, 60)  # cyan-ish accent line (BGR)


def ascii_text(s: str) -> str:
    rep = {"—": " - ", "–": "-", "‘": "'", "’": "'", "“": '"', "”": '"',
           "→": "->", "·": "|", "…": "...", "×": "x"}
    for k, v in rep.items():
        s = s.replace(k, v)
    return s.encode("ascii", "replace").decode("ascii")


# ------------------------------------------------------------------ text drawing
def _fit_scale(text: str, max_w: int, scale: float, thick: int) -> float:
    while scale > 0.4:
        (tw, _), _ = cv2.getTextSize(text, FONT, scale, thick)
        if tw <= max_w:
            break
        scale -= 0.05
    return scale


def _wrap(text: str, max_w: int, scale: float, thick: int, max_lines: int = 2) -> List[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if cv2.getTextSize(t, FONT, scale, thick)[0][0] <= max_w or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    if len(lines) > max_lines:  # squeeze the rest onto the last line (it gets shrunk to fit)
        lines = lines[:max_lines - 1] + [" ".join(lines[max_lines - 1:])]
    return lines


def put_centered(img, text: str, cy: int, scale: float, color, thick: int, max_w: Optional[int] = None) -> None:
    h, w = img.shape[:2]
    max_w = max_w or int(w * 0.9)
    scale = _fit_scale(text, max_w, scale, thick)
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thick)
    x = (w - tw) // 2
    cv2.putText(img, text, (x, cy + th // 2), FONT, scale, BLACK, thick + 4, cv2.LINE_AA)
    cv2.putText(img, text, (x, cy + th // 2), FONT, scale, color, thick, cv2.LINE_AA)


def caption_bar(img, text: str, pos: str = "bottom") -> None:
    """Large high-contrast caption: dark band (85 % opaque) + white text, up to 2 lines."""
    if not text:
        return
    h, w = img.shape[:2]
    scale = w / 1100.0  # ~1.5 at 1614 px wide -> ~30 px cap height
    thick = max(2, int(round(scale * 2)))
    lines = _wrap(ascii_text(text), int(w * 0.9), scale, thick)
    line_h = int(cv2.getTextSize("Ag", FONT, scale, thick)[0][1] * 1.7)
    band_h = line_h * len(lines) + int(line_h * 0.6)
    y0 = 0 if pos == "top" else h - band_h
    roi = img[y0:y0 + band_h]
    roi[:] = (roi.astype(np.float32) * 0.15).astype(np.uint8)
    cv2.line(img, (0, y0 if pos == "bottom" else y0 + band_h - 1), (w, y0 if pos == "bottom" else y0 + band_h - 1),
             ACCENT, 3)
    for i, ln in enumerate(lines):
        cy = y0 + int(line_h * 0.3) + line_h * i + line_h // 2
        put_centered(img, ln, cy, scale, WHITE, thick)


def speed_badge(img, speed: float, pos: str = "bottom") -> None:
    """Small yellow 'x4' badge in the corner away from the caption."""
    if abs(speed - 1.0) < 0.05:
        return
    h, w = img.shape[:2]
    txt = f"x{speed:g}"
    scale = w / 1400.0
    thick = max(2, int(round(scale * 2)))
    (tw, th), _ = cv2.getTextSize(txt, FONT, scale, thick)
    pad = int(th * 0.5)
    x1, y1 = w - 16, (16 if pos == "bottom" else h - 16 - th - 2 * pad)
    x0, y0 = x1 - tw - 2 * pad, y1
    cv2.rectangle(img, (x0, y0), (x1, y0 + th + 2 * pad), YELLOW, -1)
    cv2.putText(img, txt, (x0 + pad, y0 + pad + th), FONT, scale, BLACK, thick, cv2.LINE_AA)


def title_card(size, title: str, subtitle: str = "", label: str = "") -> np.ndarray:
    w, h = size
    img = np.full((h, w, 3), BG, np.uint8)
    s = w / 700.0
    put_centered(img, ascii_text(title), int(h * 0.44), s, WHITE, max(2, int(s * 2)))
    cv2.line(img, (int(w * 0.3), int(h * 0.53)), (int(w * 0.7), int(h * 0.53)), ACCENT, 3)
    if subtitle:
        put_centered(img, ascii_text(subtitle), int(h * 0.62), s * 0.55, (210, 210, 210), max(1, int(s)))
    if label:
        put_centered(img, ascii_text(label), int(h * 0.30), s * 0.45, ACCENT, max(1, int(s)))
    return img


# ------------------------------------------------------------------ segments
class Seg:
    def __init__(self, path: str, t_in: float = 0.0, t_out: Optional[float] = None, speed: float = 1.0,
                 caption: str = "", card: str = ""):
        self.path, self.t_in, self.t_out, self.speed = str(path), float(t_in), t_out, max(0.05, float(speed))
        self.caption, self.card = caption, card
        cap = cv2.VideoCapture(self.path)
        self.ok = cap.isOpened()
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        self.n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.size = (int(cap.get(3)), int(cap.get(4)))
        cap.release()
        dur = self.n / self.fps if self.fps else 0.0
        out = dur if t_out is None else (dur + t_out if t_out < 0 else min(t_out, dur))
        self.src_in, self.src_out = max(0.0, min(self.t_in, dur)), max(0.0, out)

    @property
    def src_len(self) -> float:
        return max(0.0, self.src_out - self.src_in)

    def out_len(self) -> float:
        return self.src_len / self.speed


def parse_seg(spec: str) -> Seg:
    parts = (spec.split("|") + ["", "", "", ""])[:5]
    path, t_in, t_out, speed, caption = parts
    p = Path(os.path.expanduser(path))
    if not p.is_absolute() and not p.exists():
        for base in (VIDEOS, CLIPS_DIR):
            if (base / p).exists():
                p = base / p
                break
    return Seg(str(p), float(t_in or 0), float(t_out) if t_out.strip() else None, float(speed or 1), caption)


def segs_from_clip_list(names: Optional[List[str]], clips_dir: Path) -> List[Seg]:
    sys.path.insert(0, str(ROOT / "scripts"))
    from record_clips import CLIPS  # the one clip list (data at the top of record_clips.py)
    by_name = {c["name"]: c for c in CLIPS}
    order = names or [c["name"] for c in CLIPS if c.get("in_montage", True)]
    segs = []
    for n in order:
        c = by_name.get(n)
        if c is None:
            print(f"[montage] unknown clip '{n}' (see: python scripts/record_clips.py --list)")
            continue
        path = clips_dir / f"{n}.mp4"
        if not path.exists():
            print(f"[montage] skip '{n}': {path} not recorded yet")
            continue
        t_in, t_out = c.get("trim", (0.0, None))
        segs.append(Seg(str(path), t_in, t_out, c.get("speed", 1.0), c.get("caption", ""), c.get("card", "")))
    return segs


# ------------------------------------------------------------------ writer
def open_writer(path: str, fps: float, size) -> cv2.VideoWriter:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    for fourcc in ("avc1", "mp4v"):
        wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if wr.isOpened():
            print(f"[montage] codec {fourcc}")
            return wr
        wr.release()
    raise SystemExit(f"[montage] cannot open a VideoWriter for {path}")


def fit_frame(img, size) -> np.ndarray:
    w, h = size
    ih, iw = img.shape[:2]
    if (iw, ih) == (w, h):
        return img
    s = min(w / iw, h / ih)
    r = cv2.resize(img, (int(iw * s), int(ih * s)), interpolation=cv2.INTER_AREA)
    out = np.zeros((h, w, 3), np.uint8)
    y, x = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
    out[y:y + r.shape[0], x:x + r.shape[1]] = r
    return out


def write_segment(wr, seg: Seg, size, fps: float, pos: str) -> int:
    cap = cv2.VideoCapture(seg.path)
    n_out = int(seg.out_len() * fps)
    first = int(round(seg.src_in * seg.fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    cur, frame, written = first, None, 0
    for j in range(n_out):
        want = first + int(round(j * seg.speed * seg.fps / fps))
        while cur <= want:  # grab (cheap) up to the wanted frame, decode only that one
            if cur == want:
                ok, frame = cap.read()
            else:
                ok = cap.grab()
            cur += 1
            if not ok:
                cap.release()
                return written
        img = fit_frame(frame, size).copy()
        caption_bar(img, seg.caption, pos)
        speed_badge(img, seg.speed, pos)
        wr.write(img)
        written += 1
    cap.release()
    return written


def save_stills(path: str, n: int) -> List[str]:
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    out = []
    for k in range(n):
        idx = int(total * (k + 1) / (n + 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, img = cap.read()
        if ok:
            p = str(Path(path).with_suffix("")) + f"_still{k + 1}.png"
            cv2.imwrite(p, img)
            out.append(p)
    cap.release()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seg", action="append", default=[], help='"path|in_s|out_s|speed|caption" (repeatable)')
    ap.add_argument("--clips", default="", help="comma list of record_clips.py clip names (default: all recorded)")
    ap.add_argument("--clips-dir", default=str(CLIPS_DIR))
    ap.add_argument("--out", default=str(VIDEOS / "montage.mp4"))
    ap.add_argument("--title", default="Nudge - an assistance dog that adapts")
    ap.add_argument("--subtitle", default="Local AI on a simulated Unitree Go2 | for deaf, blind and deafblind owners")
    ap.add_argument("--title-secs", type=float, default=2.0)
    ap.add_argument("--card-secs", type=float, default=0.5, help="title card before each scene (0 = none)")
    ap.add_argument("--end", default="", help="closing card text (empty = none)")
    ap.add_argument("--end-secs", type=float, default=1.5)
    ap.add_argument("--caption-pos", choices=["top", "bottom"], default="bottom")
    ap.add_argument("--fps", type=float, default=15.0, help="output fps (the GUI records 15 fps)")
    ap.add_argument("--max", type=float, default=60.0)
    ap.add_argument("--force", action="store_true", help="write even if longer than --max")
    ap.add_argument("--fit", type=float, default=0.0,
                    help="if longer than this many seconds, raise every scene's speed by the same factor to fit")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stills", type=int, default=0, help="save N PNG stills from the finished montage")
    args = ap.parse_args(argv)

    segs = [parse_seg(s) for s in args.seg] if args.seg else \
        segs_from_clip_list([c.strip() for c in args.clips.split(",") if c.strip()] or None, Path(args.clips_dir))
    bad = [s.path for s in segs if not s.ok or s.n == 0]
    if bad:
        print(f"[montage] cannot read: {bad}")
        return 2
    if not segs:
        print("[montage] nothing to do (no segments / no recorded clips)")
        return 2
    size = segs[0].size
    cards = args.card_secs if len(segs) > 1 or args.card_secs else 0.0
    fixed = (args.title_secs if args.title else 0) + cards * len(segs) + (args.end_secs if args.end else 0)
    body = sum(s.out_len() for s in segs)
    if args.fit and fixed + body > args.fit and args.fit > fixed:
        k = body / (args.fit - fixed)
        print(f"[montage] --fit {args.fit:g}: all speeds x{k:.2f}")
        for s in segs:
            s.speed = math.ceil(s.speed * k * 10) / 10.0  # round up so the total stays under
    total = fixed + sum(s.out_len() for s in segs)
    print(f"[montage] {len(segs)} scenes -> {args.out} @ {size[0]}x{size[1]} {args.fps:g} fps")
    for i, s in enumerate(segs, 1):
        print(f"  {i}. {Path(s.path).name:22s} {s.src_in:6.1f}-{s.src_out:6.1f} s  x{s.speed:<4g} -> {s.out_len():5.1f} s"
              f"  | {ascii_text(s.caption)}")
    print(f"[montage] TOTAL {total:.1f} s (title {args.title_secs if args.title else 0:g} s, cards {cards:g} s each)")
    if total > args.max and not args.force:
        print(f"[montage] REFUSED: {total:.1f} s > {args.max:g} s. Raise speeds / trim more, or pass --force.")
        return 1
    if args.dry_run:
        return 0

    wr = open_writer(args.out, args.fps, size)
    frames = 0

    def hold(img, secs):
        nonlocal frames
        for _ in range(int(round(secs * args.fps))):
            wr.write(img)
            frames += 1

    if args.title:
        hold(title_card(size, args.title, args.subtitle), args.title_secs)
    for i, s in enumerate(segs, 1):
        if cards:
            hold(title_card(size, s.card or s.caption, "", f"{i} / {len(segs)}"), cards)
        frames += write_segment(wr, s, size, args.fps, args.caption_pos)
    if args.end:
        hold(title_card(size, args.end), args.end_secs)
    wr.release()
    print(f"[montage] wrote {frames} frames = {frames / args.fps:.1f} s -> {args.out}")
    if args.stills:
        for p in save_stills(args.out, args.stills):
            print(f"[montage] still: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
