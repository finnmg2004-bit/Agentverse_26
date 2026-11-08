#!/usr/bin/env python3
"""
H.264 MP4 (what WhatsApp, browsers, phones and slide decks play) with PyAV, which every checkout has
(faster-whisper depends on it). OpenCV's own MP4 writer gives "mp4v" (MPEG-4 Part 2) on Windows, which
WhatsApp refuses.

Convert finished videos (writes <name>_h264.mp4 next to each, or --out for a single file):
    pixi run python scripts/video_h264.py clip.mp4 [more.mp4 ...]
    pixi run python scripts/video_h264.py montage.mp4 --out montage_whatsapp.mp4

Write H.264 directly from code (BGR frames, like cv2.VideoWriter):
    from video_h264 import H264Writer
    w = H264Writer("out.mp4", fps=30, size=(1280, 720)); w.write(bgr); w.release()
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import av
import cv2


class H264Writer:
    """Drop-in for cv2.VideoWriter (write(bgr) / release()): H.264, yuv420p, +faststart."""

    def __init__(self, path, fps: float, size, crf: int = 23) -> None:
        self.w, self.h = int(size[0]) // 2 * 2, int(size[1]) // 2 * 2   # yuv420p needs even sizes
        self.size = (int(size[0]), int(size[1]))
        self.out = av.open(str(path), "w", options={"movflags": "+faststart"})
        self.st = self.out.add_stream("libx264", rate=int(round(fps)))
        self.st.width, self.st.height, self.st.pix_fmt = self.w, self.h, "yuv420p"
        self.st.options = {"crf": str(crf), "preset": "medium", "profile": "high"}

    def isOpened(self) -> bool:   # noqa: N802 - cv2.VideoWriter API
        return True

    def write(self, bgr) -> None:
        if bgr.shape[1] != self.w or bgr.shape[0] != self.h:
            bgr = bgr[: self.h, : self.w]
        for p in self.st.encode(av.VideoFrame.from_ndarray(bgr, format="bgr24")):
            self.out.mux(p)

    def release(self) -> None:
        for p in self.st.encode():
            self.out.mux(p)
        self.out.close()


def convert(src: Path, dst: Path) -> int:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        print(f"[h264] cannot read {src}")
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    w, n = H264Writer(dst, fps, size), 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        w.write(frame)
        n += 1
    w.release()
    print(f"[h264] {src} -> {dst}: {n} frames, {fps:g} fps, {dst.stat().st_size / 1e6:.1f} MB (no audio)")
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out", help="output file (only with one input)")
    args = ap.parse_args()
    if args.out and len(args.videos) != 1:
        ap.error("--out needs exactly one input")
    ok = True
    for v in args.videos:
        src = Path(v)
        dst = Path(args.out) if args.out else src.with_name(src.stem + "_h264.mp4")
        ok &= convert(src, dst) > 0
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
