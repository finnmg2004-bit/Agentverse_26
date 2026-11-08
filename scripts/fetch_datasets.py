#!/usr/bin/env python3
"""
Download evaluation datasets (gitignored, never committed) — run with `pixi run fetch-esc50`.

  assets/datasets/ESC-50/   ESC-50: Dataset for Environmental Sound Classification
                            K. J. Piczak, ACM Multimedia 2015, https://github.com/karolpiczak/ESC-50
                            2,000 five-second clips (44.1 kHz mono WAV), 50 classes x 40, 5 folds.
                            Licence: Creative Commons Attribution-NonCommercial 3.0 (CC BY-NC 3.0);
                            clip authors are listed in the dataset's meta/esc50.csv / LICENSE.
                            Used only to evaluate hearing (scripts/test_hearing_eval.py); not
                            redistributed and not part of the runtime.

Pinned to a GitHub commit. GitHub's zip archives are not byte-stable over time, so the check is on
the content instead: SHA-256 of meta/esc50.csv + all 2,000 clips present.
Usage: python scripts/fetch_datasets.py [--keep-zip]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# NUDGE_DATASETS=<dir> puts the datasets elsewhere (e.g. one shared copy for several git worktrees)
DATA_DIR = Path(os.environ.get("NUDGE_DATASETS", PROJECT_ROOT / "assets" / "datasets"))

ESC50_COMMIT = "33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6"  # master, 2023-10-13
ESC50_URL = f"https://codeload.github.com/karolpiczak/ESC-50/zip/{ESC50_COMMIT}"
ESC50_DIR = DATA_DIR / "ESC-50"
ESC50_CSV_SHA256 = "ca660da60191a97de289983a05821c9382d852a38a2ba8428980816b68cf6246"
ESC50_N_CLIPS = 2000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def esc50_ok(verbose: bool = True) -> bool:
    csv_path = ESC50_DIR / "meta" / "esc50.csv"
    if not csv_path.exists():
        return False
    got = sha256(csv_path)
    n = len(list((ESC50_DIR / "audio").glob("*.wav")))
    ok = got == ESC50_CSV_SHA256 and n == ESC50_N_CLIPS
    if verbose and not ok:
        print(f"[warn] ESC-50 check: meta/esc50.csv sha256 {got} (want {ESC50_CSV_SHA256}), {n} clips")
    return ok


def fetch_esc50(keep_zip: bool) -> bool:
    if esc50_ok(verbose=False):
        print(f"[ok]   ESC-50 already present in {ESC50_DIR}")
        return True
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / f"ESC-50-{ESC50_COMMIT[:8]}.zip"
    if not zip_path.exists():
        print(f"[get]  ESC-50 (~615 MB, CC BY-NC 3.0) <- {ESC50_URL}")
        tmp = zip_path.with_suffix(".zip.part")
        try:
            req = urllib.request.Request(ESC50_URL, headers={"User-Agent": "nudge-fetch-datasets"})
            with urllib.request.urlopen(req, timeout=300) as resp, open(tmp, "wb") as out:
                shutil.copyfileobj(resp, out, length=1 << 22)
        except Exception as exc:
            print(f"[FAIL] ESC-50: {exc}")
            tmp.unlink(missing_ok=True)
            return False
        tmp.replace(zip_path)
    print(f"[unzip] {zip_path.name} ({zip_path.stat().st_size / 1e6:.0f} MB): meta/ + audio/ only")
    ESC50_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            parts = Path(info.filename).parts
            # keep only <root>/meta/*.csv, <root>/audio/*.wav and the licence text; no code, no notebooks
            if len(parts) < 2 or info.is_dir():
                continue
            rel = Path(*parts[1:])
            keep = (rel.parts[0] == "audio" and rel.suffix == ".wav" and len(rel.parts) == 2) or \
                   (rel.parts[0] == "meta" and rel.suffix == ".csv" and len(rel.parts) == 2) or \
                   rel.as_posix() in ("LICENSE", "README.md")
            if not keep:
                continue
            dest = ESC50_DIR / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)
    ok = esc50_ok()
    if ok and not keep_zip:
        zip_path.unlink()
    print(f"[{'ok' if ok else 'FAIL'}]   ESC-50 -> {ESC50_DIR}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep-zip", action="store_true", help="keep the downloaded zip next to the dataset")
    args = ap.parse_args()
    return 0 if fetch_esc50(args.keep_zip) else 1


if __name__ == "__main__":
    sys.exit(main())
