#!/usr/bin/env python3
"""
Convert a scanned human (Renderpeople-style OBJ, centimetres, Y-up, facing +Z) into a
MuJoCo-ready owner mesh — run with `pixi run prepare-owner`.

Output (gitignored, not redistributed): assets/models/owner/owner.obj + owner.png
  - metres, Z-up, facing +X (the owner body's forward axis), feet at z = 0, centred on x = y = 0
  - texture = the lowest-resolution diffuse map in the pack (2k), converted to PNG for MuJoCo

The source model is NOT committed (Renderpeople licence: no redistribution). Each teammate
downloads it themselves into assets/models/ and runs this script; without it the sim keeps
the primitive owner.

Several owners (start-up option `--owner <name>` / NUDGE_OWNER=<name>): every downloaded pack
assets/models/rp_<name>_OBJ/ can be converted to assets/models/owners/<name>/ (also gitignored).
  python scripts/prepare_owner.py                    # default pack -> assets/models/owner/ (as before)
  python scripts/prepare_owner.py --list             # owner names of the downloaded packs
  python scripts/prepare_owner.py --name mei_posed_001   # one pack -> assets/models/owners/mei_posed_001/
  python scripts/prepare_owner.py --all              # every pack -> assets/models/owners/<name>/
Usage: python scripts/prepare_owner.py [--src DIR] [--yaw-deg DEG] [--name NAME | --all | --list]
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "assets" / "models"
OUT_DIR = MODEL_DIR / "owner"
OWNERS_DIR = MODEL_DIR / "owners"  # one folder per owner name (start-up option --owner)
DEFAULT_SRC = MODEL_DIR / "rp_posed_00178_29_OBJ"


def owner_name(pack: Path) -> str:
    """rp_mei_posed_001_OBJ -> mei_posed_001 (the name for --owner / NUDGE_OWNER)."""
    n = pack.name
    n = n[3:] if n.lower().startswith("rp_") else n
    n = n[:-4] if n.lower().endswith("_obj") else n
    return n.lower()


def find_packs() -> dict:
    """{owner name: pack folder} of every downloaded Renderpeople OBJ pack in assets/models/."""
    return {owner_name(p): p for p in sorted(MODEL_DIR.glob("rp_*"))
            if p.is_dir() and any(p.glob("*.obj"))}


def pick_files(src: Path) -> tuple[Path, Path]:
    objs = sorted(src.glob("*.obj"), key=lambda p: p.stat().st_size)
    if not objs:
        sys.exit(f"[owner] no .obj found in {src}")
    difs = list(src.rglob("*_dif_*.jpg")) + list(src.rglob("*_dif_*.png"))
    if not difs:  # newer Renderpeople packs: tex/<name>_basecolor.jpg (4k) instead of *_dif_2k.jpg
        difs = list(src.rglob("*basecolor*.jpg")) + list(src.rglob("*basecolor*.png"))
    if not difs:
        sys.exit(f"[owner] no diffuse texture (*_dif_*.jpg/png or *basecolor*.jpg/png) found in {src}")

    def res_k(p: Path) -> int:  # rp_..._dif_2k.jpg -> 2
        m = re.search(r"_(\d+)k\.", p.name)
        return int(m.group(1)) if m else 999

    return objs[0], min(difs, key=res_k)  # lowest-poly mesh, lowest-resolution texture


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC, help="folder with the scanned OBJ + tex/")
    ap.add_argument("--yaw-deg", type=float, default=0.0, help="extra yaw if the scan doesn't face +Z")
    ap.add_argument("--name", help="convert the pack of this owner into assets/models/owners/<name>/")
    ap.add_argument("--all", action="store_true", help="convert every downloaded pack into assets/models/owners/<name>/")
    ap.add_argument("--list", action="store_true", help="list the owner names of the downloaded packs")
    args = ap.parse_args()

    packs = find_packs()
    if args.list:
        print("[owner] downloaded owners:", ", ".join(packs) or "none (put rp_*_OBJ/ folders in assets/models/)")
        return 0
    if args.all:
        if not packs:
            sys.exit("[owner] no rp_*_OBJ packs in assets/models/")
        for name, pack in packs.items():
            convert(pack, OWNERS_DIR / name, args.yaw_deg)
        return 0
    if args.name:
        pack = packs.get(args.name.lower()) or next(
            (p for n, p in packs.items() if n.startswith(args.name.lower())), None)
        if pack is None:
            sys.exit(f"[owner] no pack for owner {args.name!r} (downloaded: {', '.join(packs) or 'none'})")
        return convert(pack, OWNERS_DIR / args.name, args.yaw_deg)
    return convert(args.src, OUT_DIR, args.yaw_deg)


def convert(src: Path, out_dir: Path, yaw_deg: float = 0.0) -> int:
    """One scanned pack -> out_dir/owner.obj + owner.png."""
    obj_path, tex_path = pick_files(src)
    print(f"[owner] mesh    {obj_path.relative_to(PROJECT_ROOT)}")
    print(f"[owner] texture {tex_path.relative_to(PROJECT_ROOT)}")

    lines = obj_path.read_text(encoding="utf-8", errors="replace").splitlines()
    verts = np.array([[float(t) for t in ln.split()[1:4]] for ln in lines if ln.startswith("v ")])

    # cm -> m; scan frame (X right, Y up, Z front) -> body frame (X front, Y left, Z up).
    # body = (z, x, y) maps scan front (+Z) to +X and scan up (+Y) to +Z (right-handed).
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))

    def to_body(p: np.ndarray, scale: float) -> np.ndarray:
        b = np.stack([p[:, 2], p[:, 0], p[:, 1]], axis=1) * scale
        return np.stack([c * b[:, 0] - s * b[:, 1], s * b[:, 0] + c * b[:, 1], b[:, 2]], axis=1)

    vb = to_body(verts, 0.01)
    centre = np.array([(vb[:, 0].min() + vb[:, 0].max()) / 2, (vb[:, 1].min() + vb[:, 1].max()) / 2, vb[:, 2].min()])
    vb -= centre
    print(f"[owner] size    {np.ptp(vb[:, 0]):.2f} x {np.ptp(vb[:, 1]):.2f} x {np.ptp(vb[:, 2]):.2f} m (front x left x up)")

    out, vi, ni = [f"# Generated by scripts/prepare_owner.py from {obj_path.name} — do not commit"], 0, 0
    normals = [ln for ln in lines if ln.startswith("vn ")]
    nb = to_body(np.array([[float(t) for t in ln.split()[1:4]] for ln in normals]), 1.0) if normals else None
    for ln in lines:
        if ln.startswith("v "):
            out.append("v {:.5f} {:.5f} {:.5f}".format(*vb[vi]))
            vi += 1
        elif ln.startswith("vn "):
            out.append("vn {:.4f} {:.4f} {:.4f}".format(*nb[ni]))
            ni += 1
        elif ln.startswith(("vt ", "f ")):
            out.append(ln.strip())
        # drop g / o / usemtl / mtllib / s: MuJoCo uses one material from the MJCF
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "owner.obj").write_text("\n".join(out) + "\n", encoding="utf-8")

    img = cv2.imread(str(tex_path), cv2.IMREAD_COLOR)
    if img is None:
        sys.exit(f"[owner] could not read {tex_path}")
    if max(img.shape[:2]) > 2048:  # 4k basecolor -> 2k, like the *_dif_2k.jpg packs
        img = cv2.resize(img, (2048, 2048), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_dir / "owner.png"), img)
    print(f"[owner] wrote   {(out_dir / 'owner.obj').relative_to(PROJECT_ROOT)} ({vi} verts) + owner.png "
          f"({img.shape[1]}x{img.shape[0]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
