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
  python scripts/prepare_owner.py --name mei         # one pack -> assets/models/owners/mei/
  python scripts/prepare_owner.py --all              # every pack -> assets/models/owners/<name>/
Names: rp_mei_posed_001_OBJ -> mei, rp_dennis_posed_004_OBJ -> dennis, rp_posed_00178_29_OBJ -> posed_00178_29.
Mesh: *_30k.obj if present, else the pack's single OBJ; > --max-faces (60k) triangles -> decimated to ~30k.
Packs are looked up in NUDGE_MODELS_DIR, else assets/models/, else (git worktree) the main checkout's.
Usage: python scripts/prepare_owner.py [--src DIR] [--yaw-deg DEG] [--name NAME | --all | --list]
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAX_FACES = 60000      # a mesh much bigger than ~30k triangles is decimated to TARGET_FACES
TARGET_FACES = 30000


def _main_checkout() -> Path | None:
    """In a git worktree, .git is a file 'gitdir: <main>/.git/worktrees/<name>': the main checkout."""
    g = PROJECT_ROOT / ".git"
    if g.is_file():
        txt = g.read_text(encoding="utf-8", errors="replace").strip()
        if txt.startswith("gitdir:"):
            p = Path(txt.split(":", 1)[1].strip())
            if p.parent.name == "worktrees":
                return p.parent.parent.parent
    return None


def models_dir() -> Path:
    """Where the downloaded Renderpeople packs are: NUDGE_MODELS_DIR, else this checkout's
    assets/models/, else (a git worktree without them) the main checkout's assets/models/."""
    env = os.environ.get("NUDGE_MODELS_DIR", "").strip()
    if env:
        return Path(env)
    here = PROJECT_ROOT / "assets" / "models"
    if any(here.glob("rp_*")) or (here / "owners").is_dir():
        return here
    main = _main_checkout()
    if main is not None and any((main / "assets" / "models").glob("rp_*")):
        return main / "assets" / "models"
    return here


MODEL_DIR = models_dir()
OUT_DIR = PROJECT_ROOT / "assets" / "models" / "owner"   # the default owner (what sim/world.py loads by default)
OWNERS_DIR = MODEL_DIR / "owners"  # one folder per owner name (start-up option --owner)
DEFAULT_SRC = MODEL_DIR / "rp_posed_00178_29_OBJ"


def _rel(p: Path) -> str:
    try:
        return p.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def full_name(pack: Path) -> str:
    """rp_mei_posed_001_OBJ -> mei_posed_001."""
    n = pack.name
    n = n[3:] if n.lower().startswith("rp_") else n
    n = n[:-4] if n.lower().endswith("_obj") else n
    return n.lower()


def owner_name(pack: Path) -> str:
    """Short name for --owner / NUDGE_OWNER: rp_mei_posed_001_OBJ -> mei, rp_dennis_posed_004_OBJ ->
    dennis; packs without a person's name keep the full one (rp_posed_00178_29_OBJ -> posed_00178_29)."""
    n = full_name(pack)
    m = re.fullmatch(r"([a-z]+)_posed_\d+", n)
    return m.group(1) if m else n


def find_packs() -> dict:
    """{owner name: pack folder} of every downloaded Renderpeople OBJ pack (short names; the full
    name when two packs share a short name)."""
    packs = [p for p in sorted(MODEL_DIR.glob("rp_*")) if p.is_dir() and any(p.glob("*.obj"))]
    short = [owner_name(p) for p in packs]
    return {(n if short.count(n) == 1 else full_name(p)): p for n, p in zip(short, packs)}


def resolve(name: str) -> tuple[str, Path] | None:
    """Owner name (short, full or an unambiguous prefix) -> (short name, pack folder), or None."""
    name = name.strip().lower()
    packs = find_packs()
    if name in packs:
        return name, packs[name]
    for n, p in packs.items():
        if full_name(p) == name:
            return n, p
    hits = [(n, p) for n, p in packs.items() if n.startswith(name) or full_name(p).startswith(name)]
    return hits[0] if len(hits) == 1 else None


def pick_files(src: Path) -> tuple[Path, Path]:
    """Mesh: the *_30k.obj if the pack has one, else its single OBJ (newer packs: rp_posed_00178_29.obj),
    else the smallest. Texture: the lowest-resolution diffuse map."""
    objs = sorted(src.glob("*.obj"), key=lambda p: p.stat().st_size)
    if not objs:
        sys.exit(f"[owner] no .obj found in {src}")
    k30 = [p for p in objs if p.stem.lower().endswith("_30k")]
    plain = [p for p in objs if not re.search(r"_\d+k$", p.stem.lower())]
    obj = (k30 or plain or objs)[0]
    difs = list(src.rglob("*_dif_*.jpg")) + list(src.rglob("*_dif_*.png"))
    if not difs:  # newer Renderpeople packs: tex/<name>_basecolor.jpg (4k) instead of *_dif_2k.jpg
        difs = list(src.rglob("*basecolor*.jpg")) + list(src.rglob("*basecolor*.png"))
    if not difs:
        sys.exit(f"[owner] no diffuse texture (*_dif_*.jpg/png or *basecolor*.jpg/png) found in {src}")

    def res_k(p: Path) -> int:  # rp_..._dif_2k.jpg -> 2
        m = re.search(r"_(\d+)k\.", p.name)
        return int(m.group(1)) if m else 999

    return obj, min(difs, key=res_k)  # lowest-poly mesh, lowest-resolution texture


def decimate(lines: list, target: int) -> list:
    """Vertex-clustering decimation of OBJ lines (no extra dependency): vertices in the same grid cell
    merge to their mean, faces that collapse are dropped, UVs / normals keep their own indices. The
    cell size is searched so ~`target` triangles remain. Good enough for a visual-only mesh."""
    verts = np.array([[float(t) for t in ln.split()[1:4]] for ln in lines if ln.startswith("v ")])
    tris = []  # corner token triples; polygons are fan-triangulated
    for ln in lines:
        if ln.startswith("f "):
            toks = ln.split()[1:]
            for k in range(1, len(toks) - 1):
                tris.append((toks[0], toks[k], toks[k + 1]))
    vidx = np.array([[int(t.split("/")[0]) - 1 for t in tri] for tri in tris])

    def cluster(cell: float):
        key = np.floor((verts - verts.min(0)) / cell).astype(np.int64)
        _, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        c = inv[vidx]
        keep = (c[:, 0] != c[:, 1]) & (c[:, 1] != c[:, 2]) & (c[:, 0] != c[:, 2])
        return inv, keep

    size = float(np.ptp(verts, 0).max())
    lo, hi = 1e-4 * size, 0.2 * size
    for _ in range(14):  # bigger cell -> fewer faces
        mid = (lo * hi) ** 0.5
        if cluster(mid)[1].sum() > target:
            lo = mid
        else:
            hi = mid
    inv, keep = cluster(hi)
    n = int(inv.max()) + 1
    means = np.zeros((n, 3))
    np.add.at(means, inv, verts)
    means /= np.bincount(inv, minlength=n)[:, None]
    out = [f"v {x:.5f} {y:.5f} {z:.5f}" for x, y, z in means]
    out += [ln.strip() for ln in lines if ln.startswith(("vt ", "vn "))]
    for tri, ok in zip(tris, keep):
        if ok:
            out.append("f " + " ".join("/".join([str(inv[int(t.split("/")[0]) - 1] + 1)] + t.split("/")[1:])
                                       for t in tri))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC, help="folder with the scanned OBJ + tex/")
    ap.add_argument("--yaw-deg", type=float, default=0.0, help="extra yaw if the scan doesn't face +Z")
    ap.add_argument("--name", help="convert the pack of this owner into assets/models/owners/<name>/")
    ap.add_argument("--all", action="store_true", help="convert every downloaded pack into assets/models/owners/<name>/")
    ap.add_argument("--list", action="store_true", help="list the owner names of the downloaded packs")
    ap.add_argument("--max-faces", type=int, default=MAX_FACES,
                    help=f"decimate meshes with more triangles than this to ~{TARGET_FACES} (default {MAX_FACES})")
    args = ap.parse_args()

    packs = find_packs()
    if args.list:
        print(f"[owner] downloaded owners ({_rel(MODEL_DIR)}):",
              ", ".join(f"{n} ({p.name})" for n, p in packs.items()) or "none (put rp_*_OBJ/ folders in assets/models/)")
        return 0
    if args.all:
        if not packs:
            sys.exit("[owner] no rp_*_OBJ packs in assets/models/")
        for name, pack in packs.items():
            convert(pack, OWNERS_DIR / name, args.yaw_deg, args.max_faces)
        return 0
    if args.name:
        hit = resolve(args.name)
        if hit is None:
            sys.exit(f"[owner] no pack for owner {args.name!r} (downloaded: {', '.join(packs) or 'none'})")
        return convert(hit[1], OWNERS_DIR / hit[0], args.yaw_deg, args.max_faces)
    return convert(args.src, OUT_DIR, args.yaw_deg, args.max_faces)


def convert(src: Path, out_dir: Path, yaw_deg: float = 0.0, max_faces: int = MAX_FACES) -> int:
    """One scanned pack -> out_dir/owner.obj + owner.png."""
    obj_path, tex_path = pick_files(src)
    print(f"[owner] mesh    {_rel(obj_path)}")
    print(f"[owner] texture {_rel(tex_path)}")

    lines = obj_path.read_text(encoding="utf-8", errors="replace").splitlines()
    n_faces = sum(1 for ln in lines if ln.startswith("f "))
    if n_faces > max_faces:
        lines = decimate(lines, TARGET_FACES)
        print(f"[owner] decimated {n_faces} -> {sum(1 for ln in lines if ln.startswith('f '))} faces")
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
    print(f"[owner] wrote   {_rel(out_dir / 'owner.obj')} ({vi} verts) + owner.png "
          f"({img.shape[1]}x{img.shape[0]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
