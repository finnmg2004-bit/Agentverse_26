#!/usr/bin/env python3
"""
Regenerate the in-git owners in assets/owner_default/ (everyone gets them, no Renderpeople download; credits and
YOLO numbers in assets/owner_default/README.md):

    pixi run python scripts/prepare_default_owner.py --makehuman "Busy man/Style human obj.obj"   # default owner
    pixi run python scripts/prepare_default_owner.py --visitor     # FemaleVisitor (Open Robotics, CC BY 4.0)

Both: all textures merged into one atlas (the sim's owner loader takes one texture), Z-up, front +X, feet at z = 0,
centred on x = y = 0. MakeHuman exports (Y-up, decimetres, facing +Z) are scaled to --height and lose the
eyebrows / teeth / tongue (alpha-card or hidden parts). A Renderpeople scan in assets/models/ (never in git) still
takes priority on a PC that has one (sim/world.py); NUDGE_OWNER=busy / visitor picks these by name.
"""

from __future__ import annotations

import argparse
import io
import math
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "owner_default"
URL = "https://fuel.gazebosim.org/1.0/OpenRobotics/models/FemaleVisitor/tip/FemaleVisitor.zip"


def convert(src: Path, out: Path) -> None:
    mtl, cur = {}, None
    for line in src.with_suffix(".mtl").read_text().splitlines():
        p = line.split()
        if p[:1] == ["newmtl"]:
            cur = p[1]
        elif p[:1] == ["map_Kd"] and cur:
            mtl[cur] = src.parent / p[-1]
    V, VT, faces, use = [], [], [], None
    for line in src.read_text(errors="replace").splitlines():
        p = line.split()
        if not p:
            continue
        if p[0] == "v":
            V.append([float(x) for x in p[1:4]])
        elif p[0] == "vt":
            VT.append([float(x) for x in p[1:3]])
        elif p[0] == "usemtl":
            use = p[1]
        elif p[0] == "f":
            idx = [tuple(int(a) if a else 0 for a in (q.split("/") + ["", ""])[:2]) for q in p[1:]]
            faces += [(use, idx[0], idx[k], idx[k + 1]) for k in range(1, len(idx) - 1)]
    V = np.array(V)
    VT = np.array(VT) if VT else np.zeros((1, 2))
    mats = [m for m in dict.fromkeys(f[0] for f in faces) if m in mtl]
    imgs = [cv2.imread(str(mtl[m])) for m in mats]
    h = max(i.shape[0] for i in imgs)
    imgs = [cv2.resize(i, (int(i.shape[1] * h / i.shape[0]), h)) for i in imgs]
    width = sum(i.shape[1] for i in imgs)
    x0 = np.cumsum([0] + [i.shape[1] for i in imgs])
    P = V if int(np.argmax(V.max(0) - V.min(0))) == 2 else np.column_stack([V[:, 2], V[:, 0], V[:, 1]])  # Z-up
    P[:, 2] -= P[:, 2].min()
    P[:, :2] -= (P[:, :2].max(0) + P[:, :2].min(0)) / 2
    lines = ["# assets/owner_default/visitor: FemaleVisitor by Wan Yi Seow, Open Robotics, CC BY 4.0 (see ../README.md)",
             "usemtl owner"] + [f"v {x:.5f} {y:.5f} {z:.5f}" for x, y, z in P]
    vts, fl = [], []
    for m, *corner in faces:
        k = mats.index(m) if m in mats else 0
        ids = []
        for vi, ti in corner:
            u, v = VT[ti - 1] if ti else (0.0, 0.0)
            vts.append(f"vt {(x0[k] + (u % 1.0) * imgs[k].shape[1]) / width:.5f} {v % 1.0:.5f}")
            ids.append(f"{vi}/{len(vts)}")
        fl.append("f " + " ".join(ids))
    out.mkdir(parents=True, exist_ok=True)
    (out / "owner.obj").write_text("\n".join(lines + vts + fl) + "\n")
    cv2.imwrite(str(out / "owner.png"), np.hstack(imgs))
    print(f"[owner] wrote {out / 'owner.obj'} ({len(P)} verts, {len(fl)} tris, "
          f"{np.round(P.max(0) - P.min(0), 2)} m) + owner.png")


def convert_makehuman(src: Path, out: Path, height: float = 1.80,
                      skip=("eyebrow012", "teethMaterial", "tongue01Material"), tile: int = 1024) -> None:
    """MakeHuman OBJ export (many materials, Windows texture paths) -> owner.obj + grid-atlas owner.png."""
    mtl, cur = {}, None
    for line in src.with_suffix(".mtl").read_text(errors="replace").splitlines():
        p = line.split(maxsplit=1)
        if p[:1] == ["newmtl"]:
            cur = p[1].strip()
        elif p[:1] == ["map_Kd"] and cur:
            mtl[cur] = src.parent / p[1].strip().replace("\\", "/")
    V, VT, faces, use = [], [], [], None
    for line in src.read_text(errors="replace").splitlines():
        p = line.split()
        if not p:
            continue
        if p[0] == "v":
            V.append([float(x) for x in p[1:4]])
        elif p[0] == "vt":
            VT.append([float(x) for x in p[1:3]])
        elif p[0] == "usemtl":
            use = p[1]
        elif p[0] == "f" and use not in skip:
            idx = [tuple(int(a) if a else 0 for a in (q.split("/") + ["", ""])[:2]) for q in p[1:]]
            faces += [(use, idx[0], idx[k], idx[k + 1]) for k in range(1, len(idx) - 1)]
    V, VT = np.array(V), np.array(VT)
    mats = [m for m in dict.fromkeys(f[0] for f in faces) if m in mtl]
    g = math.ceil(math.sqrt(len(mats)))
    atlas = np.zeros((g * tile, g * tile, 3), np.uint8)
    for k, m in enumerate(mats):
        img = cv2.imread(str(mtl[m]), cv2.IMREAD_UNCHANGED)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:  # alpha (hair cards): composite onto dark grey, MuJoCo draws it opaque
            a = img[:, :, 3:4].astype(float) / 255.0
            img = (img[:, :, :3] * a + 40 * (1 - a)).astype(np.uint8)
        r, c = divmod(k, g)
        atlas[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = cv2.resize(img[:, :, :3], (tile, tile))
    P = np.column_stack([V[:, 2], V[:, 0], V[:, 1]])  # Y-up, facing +Z -> Z-up, facing +X
    used = np.unique([vi for f in faces for vi, _ in f[1:]]) - 1
    P *= height / (P[used, 2].max() - P[used, 2].min())
    P[:, 2] -= P[used, 2].min()
    P[:, :2] -= (P[used, :2].max(0) + P[used, :2].min(0)) / 2
    vts, fl = [], []
    for m, *corner in faces:
        r, c = divmod(mats.index(m) if m in mats else 0, g)
        ids = []
        for vi, ti in corner:
            u, v = VT[ti - 1] if ti else (0.0, 0.0)
            vts.append(f"vt {(c + (u % 1.0)) / g:.5f} {1 - (r + 1 - (v % 1.0)) / g:.5f}")
            ids.append(f"{vi}/{len(vts)}")
        fl.append("f " + " ".join(ids))
    out.mkdir(parents=True, exist_ok=True)
    head = ['# assets/owner_default: "Busy man" by Ricktho (Nudge team), CC0, made with MakeHuman (CC0 assets); '
            "see README.md", "usemtl owner"]
    (out / "owner.obj").write_text("\n".join(head + [f"v {x:.5f} {y:.5f} {z:.5f}" for x, y, z in P] + vts + fl) + "\n")
    cv2.imwrite(str(out / "owner.png"), atlas)
    print(f"[owner] wrote {out / 'owner.obj'} ({len(fl)} tris, {np.round(P[used].max(0) - P[used].min(0), 2)} m)"
          f" + owner.png {atlas.shape[1]}x{atlas.shape[0]} from {mats}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--visitor", action="store_true", help="download + convert FemaleVisitor -> visitor/")
    ap.add_argument("--makehuman", type=Path, help="MakeHuman OBJ export -> the default owner")
    ap.add_argument("--height", type=float, default=1.80)
    a = ap.parse_args()
    if a.makehuman:
        convert_makehuman(a.makehuman, OUT, a.height)
    if a.visitor:
        print(f"[owner] downloading {URL}")
        data = urllib.request.urlopen(URL, timeout=60).read()
        with tempfile.TemporaryDirectory() as tmp:
            zipfile.ZipFile(io.BytesIO(data)).extractall(tmp)
            convert(Path(tmp) / "meshes" / "FemaleVisitor.obj", OUT / "visitor")
    if not (a.makehuman or a.visitor):
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
