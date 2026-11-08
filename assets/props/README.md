# Props (shared, in git)

Extra objects anyone can place in the house. Not used by the sim yet (no `sim/world.py` hook).

| Folder | What | Source / licence | Notes |
|---|---|---|---|
| `baby/` | Newborn baby (`baby.obj` + `baby.mtl` + `baby_basecolor.jpg`) | Made by Teo with Tripo AI (Tripo plan allows sharing) | Shrunk from 432k to 19k triangles (vertex clustering, normals dropped; only the base-colour texture kept). Units: 1.0 high (Y up, Tripo export): scale ~0.5 for a newborn. Loads in MuJoCo as a mesh. |
| `crib/` | "Baby bed with mattress guard" (`baby_bed_with_mattress_guard.glb`) | aswwwcom, Sketchfab, **CC BY 4.0**: https://sketchfab.com/3d-models/baby-bed-with-mattress-guard-730d615e84dd49a5b1b1a4c4e0256542 | Original glTF (22 MB, 17 meshes, 13 textures). MuJoCo needs OBJ/STL: convert before use. Credit required. |
