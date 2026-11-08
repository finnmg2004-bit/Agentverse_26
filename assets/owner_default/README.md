# Default owner (in git: everyone gets it, no download)

The sim uses this owner whenever no Renderpeople scan is installed on the PC (`assets/models/owner/`, never
in git: Renderpeople licence). Pick one by name at start-up with `NUDGE_OWNER=<name>` (`--owner`):

| Name | Files | Model | Licence | YOLOv8n detects it |
|---|---|---|---|---|
| `busy` (default) | `owner.obj`, `owner.png` | "Busy man" by **Ricktho** (Nudge team), made with MakeHuman (its stock assets are CC0) | **CC0 1.0** | 404 / 412 views (98 %) |
| `visitor` | `visitor/owner.obj`, `visitor/owner.png` | "FemaleVisitor" by **Wan Yi Seow, Open Robotics**, from Gazebo Fuel (<https://app.gazebosim.org/OpenRobotics/fuel/models/FemaleVisitor>) | **CC BY 4.0** (<https://creativecommons.org/licenses/by/4.0/>) | 392 / 412 views (95 %) |

Both modified for MuJoCo: one OBJ + one texture atlas (the sim's owner loader takes one texture), Z-up, front +X,
feet at z = 0, centred; Busy man scaled to 1.80 m, eyebrows / teeth / tongue dropped. Regenerate with
`scripts/prepare_default_owner.py` (`--visitor` downloads FemaleVisitor; `--makehuman <obj>` converts a MakeHuman
OBJ export). Detection numbers from `scripts/check_owner_yolo.py` (6 rooms x 4 headings x 1.5 / 2.5 / 3.5 m).
