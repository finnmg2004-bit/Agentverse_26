# Start-up options: owner + house (Teo, 2026-11-08, branch `claude/startup-options-d46587`)

Everything is **default off**. With no flags the sim, the brain and the scenario suite behave exactly as before (`doorbell` PASS 61.4 s; `check-world` passes for the house).

## One place to choose: `scripts/run_all.py` (= `pixi run start`)
```
python scripts/run_all.py --owner-room guest --owner-state asleep_bed --profile deaf --owner-wander off
python scripts/run_all.py --owner mei_posed_001
python scripts/run_all.py --house apt                  # EXPERIMENTAL second house
python scripts/run_all.py --from-gui                   # use what the GUI "Start-up" tab saved
```
| Flag | Values | What it does |
|---|---|---|
| `--owner-room` | an `OWNER_SPOTS` room | owner's start room; `[R]` puts them back there |
| `--owner-state` | `awake` / `asleep_bed` / `asleep_sofa` | asleep in the guest bed / napping on the living-room sofa (sets Ricktho's `NUDGE_OWNER_IN_BED=bed` / `sofa`, then owner state `asleep` at start and after `[R]`) |
| `--profile` | `deaf` / `blind` / `deafblind` | C10 profile (passes the existing `run_sim.py --profile`) |
| `--owner-wander` | `on` / `off` | `off` = existing `--owner-static` |
| `--owner` | a scan name | which scanned person is the owner (`NUDGE_OWNER`) |
| `--house` | `house` / `apt` | which house (`NUDGE_HOUSE`, also set for the brain) |

One line is printed at start by `run_all` (`[start] OPTIONS: ...`) and by the sim (`[sim] START OPTIONS: house=... owner=... room=... state=... profile=... wander=...`).
The same flags work on `sim/run_sim.py` directly. `--owner` / `--owner-state` / `--house` are turned into env vars before the sim modules load, because those modules read env at import.

## GUI: new tab `gui/panels/startup.py` ("Start-up")
It has dropdowns and radio buttons for the same six choices, and a "Save for next run" button that writes `assets/startup.json` (gitignored). It also shows the matching `run_all.py` command. **It never restarts the running sim.** If a choice is ignored, it says so: asleep in bed means the guest bedroom, napping means the living room, and apt is marked experimental. I checked that the panel builds in a headless Dear PyGui context. I didn't open it in the live GUI.

## Several owners (`scripts/prepare_owner.py`, `sim/world.py`)
- `python scripts/prepare_owner.py --list` shows the downloaded packs, by name (`rp_mei_posed_001_OBJ` becomes `mei_posed_001`).
- `--name <n>` converts one pack to `assets/models/owners/<n>/`; `--all` converts every pack.
- With no arguments it does what it did before: the default pack goes to `assets/models/owner/`.
- `NUDGE_OWNER=<n>` (or `--owner <n>`) loads `owners/<n>/`. If that folder is missing, the sim converts the matching pack the first time. If there's no such pack, it prints one line and uses the default owner.
- Unset or `default` means `assets/models/owner/` exactly as before.
- In `sim/world.py` I only changed how the owner folder is chosen: `_chosen_owner_dir()`, plus a one-line skip in `_ensure_scanned_owner`. `_add_scanned_owner` is untouched, for the owner-animation session.
- Licence: both folders sit under `assets/*`, which is gitignored (`git check-ignore` confirms it), so nothing from Renderpeople can be committed.
- Tested: `--name mei_posed_001` converts it; `NUDGE_OWNER=mei_posed_001` picks it; an unknown name falls back to the default owner.

## Second house: `--house apt` (EXPERIMENTAL, works for the doorbell)
`sim/layouts/apt.py` defines every name that `house_layout.py` exports, using coordinates from upstream `scenes/apt/layout.py` (floor 62 of the Manhattan duplex). A small hook at the end of `sim/house_layout.py`, `_select_house()`, copies those names over its own only when `NUDGE_HOUSE=apt`. The submodule is not edited.
- **Room keys are Nudge's own**, so code that names rooms keeps working:
  - foyer = `entry`, great room = `living_room`, library = `family_room`, east hall = `robot_home`
  - plus `dining`, `gallery`, `media` (TV), `laundry`, `guest_bath`, `kitchen`
- **Front door fix:** upstream's front door is a decorative leaf in front of a *solid* wall, 232 m above the street, so the flat had no exit. At load time, and in the apt only, `world.py` removes that wall and leaf (`REMOVE_GEOMS`). It then adds walls with a real 1.3 m doorway plus a small lift-lobby stub outside (`ADD_GEOMS`). Both hooks do nothing in the house.
- **Results:**
  - `NUDGE_HOUSE=apt python sim/check_world.py`: **all checks passed** (own nav-map cache `assets/maps/nerv_apt_nav.npz`).
  - `doorbell` in the apt: **PASS 51.1 s** (owner in the great room, found, nudged, guided to the front door).
- **What's missing or limited:**
  - The **guest bedroom is out of bounds**: its only door opens onto the stair flights, and the planner finds no path. So `asleep_bed` doesn't work in the apt: `set_owner_room("guest")` prints "unknown room". The sofa nap uses the house's sofa geom name, so it doesn't work either.
  - The kitchen is reachable (the nav map goes across the stair hall, 16 m from the start). I didn't test kettle, oven timer or smoke there.
  - Smoke evacuation goes out of the front door into the lobby stub. I didn't run a smoke scenario.
  - The flat has one exit, so `back_door` / `back_exit_outside` / `terrace_*` are aliases of the front door. That way `safety.EVACUATION_TARGET` and the sim judge still find them.
  - The chair `[C]` blocks the entry-to-media door, and `[E]` blocks the front door, the only exit (no re-route possible). I didn't test either.
  - Missing fridge, stove, sink and hood meshes get no stand-ins: their stand-in names differ from the house's.
  - Camera, YOLO and the TV doorbell are untested in the apt. The TV is moved to the media-room screen.
  - The **GUI and the dashboard need `NUDGE_HOUSE=apt`** in their own env to draw the apt map. `run_all` sets it for the sim and the brain only.
  - **The apt takes ~25 s to start on the GTX 1650** (compile ~10 s). `test_scenarios.py` waits only 30 s for the sim, so under load the first run failed with "sim never connected". For my run I raised the wait in a scratchpad wrapper and left Sean's file alone. **Suggestion for Sean:** make the 30 s configurable (e.g. `NUDGE_SIM_CONNECT_S`) before running apt scenarios on slow PCs.
- `sim/check_world.py`: the house-only check "chair in `door_guest_living` → reroute" is now skipped when the layout has no such door (3-line guard, no change for the house).

## Tests (this PC, one run each, all via `scripts/locked_test.py`)
| Run | Result |
|---|---|
| `doorbell`, defaults | **PASS 61.4 s** (unchanged) |
| `doorbell` + `--owner-room guest --owner-state asleep_bed` | **PASS 87.5 s** (2 LLM answers + 1 watchdog). A headless sim check shows `owner lies down (in the guest bed)` and the START OPTIONS line |
| `check_world` house | all checks passed |
| `check_world` apt | all checks passed |
| `doorbell` + `--house apt` | **PASS 51.1 s** |
Not run: the full suite (not on this PC), `asleep_sofa` end to end, and the GUI tab live.

## Files
- Changed:
  - `scripts/run_all.py`
  - `scripts/prepare_owner.py`
  - `sim/run_sim.py`: `_start_options_to_env`, `_add_start_option_args`, `_apply_start_option_args`, `NudgeSimulator.apply_start_options` + 1 line in `reset_scene`
  - `sim/world.py`: `_chosen_owner_dir`, `_remove_layout_geoms`
  - `sim/house_layout.py`: `_select_house` at the end
  - `sim/check_world.py`: guard
- New:
  - `gui/panels/startup.py`
  - `sim/layouts/__init__.py`, `sim/layouts/apt.py`
  - this report

For AGENTS.md (parent session / Eden): one line under §7 or §12 ("start-up options: `run_all.py --owner-room/--owner-state/--profile/--owner-wander/--owner/--house`, GUI tab Start-up"), and §4 could mention `sim/layouts/apt.py` as experimental.
