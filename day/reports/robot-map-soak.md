# Robot map soak (`NUDGE_ROBOT_MAP=1`), test 3 of the robot-map decision (AGENTS.md 12.7)

**Verdict: reliable: switch on for the demo** (as far as test 3 goes). Soak 9/9 PASS, no flaky runs, unchanged `main` code. Still needs Sean's tests 1 + 2 to pass too (the team rule).

- **Code:** `main` at `48e3e2f` (branch `claude/robot-map-soak-14609c`, no code changes).
- **Machine:** Teo's GTX 1650 dev box, 2026-11-08 12:02–12:14, one sim at a time through `scripts/locked_test.py`.
- **Command:** `NUDGE_ROBOT_MAP=1 NUDGE_LLM=off pixi run python scripts/soak.py -n 3 doorbell smoke smoke_blocked_exit`. Decisions came from the plain-code rules (`NUDGE_LLM=off`): the map changes navigation, not decisions, and Gemma is slow on this GPU.
- **Map:** the committed `maps/robot_map.npz`. This PC has no `assets/maps/robot_map.npz`, so the brain loaded the same map every other PC gets. Default house, no `--coco-objects`.

## Baseline (unchanged code)
| scenario | pass rate | rounds | avg time |
|---|---|---|---|
| `doorbell` | 3/3 | PASS PASS PASS | 59 s |
| `smoke` | 3/3 | PASS PASS PASS | 39 s |
| `smoke_blocked_exit` | 3/3 | PASS PASS PASS | 81 s |

Each round took ~236 s; 3 rounds took 12 min. The flag reached the brain in every run: the brain's robot-map change log (`[robot_map] CHANGE ...`) is in all 9 runs. In `smoke_blocked_exit` the LiDAR marked `exit_back` blocked, the robot re-planned, and the owner got outside.

## Fixes
**None.** The baseline was green, so the code is unchanged and there was no after-fix soak. The flag-off doorbell check was not needed either.

**Owner shows up as a change (known quirk), left as is.** In all 9 runs, before the robot first saw the owner, the owner at the living-room spot showed up as a new obstacle near (-10.7, 4.8) for ~8–13 s. It was cleared once the owner was seen (`... gone`). No run failed or slowed down because of it. I did not try to filter it:
- To the LiDAR, the owner and the chair are both ~0.4–0.5 m blobs. Dropping person-sized changes would also drop the chair, and spotting the chair is what change detection is for.
- The handoff asked to skip the quirk if the baseline was green.

## After-fix soak
Not run (no code changes). The baseline table above is the result.

## Not covered here
- Tests 1 + 2: the full flag-on suite with and without `--coco-objects` (Sean, RTX 5060).
- Gemma deciding (`NUDGE_LLM=off` here).
- The `--coco-objects` house (moved sofa).
- Live GUI test 5.
