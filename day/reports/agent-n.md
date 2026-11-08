# Agent N: global localisation + kidnapped robot, branch `day/global-localize`

Builds on agent I's particle filter (`nudge-brain/localizer_pf.py`). Everything new is behind
`NUDGE_LOCALIZE_GLOBAL=1` (on top of `NUDGE_LOCALIZE=1`). With the flag off, tracking mode is unchanged: the offline
smoke tour gives the same numbers as agent I's (mean 0.086 / max 0.152 m).

## What it does
- **Start-up look-around:** with the flag on, the filter starts in mode `"global"`. `pf.wants_spin()` makes the
  brain turn once on the spot at 1.2 rad/s (about 5.5 s; a 2-line hook in `main.py` before `send_cmd`). This is the
  only change to behaviour, and it happens only at start-up. During the turn, LiDAR hits and the points each beam
  passed through (free space) are collected in the body frame. Odometry moves them along with the robot.
- **Global search:** an exhaustive scan match of the buffered 360 deg scan against every free 25 cm cell x 72
  headings of the house (about 7k cells x 72). It uses a padded likelihood field (sigma 0.22 m) plus a penalty
  for beams that would pass through a wall. One search takes **0.35-0.6 s** (it was 5-7 s before the padding).
  The 300 best hypotheses x 12 particles seed the filter, weighted by a tempered prior.
- **Convergence test:** `converged` = std_xy < 0.3 m and fit > 0.7 for 2 s, **and a uniqueness check**: the
  buffered scan must fit at least 6 log-units better within 0.75 m of the estimate than at the best place more
  than 1.5 m away. This is needed because `fit` alone is useless here: a wrong mode 4-8 m away scored fit 0.998.
  If the check fails, the filter searches again from all modes. While std > 1 m or fit is low, it re-searches
  every 4 s.
- **Kidnap recovery (flag on, while tracking):** if fit stays below 0.3 for 2 s, the robot is lost. The filter
  goes back to mode `"global"` and runs a search on the scans collected since the fit dropped (no spin needed,
  it searches while driving). A pose jump of more than 0.5 m in one 50 ms step is not counted as leg odometry
  (a robot that is picked up does not feel it in its legs).
- **Brain status `localization`** adds the fields `mode` ("global"|"tracking"), `converged`, `spinning`,
  `converge_s`, `relocalizations` and `margin`. The existing fields are unchanged, and `n` is now the current
  particle count.

## Results (offline, `scripts/test_localize.py`, real sim kinematics + `mj_ray` LiDAR, controller drives on the estimate)
| test | result | time to converge | error |
|---|---|---|---|
| `--teleport --trials 6 --seed 3` (known start; after 12 s the robot is put down at a random spot in a random room) | **6/6 recovered** (PASS) | median 11.4 s, max 26.6 s after the drop | at convergence 0.03-0.12 m, final mean 0.09 m |
| `--teleport --trials 2` (earlier run, seed 0) | 2/2 | 8.3 / 17.0 s | 0.22 / 0.13 m |
| `--kidnap --trials 6` (global start at a random pose, spin, then the doorbell tour) | **3/6 converged, all 3 correct**, 0 false convergences (FAIL, the gate is 80 %) | median 11.6 s, max 17.3 s from start (includes the 5.5 s spin) | final 0.08 m |
| `--kidnap --trials 4`, earlier version without the uniqueness check | 2/4 correct, **2 converged on the wrong place** (4 m and 8.5 m off) | | |

In the 3 global-start failures (a start in the gallery, two in the family room) the filter stays
`mode: "global", converged: false` for the whole 75 s and keeps searching (7-17 searches). It never claims a wrong
pose. These rooms are large and open, and from there the 4 m LiDAR sees too few walls to tell them apart from
look-alike rooms. In one of them the estimate was in fact right (0.31 m) but did not pass the uniqueness check.
When the robot is kidnapped while tracking it recovers more easily, because it keeps driving and sees new walls.

**Regression, flags off** (`scripts/test_scenarios.py --port-base 7255 doorbell smoke`, 13:01): **doorbell PASS,
smoke PASS** (49.8 s).
**Live, `NUDGE_LOCALIZE=1 NUDGE_LOCALIZE_GLOBAL=1` doorbell (port 7265, 13:03): FAIL.** The filter never
converged (`mode: global, converged: false`, err 10.3 m, 3600 particles, so at least one search ran) and the
mission did not finish in 120 s. **The same start pose offline (entry, `ROBOT_START_POSE`) converges in 10.3 s, err
0.04 m**, so this is an integration problem in the live loop, not the method. I did not have time to debug it.
Suspects: whether the spin is really executed live (`spinning` in the status, sim clamps), the state rate / `t`
during the start-up spin, and the brain's episode reset calling `pf.reset()` again in the middle of the spin.
**Do not turn on `NUDGE_LOCALIZE_GLOBAL=1` for the demo.**

## Limits / next
- Global start in a large open room (gallery, family room) often does not converge within 75 s. Ideas: more
  look-around turns when it is not converged, driving towards walls / doorways (active localisation), a finer
  search around the top modes, and a longer LiDAR range or a rear LiDAR.
- The search blocks the brain loop for about 0.5 s when it runs. That is fine at start-up, but a background thread
  would be better for recovery.
- The brain does not yet act on `converged == false` after the spin: it plans on the estimate anyway. The GUI
  could show "localising..." from `localization.mode`.

## Commands
```
pixi run python scripts/test_localize.py --kidnap --trials 10      # global start, random poses
pixi run python scripts/test_localize.py --teleport --trials 10    # kidnapped mid-tour, recovery
pixi run python scripts/test_localize.py --tour smoke              # tracking (unchanged)
NUDGE_LOCALIZE=1 NUDGE_LOCALIZE_GLOBAL=1 pixi run python scripts/test_scenarios.py --port-base 7265 doorbell
```

## Files / merge notes
- `nudge-brain/localizer_pf.py` (global search, buffer, convergence + uniqueness, lost detection, teleport
  guard, status fields), `scripts/test_localize.py` (`--kidnap` reworked into random-start trials, new
  `--teleport`, `--trials`), `nudge-brain/main.py` +2 lines (Finn's file: the spin override behind `self.pf and
  self.pf.wants_spin()`, which is never true with the flag off).
- `pf.reset()` with no arguments (the brain's episode reset) does a global init again when the flag is on.
