# Agent I: P11 Localisation (particle filter), branch `day/localize`

(Started on P5 apartment; Eden moved me to P11 at ~11:23 because Teo already has the apartment. Nothing of the
apartment work was committed or pushed.)

## What it does
The brain no longer has to read its own pose from the sim. `nudge-brain/localizer_pf.py` is a Monte-Carlo
localiser (particle filter, 400 particles, ~0.7 ms per update at 20 Hz):
- **Predict** from odometry. Default = a leg-odometry model: the pose change between two `state` messages, read
  by a deliberately bad sensor (+4 % scale, 0.9 deg/m yaw drift, slip noise), plus per-particle noise. A particle
  that would walk into a wall stays put. `NUDGE_LOCALIZE_ODOM=cmd` = odometry from the brain's own velocity
  commands only (no sim data at all). That drifts metres when the robot is held up, e.g. behind the owner in the
  long living room (smoke route FAIL offline), so it is not the default.
- **Correct** with the 36-ray LiDAR (`state.lidar_ranges`) on a likelihood field. The field is the distance
  transform of the static nav map (`sim/nav_map.py`, same 0.10-0.55 m band the LiDAR sees). A uniform term in the
  model absorbs the owner and chair, which are not in the map.
- **Resample** (low variance) when the effective sample size is below N/2. **Init** around `ROBOT_START_POSE`.
- The sim's pose is used only for the odometry sensor's input (its readings are biased and noisy) and to
  **measure** the error.

**Flag:** `NUDGE_LOCALIZE=1` (default off; with it off, nothing changes except `localization: null` in the brain status).
With it on, every `state` message goes through the filter before `world.update_state`. Navigation, room, handle
owner position, hearing and the vision depth projection then all use the estimate.

**Brain status (additive):** `localization: {pose, std_xy, err_xy, err_yaw_deg, err_xy_mean, err_xy_max,
odom_only_err_xy, n_eff, fit, n, odom}` or `null`. `err_*` and `odom_only_err_xy` are measured against the sim pose
and are for display only. `fit` = the share of LiDAR hits that land on a mapped wall.

**GUI:** the map draws the estimate as an orange outline robot with a 2-sigma circle and a "self-localised (err x m)"
label. The white triangle is still the true pose.

## Results
Offline: `pixi run python scripts/test_localize.py [--tour doorbell|smoke|long]`. This uses the real sim
kinematics + wall clamping and real `mj_ray` LiDAR, with no ZMQ. The controller drives on the ESTIMATE.
| tour | PF position error mean / p95 / max | heading mean / max | odometry alone (no LiDAR) |
|---|---|---|---|
| doorbell (entry -> living room -> front door, 31 s) | 0.083 / 0.135 / 0.149 m | 0.8 / 3.4 deg | max 1.5 m off |
| smoke (living room -> back door -> outside, 27 s) | 0.086 / 0.144 / 0.152 m | 1.4 / 4.2 deg | 2.6 m off at the end |
| long (5 rooms, 62 s) | 0.084 / 0.161 / 0.206 m | 1.2 / 5.3 deg | **9.4 m** off at the end |

**Live** (whole brain + sim process, flag on, `test_localize.py --live`). This first live run, 11:25 under the lock,
used the earlier **cmd-odometry** version:
| scenario | result | PF error mean / p95 / max | heading mean / max |
|---|---|---|---|
| doorbell | **PASS** 58.9 s | 0.042 / 0.072 / 0.089 m | 0.9 / 4.2 deg |
| smoke | **FAIL** (owner 13 m short of the back exit after 120 s) | 2.83 / 4.15 / 4.40 m | 8.7 / 27.6 deg |

The smoke failure is the reason the default became leg odometry (`delta`). With cmd-odometry the robot gets held
up behind the owner in the long living room, the filter believes the commands, and it slides 2 m along the
featureless x-axis (the west wall is beyond the 4 m LiDAR range). The same route reproduced offline: FAIL with
`--odom cmd`, PASS with the default (table above).
**Live re-run with the default (`delta`)**, 12:04 under the lock, on top of main d26e427:
| scenario | result | PF error mean / p95 / max | heading mean / max | odometry alone |
|---|---|---|---|---|
| doorbell | **PASS** 60.2 s | 0.078 / 0.138 / 0.156 m | 1.1 / 5.7 deg | max 4.05 m off |
| smoke | **PASS** 47.9 s (owner evacuated at the back door) | 0.095 / 0.167 / 0.205 m | 2.1 / 5.8 deg | max 2.60 m off |

**Flag off (default), same run:** doorbell **PASS** 59.7 s, smoke **PASS** 50.6 s, so behaviour is unchanged.

Pitch line: *"Nudge localises itself with LiDAR on its saved map. Its odometry alone would be up to 4 m off after
one doorbell mission; with the particle filter it stays within 16 cm (mean 8 cm)."*

**GUI screenshot** (flag on, doorbell, guiding phase, sim t=39 s): `day/reports/agent-i-localize.jpg`, with the
map crop in `day/reports/agent-i-localize-map.png`. The map shows the orange estimate on top of the white truth,
labelled "self-localised (err 0.08 m)". The two are only 8 cm apart, so they overlap at this scale.

## Not working / limits
- **Kidnapped robot / global init** (`NUDGE_LOCALIZE_GLOBAL=1`, `--kidnap`): FAILS. It does not converge in this
  house: 30 x 22 m, many look-alike rooms, and the LiDAR only covers 180 deg out to 4 m. Next step: more particles
  plus a turn-on-the-spot scan before moving, or augmented MCL.
- Starts from `house_layout.ROBOT_START_POSE`. If a start-up option places the robot elsewhere, the filter must be
  told that pose (call `brain.pf.reset(pose)`).
- The filter is confident (std about 5 cm) and its mean is a little biased by the systematic odometry error, so it
  runs about 8 cm off.

## Commands
```
pixi run python scripts/test_localize.py --tour long             # offline, no lock needed, ~1 min
pixi run python scripts/locked_test.py --cmd python scripts/test_localize.py --live doorbell smoke   # flag on, error per scenario
NUDGE_LOCALIZE=1 pixi run python scripts/locked_test.py doorbell smoke                              # same, plain scenario runner
NUDGE_LOCALIZE=1 pixi run start                                                                     # live, GUI map shows the ghost robot
```

## Merge notes
- Rebased on `origin/main` (d26e427, robot map merged). `main.py` hooks merge cleanly. `NUDGE_ROBOT_MAP=1`
  together with `NUDGE_LOCALIZE=1` has not been tested: the robot-map planner then gets the estimated pose (fine)
  but still plans on its own map.
- `localization` is a new brain-status field (only ever add fields). It is `null` when the flag is off.

## Files
- new: `nudge-brain/localizer_pf.py`, `scripts/test_localize.py`
- hooks (Finn's `main.py`, 16 lines, all behind `if self.pf`): import, `__init__`, `reset`, state hook in `tick`,
  `set_cmd` before `send_cmd`, vision pose in `on_camera`, `localization` in `status()`.
- `gui/nudge_control.py`: +16 lines in the map drawing (Eden).
