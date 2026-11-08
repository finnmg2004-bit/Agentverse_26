# Adaptability benchmark (randomised, unseen runs)

- **Generated:** 2026-11-08 12:30 local time (benchmark started 2026-11-08 11:27); script `scripts/benchmark_adapt.py`
- **Seed:** 2026; **planned runs:** n = 24; **completed:** 12 (the rest were not run: time limit)
- **Machine:** RTX 5060 laptop (Windows 11), everything local (sim + brain on the same PC, no cloud calls)
- **Per-run limit:** 150 s wall clock (180 s when the owner is asleep)
- **Decision maker:** local LLM (Gemma via Ollama) online in 11/12 runs; per-step decisions are a mix of LLM and deterministic rules (rules take over when the LLM is slow/offline)

The configurations are **randomised and not hand-tuned**: each run draws the owner's room (one of the six rooms with a verified owner spot) and, on half the runs, a random furniture-free spot in that room (>= 1.0 m clearance, >= 1.5 m from any doorway); background noise; the chair dropped into a random narrow doorway on ~1/3 of runs (else stowed); the household sound; the owner asleep in the guest bed on ~15 % of runs. Each run uses the **same pass conditions as the hand-written scenario for that sound** in `scripts/test_scenarios.py` (sim reports SCENARIO COMPLETE = owner delivered / evacuated, plus the expected brain log lines and vibration pattern; asleep runs must also wake the owner). Nothing was re-run or cherry-picked: this is the full list for this seed.

## Summary

- **Overall: 9/12 (75 %)**
- Excluding the known-weak combination knock + tv/vacuum/party (docs/hearing-eval.md): 9/12 (75 %)
- Known-weak runs (knock under tv/vacuum/party), reported separately: -
- Median run time (passed runs): 80 s

### By sound

| sound | pass rate |
|---|---|
| doorbell | 2/2 (100 %) |
| kettle | 2/2 (100 %) |
| oven_timer | 2/3 (67 %) |
| smoke | 3/5 (60 %) |

### By noise

| noise | pass rate |
|---|---|
| quiet | 1/1 (100 %) |
| fan | 1/1 (100 %) |
| tv | 4/4 (100 %) |
| vacuum | 2/5 (40 %) |
| party | 1/1 (100 %) |

### By owner room

| room | pass rate |
|---|---|
| dining | 1/1 (100 %) |
| family_room | 1/2 (50 %) |
| guest | 3/3 (100 %) |
| kitchen | 3/5 (60 %) |
| living_room | 1/1 (100 %) |

### Obstacles and owner state

| condition | pass rate |
|---|---|
| chair in a doorway | 0/1 (0 %) |
| chair stowed | 9/11 (82 %) |
| owner at a random spot | 2/5 (40 %) |
| owner at the room's usual spot | 5/5 (100 %) |
| owner asleep | 2/2 (100 %) |

## All runs

| # | sound | noise | owner | chair | result | time | failure reason |
|---|---|---|---|---|---|---|---|
| 0 | doorbell | tv | kitchen (usual spot) | stowed | PASS | 108 s |  |
| 1 | smoke | party | kitchen (usual spot) | stowed | PASS | 141 s | passed late: completed after the sim's 120 s 'unresolved' mark |
| 2 | smoke | vacuum | kitchen (7.22, 3.51) | door_kitchen_garden | **FAIL** | 150 s | [sim] SCENARIO FAILED: smoke_alarm unresolved after 120s (owner at 1.21, -0.55, 16.0 m from back_exit_outside).; sim did not report SCENARIO COMPLETE for ['smoke_alarm'] within 150s |
| 3 | smoke | tv | asleep (guest bed) | stowed | PASS | 74 s |  |
| 4 | oven_timer | quiet | kitchen (usual spot) | stowed | PASS | 75 s |  |
| 5 | smoke | vacuum | kitchen (7.32, 3.89) | stowed | **FAIL** | 150 s | [sim] SCENARIO FAILED: smoke_alarm unresolved after 120s (owner at 1.40, -0.60, 16.2 m from back_exit_outside).; sim did not report SCENARIO COMPLETE for ['smoke_alarm'] within 150s |
| 6 | doorbell | vacuum | living_room (usual spot) | stowed | PASS | 62 s |  |
| 7 | smoke | vacuum | family_room (usual spot) | stowed | PASS | 51 s |  |
| 8 | oven_timer | vacuum | family_room (-12.85, 15.77) | stowed | **FAIL** | 150 s | [sim] SCENARIO FAILED: oven_timer unresolved after 120s (owner at -12.85, 15.77, 28.5 m from stove).; sim did not report SCENARIO COMPLETE for ['oven_timer'] within 150s; missing brain log: ['alert_owner(oven_timer', 'gu |
| 9 | kettle | fan | guest (-8.37, -1.96) | stowed | PASS | 101 s |  |
| 10 | kettle | tv | dining (-4.0, 3.01) | stowed | PASS | 80 s |  |
| 11 | oven_timer | tv | asleep (guest bed) | stowed | PASS | 151 s | passed late: completed after the sim's 120 s 'unresolved' mark |

Reproduce: `python scripts/benchmark_adapt.py --n 24 --seed 2026` (each run takes the PC test lock separately via scripts/locked_test.py; do not wrap the driver itself in the lock). Per-run logs: `~/nudge-overnight/benchmark_adapt/seed2026_<i>.log`.
## Findings (hand-written after the run; regenerating the report drops this section)

- **Run order:** runs 0-6 ran 11:27-11:43, then the benchmark paused so a teammate's robot-map test job could use the PC test lock, and runs 7-11 ran 12:20-12:30. Every run took the lock, so no run shared the PC with another sim. The run stopped at the time limit after 12 of the 24 planned configurations. With n = 12 the rates are rough (one run moves a rate by ~8 points).
- **Smoke alarm with the owner in the kitchen (runs 2, 5): fails.** The fire is in the kitchen, so Nudge leads the owner away from it towards the default evacuation target (back door, living room). That is the whole length of the house, and at 120 s the owner is still in the entry hall, about 1 m from the **front** door. A nearest-safe-exit choice (front door) would very likely pass. Run 1 (same room, usual spot) only just made it (141 s). This is a real policy weakness, not noise.
- **Owner in the far corner of the large family room (run 8): not found in time.** The owner was at (-12.85, 15.77), about 10 m from the room's scan point, and no alert was ever given (9 LLM decisions, all search). The owner search does not cover big rooms well.
- **Two passes were slow** (runs 1 and 11): they completed inside the harness limit but after the sim's own 120 s "unresolved" mark.
- **What held up:** every doorbell, kettle and asleep-owner run passed, all four TV-noise runs passed, and every owner at a room's usual spot was reached (5/5). All three failures were under vacuum noise, but the sound was still heard every time: the failures came from routing/search, not hearing. Only one run had the chair in a doorway (it failed for the kitchen-smoke reason above), so it says nothing about chair handling. No knock run came up for this seed.
