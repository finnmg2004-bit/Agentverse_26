# Overnight report: lane `eden-mapping` (branch `overnight/eden-mapping`)

## Morning notes (keep current: what Eden needs to know to merge)
- Done: E3.5 change detection in robot_map.py: hits inside confidently-free space (no occupied cells within 5x5) become `changes` [{kind,x,y,cells,t}] in the :5559 header + a `[robot_map] CHANGE` log line. Only synthetic-checked (static wall = 0 changes, chair = 1 change); live sim `[C]` run and false-positive rate from pose jitter not measured. GUI panel does not draw changes yet.
- Done: E3.2 `nudge-brain/object_memory.py` (multi-class YOLO + depth -> objects, room naming) wired into robot_map.py via `--objects` (objects+rooms in the :5559 header). Live headless run from the start pose: pipeline OK, 0 objects detected (entry hall has only box stand-ins; real detection rate unknown until E3.3 tour). Room naming unit-checked on synthetic detections; a wide kitchen splits into 2 clusters (CLUSTER_DIST 3.5 m).
- Done and tested: E3.1 `nudge-brain/robot_map.py` (own process, :5559 `robot_map`); unit-checked on synthetic LiDAR only, not yet against a live sim
- Behind a flag / experimental:
- E3.3 test_mapping.py: object detections are weak (YOLO labels furniture stand-ins as tv; duplicates tv_1-3, one outside the house); room naming is not reliable. Map walls are good.
- Not done:
- Merge risks (files, contract assumptions):
- How to see it working:

## Needs from others
- (lane: what you need, why)

## Contract assumptions / changes proposed
-

## Heartbeat log
- 00:57: E3.1 robot_map.py written (log-odds grid, PNG pub 1 Hz, resets on episode change), tested: synthetic unit check PASS; live-sim run not done
- 01:25: E3.2 object_memory.py + robot_map --objects, tested: synthetic unit check PASS; live sim run PASS (runs, 0 detections), see notes
- 01:56: E3.3 scripts/test_mapping.py (tour entry,dining,living_room,guest; known 75.6%, wall precision 97.0%, recall 95.0%; objects: tvs are false positives (3 'tv' incl. one outside house), see notes), tested: test_mapping PASS
- 02:25: E3.4 gui/panels/robot_map.py (own :5559 subscriber, grid texture + robot + objects), tested: module import + coord unit check PASS; live GUI render not run (needs sim+robot_map+gui)
- 02:55: E3.5 change detection (appeared obstacles -> header `changes` + log), tested: synthetic unit check PASS; live sim [C] not run
