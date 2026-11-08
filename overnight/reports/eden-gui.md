# Overnight report: lane `eden-gui` (branch `overnight/eden-gui`)

## Morning notes (keep current: what Eden needs to know to merge)
- Done and tested: E2.1 tasks panel (gui/panels/tasks.py), stub-tested headless; no scenario needed; E2.2 clock buttons + label (nudge_control.py), screenshot start OK
- Behind a flag / experimental:
- Not done:
- Merge risks (files, contract assumptions):
- How to see it working: pixi run gui, "Tasks" tab (fills once the brain sends `tasks`)

## Needs from others
- (lane: what you need, why)

## Contract assumptions / changes proposed
-

## Heartbeat log
- 01:15: E2.1 tasks panel, tested: stub ctx script PASS
01:40: E2.2 clock UI, fc6e16a, tested: GUI --screenshot start PASS
02:15: E2.3 --record/--record-seconds in gui/nudge_control.py (dpg frame buffer -> cv2.VideoWriter mp4v ~13-15 fps; fixed mvBuffer read via np.frombuffer; prints black-frame warning), tested: 6 s GUI-only recording, 77 frames, mean brightness 26 (not black) PASS; no scenarios (GUI-only change)
02:50: E2.4 take A dry run recorded to ~/nudge-overnight/videos/take_a.mp4 (80 s, 1020 frames, ~13 fps, whole GUI window; not in git). Script ~/nudge-overnight/tmp/take_a.py (sim + brain + GUI --auto "R@1;say=carpet@6;B@14;M@52" under the test lock). Brain log: carpet rule at +1.4 s (Gemma 0.5 s), doorbell -> owner acked after ~20 s -> guiding at ~109 s log time -> smoke pre-empt at M (+~15 s into guiding), exit back_exit_outside keeping off the carpet. tested: take_a run PASS (visual check of the video left for the morning)
Morning notes: first run failed only because pixi mangled an apostrophe in --auto; use plain sentences. Ollama must be up before the brain starts (it was offline at first launch, online later).
