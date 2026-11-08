# Agent K report: demo video as a montage (branch `day/montage`)

## What's on the branch
- `scripts/record_clips.py` — the clip list is plain data at the top (`CLIPS`: sim args, env flags, GUI `--auto`
  script, duration, trim, speed, caption, card, `ready`, `in_montage`). Per clip: sim `--no-viewer` + brain
  `--no-dashboard` → waits for `Local LLM online` → GUI `--record clips/<name>.mp4 --record-seconds N --auto ...`
  → stops all three → checks the mp4 has frames. `--list`, `--only a,b`, `--dry-run`, `--out`.
  Output + logs: `~/nudge-overnight/videos/clips/` (not git).
- `scripts/make_montage.py` — trim, speed-up (frame skipping), caption bar (bottom, 85 % dark band, large white
  text, up to 2 lines), yellow speed badge ("x4"), opening title card, 0.5 s card before each scene, optional end
  card; one mp4 at the clips' resolution (OpenCV `avc1` = H.264 works on this laptop, falls back to `mp4v`).
  Prints the total, **refuses > 60 s unless `--force`**, `--fit 58` raises all speeds evenly, `--stills N`
  saves PNGs from the finished file. Input either the clip list (`--clips voice,tv,smoke`) or any videos
  (`--seg "path|in|out|speed|caption"`).
- `gui/nudge_control.py` (small, additive, GUI owner please note): `--auto` gets `sound=<registry sound>` and
  `gui=<json sim action>` (needed for the P4 glass clip: `gui={"action": "play", "file": "unknown/glass_break.wav", ...}`).
- `docs/VIDEO_HOWTO.md` — exact commands for the recording laptop + how to edit the clip list.

## Tests
- Montage from the overnight takes A/B (4 scenes: carpet rule x2, doorbell x4, bedtime task x4, smoke x3, + end card):
  `~/nudge-overnight/videos/montage_test.mp4`, **34.3 s**, 1614x950, 15 fps, H.264, 49 MB.
  Stills: `montage_test_still1..3.png` + `montage_test_title.png` (same folder) — captions readable.
- Refusal: a 70.5 s plan is refused; `--fit 20` squeezes a 27 s plan to 19.8 s.
- Live clip under the test lock (`locked_test.py --wait 900 --cmd python scripts/record_clips.py --only smoke`,
  60 s version): sim + brain + GUI started and stopped cleanly, `Local LLM online`, **761 frames (50.7 s at the
  file's 15 fps), real picture** (mean brightness 60-75 after the first 2 s). Brain log: smoke heard 0.99 →
  SAFETY PRE-EMPT → owner found by YOLO in the living room → 2nd nudge → Gemma "Back exit is much safer from the
  fire" → guiding at 34 s → "Shall I call 999?" at 36 s. Kept as `~/nudge-overnight/videos/smoke_take1.mp4`;
  `make_montage.py --clips smoke` on it → `montage_cliplist_test.mp4` (10.5 s) OK.
  From that log the smoke clip is now 90 s with `answer=yes` at 30-57 s and [E] at 40 s (after guiding starts);
  the re-record of that version **did not run**: the shared lock was busy for 15 min (`gave up after 900 s`).
- Note: the GUI's `--record` writes ~12.7 frames per real second into a 15 fps file, so clips play ~1.2x fast
  already; the badge shows the montage factor on top of that (harmless, but "x6" is really ~x7 of real time).
  The first ~2 s of every clip are the dark "Waiting for the sim" screen: keep `trim` in >= 3 s.

## Clip status (12:00)
`voice`, `profiles`, `night_task`, `smoke`, `stuck`, `apartment` use features on main. `tv` (P6, `day/tv-investigate`)
and `unknown` (P4, `day/unknown-events`: `play` action + `NUDGE_UNKNOWN_EVENTS=1`) are `ready=False` until merged.
Default montage: voice, tv, night_task, unknown, smoke. Timings in the `auto` scripts are first guesses: watch each
clip once and adjust `trim` / `speed` (no re-recording needed).

## Left
- Record all clips on the RTX 5060 after the freeze; tune trims; check the total (<= 60 s).
- No audio in the mp4 (no ffmpeg). Voice-over / Piper speech would need an editor or ffmpeg.
