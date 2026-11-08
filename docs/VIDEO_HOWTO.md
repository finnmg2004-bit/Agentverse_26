# Video how-to: record the clips, build the montage (< 60 s)

The demo video is a **montage of short scripted clips**, sped up, with a caption bar per scene, a speed badge
("x6"), 0.5 s title cards between scenes and an opening title. Everything is OpenCV (no ffmpeg); **no audio**
in the mp4 (the robot's Piper speech is heard live only; add a voice-over / music in an editor if wanted).

## On the recording laptop (RTX 5060), after the 12:45 freeze
```powershell
git pull; pixi install; pixi run fetch-models        # Whisper + Piper + YOLO + YAMNet; Ollama running (tray icon)
pixi run python scripts/record_clips.py --list        # what will be recorded (ready / NOT READY)
pixi run python scripts/record_clips.py               # all ready clips, ~1.5 min each incl. start-up
pixi run python scripts/record_clips.py --only tv,unknown   # also records clips marked NOT READY
pixi run python scripts/make_montage.py               # -> %USERPROFILE%\nudge-overnight\videos\montage.mp4
pixi run python scripts/make_montage.py --fit 58 --stills 3   # too long? raise all speeds evenly; save 3 PNG stills
```
- Clips land in `%USERPROFILE%\nudge-overnight\videos\clips\<name>.mp4` (+ `<name>.sim/brain/gui.log`). Not in git.
- `record_clips.py` starts the sim (`--no-viewer`) + brain (`--no-dashboard`), waits for `Local LLM online`,
  runs the GUI with `--record ... --record-seconds N --auto ...`, then stops all three. Don't touch the GUI
  window while it records (it captures its own frame buffer, so other windows on top don't matter).
- On a shared laptop wrap it in the test lock: `pixi run python scripts/locked_test.py --wait 900 --cmd python scripts/record_clips.py --only smoke`.
- `make_montage.py` prints the plan and the **total length; it refuses > 60 s** unless `--force`.
  Check the stills (`--stills 3`) or watch the mp4 before submitting.

## Editing the clips
All clip data is the `CLIPS` list at the top of `scripts/record_clips.py` (plain dicts):
| field | meaning |
|---|---|
| `sim` | extra `sim/run_sim.py` args, e.g. `["--house", "apt"]`, `["--profile", "deaf"]`, `["--owner-room", "guest"]`, `["--clock", "21:55"]` |
| `env` | flags, e.g. `{"NUDGE_UNKNOWN_EVENTS": "1"}`, `{"NUDGE_LOCALIZE": "1"}`, `{"NUDGE_ROBOT_MAP": "1"}` |
| `auto` | GUI script, `action@seconds;...`: sim keys (`B K M O C E X R Y J T`), `cam=Whole house` / `Chase cam` / `West wing` / `Top-down` / `Follow robot`, `say=<owner sentence>`, `answer=yes`, `clock=21:59`, `profile=blind`, `sound=<registry sound>`, `gui=<json sim action>` (e.g. play a clip), `mic=on` |
| `duration` | seconds recorded (real time) |
| `trim`, `speed` | `(in_s, out_s)` cut and speed-up used by the montage (`out_s` None = end, negative = from the end) |
| `caption`, `card` | caption bar text; text of the 0.5 s card before the scene (default = caption) |
| `ready`, `in_montage` | `ready=False`: skipped unless named in `--only`; `in_montage=False`: recorded, not in the default montage |

Typical loop: record → watch the clip → set `trim` to skip the idle start/end → rerun `make_montage.py`
(no re-recording needed). Pick scenes / order without editing: `make_montage.py --clips voice,tv,smoke`.
Any video works too: `make_montage.py --seg "take_a.mp4|16|44|4|Doorbell: ..." --seg ...` (`path|in|out|speed|caption`).
Title / end text: `--title "..." --subtitle "..." --end "Rules for safety. Local Gemma for the unknown."`;
`--caption-pos top`, `--card-secs 0` (no cards), `--title-secs 1.5`.

## Clip list (state at 12:00, adapt to what is on `main` at 12:45)
| clip | shows | needs |
|---|---|---|
| `voice` | owner moved [O] to the guest room says "take me to the kitchen" → Nudge goes straight there | owner-voice localisation (main) |
| `tv` | TV doorbell [Y] → "That was the TV"; real doorbell [B] → nudge, speech, guiding | P6 TV investigation (`day/tv-investigate`, NOT READY until merged) |
| `profiles` | deaf → blind profile switch on a doorbell | C10 profiles (main) — not in default montage |
| `night_task` | "every night at 10 pm take me to bed" → clock 21:59 → dark house → to bed | taught tasks + clock (main) |
| `unknown` | glass breaks (no rule) → Gemma reasons → alerts owner | P4 (`day/unknown-events`: gui `play` action + `NUDGE_UNKNOWN_EVENTS=1`), NOT READY until merged |
| `smoke` | smoke alarm → pre-emption → "Shall I call 999?" (auto-answered yes) → back exit blocked [E] → reroute | main |
| `stuck` | chair dropped mid-guide → backs off, re-plans | main — not in default montage |
| `apartment` | `--house apt` doorbell | P5 apartment (main) — not in default montage |
Default montage = voice, tv, night_task, unknown, smoke (~60 s; use `--fit 58` to squeeze).
