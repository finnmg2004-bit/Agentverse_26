# Overnight session prompt (all lanes)

Start a session in the lane's checkout with:

    /loop 30m You are overnight lane <LANE>. Do one tick of overnight/PROMPT.md.

`<LANE>` = `sean`, `ricktho`, `eden-tasks`, `eden-gui`, `eden-mapping`; E0 uses
`/loop 60m You are overnight lane integration. Do one tick of overnight/PROMPT.md.`

---

You are an unattended Claude Code session on the Nudge team's overnight run. No human will answer
until ~07:00, so never ask questions and never wait for input: pick the sensible option, note it in your
report, move on. Project context: `AGENTS.md`. Overnight rules, lanes, files and contracts:
`overnight/README.md`. Your work list: `overnight/QUEUE-<pc>.md` (`sean`, `ricktho`, or `eden` for all
`eden-*` lanes and `integration`), your lane's section only.

## One tick
1. Look at the local time. **After 06:15: do nothing except a final push if anything is uncommitted, then end the tick.** After 06:00: finish only the current item, then write the final report.
2. Write the time to `~/nudge-overnight/heartbeat-<lane>.txt`. If your previous heartbeat is > 50 min old, switch to low-burn (`overnight/README.md`, "Pacing"). If `~/nudge-overnight/state.json` says `"mode": "low"`, follow the low-burn rules (lane `eden-mapping`: just end the tick).
3. `git fetch -q origin`; read `git show origin/overnight/integration:overnight/STATUS.md` (may not exist yet). Problems addressed to your lane come before the queue.
4. Make sure you are on branch `overnight/<lane>` (create it from `origin/main` if missing). Never work on `main`.
5. Do the top unchecked queue item (or a working slice of it): keep to your lane's files, keep `main`'s behaviour unchanged unless the item says otherwise, and put risky behaviour behind a flag / `experimental` scenario.
6. Test with `pixi run python scripts/locked_test.py <scenario...>` (waits for the PC's test lock). Unit tests or a quick script are fine for code with no scenario. A failure you can't fix this tick: revert or flag it off, and write it down.
7. Commit only your files with `git commit -F <msgfile>` (message: what + why + how tested). **No Claude/AI attribution, no `Co-Authored-By` lines.** Then `git push origin HEAD:overnight/<lane>`.
8. In `overnight/reports/<lane>.md` (create it from `overnight/reports/TEMPLATE.md` if missing): tick the queue item in your branch's queue copy, add one heartbeat line `HH:MM: <what>, <commit>, tested: ... PASS/FAIL`, and keep "Needs from others" / "Morning notes" current. Commit + push that too.
9. End the tick. Don't start a second item in the same tick.

## Hard rules
- Never push to `main`, never `--force`, never `git push` without an explicit `HEAD:overnight/<lane>` target. A pre-push hook enforces this; if it blocks you, you made a mistake, so don't work around it.
- Never edit another lane's files, `AGENTS.md`, `TODO.md` or `worlds/nerv-world/`. Never commit `.env`, keys, model weights or videos.
- Never kill processes you didn't start, except stray `run_sim.py`/`main.py`/`nudge_control.py`/`voice.py` holding ports 5555–5559 when you hold the test lock.
- Runtime stays 100 % local (no cloud API calls in the sim/brain code).
- If something outside your lane is broken and blocks you, skip to the next item and write it under "Needs from others".
- Keep token use modest: read only the parts of files you need, avoid re-reading big files every tick, use subagents only when they clearly save work.
