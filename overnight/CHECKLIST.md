# Before you sleep: checklist (every overnight PC, ~20 min)

Lanes: Sean's PC = `sean`, Ricktho's PC = `ricktho`, Eden's PC = `integration`, `eden-tasks`, `eden-gui`, `eden-mapping`.

1. **PC**
   - Plugged in; closing the lid does nothing; never sleeps or hibernates on AC (step 3 checks these and prints the fix).
   - Windows Update paused (Settings > Windows Update > Pause updates).
   - Close the MuJoCo viewer, GUI, brain and voice windows.
   - Ollama running in the tray.
2. **Repo** (in your normal checkout):
   ```
   git switch main
   git pull --rebase origin main      # gets overnight/ and the scripts
   pixi install
   pixi run fetch-models              # only if you haven't since Whisper/Piper were added
   ```
3. **Check script:** must end with `0 FAIL`.
   ```
   pixi run python scripts/overnight_check.py --lane <lane> --setup
   ```
   What it does:
   - switches to `overnight/<lane>`
   - installs the push guard (pushes to `main` get blocked)
   - pushes the branch once, proving git can push with no login prompt
   - checks models, the house, ports, Ollama + Gemma speed and power settings
   - runs the `doorbell` scenario once

   If the push asks for a login: do one push by hand so Git Credential Manager remembers you, then re-run the script.
4. **Start the session** in the same folder:
   - Run `claude` and press **Shift+Tab** until the footer says **auto mode**. If your version has no auto mode, start with `claude --dangerously-skip-permissions` instead (the push guard still blocks `main`).
   - Paste:
     ```
     /loop 30m You are overnight lane <lane>. Do one tick of overnight/PROMPT.md.
     ```
5. **Watch the first tick to the end** (10–20 min). It must:
   - run with no permission prompts
   - write `overnight/reports/<lane>.md`
   - push to `overnight/<lane>`; confirm with `git log -1 origin/overnight/<lane>`

   If it got stuck on a prompt: allow it ("don't ask again"), then let the next tick run.
6. **Post in the team chat:** `<lane>: check 0 FAIL, first tick pushed <commit>, sleeping`. Leave the terminal window open.

**Morning:** don't touch anything until Eden has merged. After that, remove the push guard:
```
pixi run python scripts/overnight_check.py --uninstall-hook
```
