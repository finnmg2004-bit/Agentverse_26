# Agent D report: owner voice (#10), branch `day/owner-voice`

**Goal:** when the owner says anything to Nudge (GUI box, tests, voice mic → `user_command`), the sim plays the
owner's voice from where they stand into the two ears. The brain hears the speech, localises it (direction and
loudness, like any sound) and updates its owner belief, so a later search goes to the right room first.

## Done
- **Rebased** onto `origin/main` (Eden's WIP commit on top; nothing conflicted).
- `nudge-brain/hearing.py`: **speech damping** (the KEY RISK). The owner talking alone scored doorbell up to 0.67,
  knock 0.97 and oven timer 0.58 (one window ≥ 0.5 starts a mission). When YAMNet hears speech and does not hear
  that sound itself, the *spectral* score of doorbell / knock (and an oven timer weaker than tonality 40; the real
  timer is ≥ 82) is scaled down to ×0.2 (ramp over speech 0.15–0.5). **A tier-1 sound (smoke alarm) is never
  damped.** A real doorbell or knock masks the voice for YAMNet (speech ≤ 0.06 while the owner talks over it), so
  real sounds still score 0.97–0.98 with the owner talking. `NUDGE_SPEECH_DAMP=0` turns it off.
- `nudge-brain/sound_localizer.py`: accepts the label `"owner_voice"` (level and reference RMS from `OWNER_VOICE` and
  its wav; the loudness sigma is wider, 0.7, because speech has pauses). Registry sounds behave exactly as before.
- `nudge-brain/owner_belief.py`: `hear(room_prob)` takes a snapshot when the utterance starts, then
  p ∝ snapshot × (0.05 + room_prob). It is idempotent within an utterance. `end_utterance()` resets it.
- `nudge-brain/owner_voice.py` (new): `OwnerVoice`. It takes windows with speech ≥ 0.5 and no registry sound ≥ 0.35,
  feeds them to the localiser and the belief, and ends the utterance after 3 s without speech. It logs one line per
  utterance (another only if the room changes):
  `OWNER VOICE: heard the owner from the west -> probably guest (P 0.55)`.
  The C6 status is `{"room","p","xy","t"}` plus `voice_room`, `n`, `room_probs`.
- `nudge-brain/main.py` (small hooks only, for Finn): an import, an init line, a reset line, one call in `on_audio`,
  a status field `owner_voice`, and the helper `hear_owner_voice()`. The helper also **re-routes** a running
  `go_to_room` search when the voice points to a different best room with P ≥ 0.5 (log
  `OWNER VOICE: searching X instead of Y`). Nothing goes into `world.sound_estimates`.
- `scripts/test_scenarios.py`: new scenario **`owner_voice`**, in the default suite. Steps: `O` moves the owner to the
  guest room, then "Nudge, please take me to the kitchen". It expects `OWNER VOICE` + `probably guest` +
  `Owner seen in guest`, forbids `go_to_room(living_room)` and any `SOUND start`, and passes once the owner stays in
  the kitchen for 2 s or more.
- `scripts/test_owner_voice.py`: it now fails only on a **false mission** (≥ 0.5). A wrong room from one static pose is
  reported but is not a failure.
- `gui/nudge_control.py`: the map shows a pink "heard owner: <room> (P x)" ring marker at `owner_voice.xy`. It is
  drawing code only and fades after 30 s.

## Tested
- `scenario owner_voice`: **PASS 2/2** (77 s). The voice is localised before the command is parsed (LLM 1.2 s), so the
  first search goes straight to `go_to_room(guest)`.
- Regression `doorbell carpet_rule carpet_rule_voice owner_moved smoke demo`: see the result below.
- Offline `test_owner_voice.py --noise quiet fan tv vacuum party`: **0 false sounds ≥ 0.35 in quiet and fan**. Rooms
  from one pose with no movement: quiet 3/6 and fan 3/6 correct top room. The guest room is right in quiet (0.96).
  Rooms that are wrong are front/back mirrors or show a loudness model mismatch.

## Not done / limits
- **TV / party / vacuum noise profiles:** YAMNet barely hears speech under babble (speech ≈ 0–0.3). The voice can
  still give doorbell or knock 0.35–0.63 there. 2 of 18 owner spots go ≥ 0.5 (owner standing in the entry, knock).
  The default noise is `quiet`, so the demo is not affected.
- **Localisation from one pose** is front/back ambiguous, and there is a sim-vs-localiser wall-count mismatch for the
  living room (measured loudness is about 2.2× the model). The belief treats the result as soft evidence (5 % floor).
- Real-world note: a real mic would need the same damping for real speech. In the sim the clip is TTS.

## Merge risks
- `main.py`: about 20 lines, all additive (Finn edits this file today; the conflicts should be trivial).
- `hearing.py`: the damping sits inside the YAMNet fuse loop, so it touches Ricktho's file (review please). Without
  YAMNet, nothing changes.
- `sensors.py` / `run_sim.py` are unchanged since Eden's WIP commit (the `voice_pos` signature is kept for agent B).

## How to see it
`pixi run python scripts/locked_test.py owner_voice -v` prints the `OWNER VOICE` line and `SKILL go_to_room(guest)`.
In the GUI: press `O` (owner to the guest room), type anything in "Tell Nudge". A pink ring appears near the guest
room.
