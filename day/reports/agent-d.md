# Agent D report: owner voice (#10), branch `day/owner-voice`

**Goal:** when the owner says anything to Nudge (GUI box, tests, voice mic → `user_command`), the sim plays the
owner's voice from where they stand into the two ears. The brain hears the speech, localises it (direction and
loudness, like any sound) and updates its owner belief, so the search goes to the right room first.

## Done
- **Rebased on origin/main after Ricktho's review.** `nudge-brain/hearing.py` is now **exactly main's version**: his
  speech gate (`9bc974f`), `result["speech"]` and `speech_gated`. My own damping was dropped, as the review asked.
  `sim/sensors.py` keeps main's `SpeakerOut` lines next to the `_mix_source` refactor. The owner's voice is NOT
  played on the speakers (`speaker.update(active_list)` only gets registry sounds).
- `nudge-brain/sound_localizer.py`: accepts the label `"owner_voice"` (level and reference RMS from `OWNER_VOICE` and
  its wav; the loudness sigma is wider, 0.7, because speech has pauses). Registry sounds are unchanged.
- `nudge-brain/owner_belief.py`: `hear(room_prob)` takes a snapshot when the utterance starts, then
  p ∝ snapshot × (0.05 + room_prob). It is idempotent within an utterance. `end_utterance()` resets it. Main's
  `CLEARED_*` constants are kept.
- `nudge-brain/owner_voice.py` (new): `OwnerVoice`. It takes windows with speech ≥ 0.5 and no registry sound ≥ 0.35.
  Scores listed in `speech_gated` are ignored, because they are the voice itself. It feeds the localiser and the
  belief, and ends the utterance after 3 s without speech. It logs one line per utterance:
  `OWNER VOICE: heard the owner from the west -> probably guest (P 0.55)`.
  The C6 status is `{"room","p","xy","t"}` plus `voice_room`, `n`, `room_probs`.
- `nudge-brain/main.py` (small additive hooks, for Finn): an import, an init line, a reset line, one call in
  `on_audio`, a status field `owner_voice`, and the helper `hear_owner_voice()`. The helper also re-routes a running
  `go_to_room` search when the voice points to another best room with P ≥ 0.5.
- `scripts/test_scenarios.py`: new scenario **`owner_voice`**, in the default suite. Steps: `O` moves the owner to the
  guest room, then "Nudge, please take me to the kitchen". It expects `OWNER VOICE` + `probably guest` +
  `Owner seen in guest`, forbids `go_to_room(living_room)` and any `SOUND start`, and passes once the owner stays in
  the kitchen for 2 s or more.
- `scripts/test_owner_voice.py`: it fails only on a **false mission** (a sound ≥ 0.5 from the voice alone).
- `gui/nudge_control.py`: the map shows a pink "heard owner: <room> (P x)" ring at `owner_voice.xy`. It is drawing
  code only and fades after 30 s.

## Tested (after the rebase on main)
- Offline `test_owner_voice.py --noise quiet fan tv vacuum party`, with main's gate: **no false mission in quiet or
  fan.** A few residual windows sit at 0.35–0.40 (below the cap, so not gated, and below the 0.5 mission threshold).
  There are 3 false missions under babble-like noise, where YAMNet speech is < 0.5 so the gate is off: tv/entry
  doorbell 0.52, vacuum/entry knock 0.70, party/entry knock 0.57.
- Scenarios: see `RESULTS` at the end of this file.

## Not done / limits
- TV / vacuum / party noise: see above. That belongs to the hearing owner (gate threshold). Default noise is `quiet`.
- Localisation from one robot pose is front/back ambiguous and has a living-room loudness mismatch (sim walls vs
  localiser walls). Correct top room in 3/6 (quiet) from a single pose. The belief uses it as soft evidence (5 %
  floor), and the guest room (the demo) is clear at 0.96.

## Merge risks
- `main.py`: about 20 additive lines. `owner_belief.py`: additive, and the rebase conflict with main's `CLEARED_*`
  lines was resolved by keeping both.
- `hearing.py`: no diff vs main.

## How to see it
`pixi run python scripts/locked_test.py owner_voice -v` prints `OWNER VOICE ... probably guest` and then
`SKILL go_to_room(guest)`. In the GUI: press `O` (owner to the guest room), type anything in "Tell Nudge". A pink ring
appears at the guest room.

## RESULTS (rebased on origin/main 239f9ed, hearing.py = main's)
- `owner_voice`: **PASS 2/2** (77 s, 80 s). In both runs the command parse (~0.9 s) beat the voice localisation, so
  the robot set off to the living room. It turned about 1 s later: `OWNER VOICE: heard the owner from the west ->
  probably guest (P 0.55)`, then `OWNER VOICE: searching guest instead of living_room`. It never scanned the living
  room. In an earlier run, when the voice won the race, it went straight to `go_to_room(guest)`. The scenario checks
  for no `look_around`, no `SOUND start` and `Owner seen in guest`.
- Regression set `doorbell carpet_rule carpet_rule_voice owner_moved smoke demo`: **all PASS**.
- Follow-up idea (not done): hold the first search decision of a SPOKEN request for ~1.5 s until the voice is
  localised (agent.py / main.py, Finn's lane). Then it goes straight to the right room every time.
- Branch history: no force push (it was denied). The old pre-rebase tip is merged in with `-s ours` (tree = the
  rebased branch), so `day/owner-voice` is a fast-forward. For a linear history, Eden can cherry-pick the 4
  non-merge commits on top of main instead.
