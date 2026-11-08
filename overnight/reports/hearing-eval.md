# Hearing evaluation: findings (Teo, 2026-11-08)

Full generated report with every table: [`docs/hearing-eval.md`](../../docs/hearing-eval.md); chart:
[`docs/img/hearing-eval.png`](../../docs/img/hearing-eval.png). Re-run: `pixi run fetch-esc50` once, then
`pixi run hearing-eval --n 1000 --seed 0` (~5 min on CPU).

## What was tested
- **1,000 random trials**, seed 0. Each trial picks a clip, a random robot pose anywhere on the ground floor and
  one of the 5 noise profiles (quiet / fan / tv / vacuum / party). The clip goes through the sim's own ear code
  (`SimSensors.synthesize_stereo_audio`: distance + wall attenuation, ILD/ITD, ambient noise, AGC, int16).
  Every 0.5 s window goes through the brain's `StereoHearing.process_audio`.
- **Alert** = what `main.py` acts on: smoke score >= 0.6 (safety pre-emption), or a non-danger registry sound
  as the top label with confidence >= 0.5 in one window. The debounced `SoundEvents` start (2 windows; smoke 1)
  is reported next to it.
- **Sounds:**
  - 460 registry trials: the synthesized demo clips; the real Wikimedia recordings (doorbell, knock, smoke
    alarm); 40 real ESC-50 door knocks.
  - 482 trials of the other 49 ESC-50 classes (baby crying, alarm clock, glass breaking, vacuum, washing
    machine, dog, footsteps, siren, ...), at a random spot in a random room, 0.5-2x loudness.
  - 58 trials of background noise only.
- **Dataset:** ESC-50 (K. J. Piczak 2015, github.com/karolpiczak/ESC-50 @ `33c8ce9e`), 2,000 clips x 50
  classes, **CC BY-NC 3.0**. It is in gitignored `assets/datasets/` and is never committed.

## Headline numbers (demo configuration: synthesized clips, YAMNet + spectral rules)
| | result |
|---|---|
| (a) Registry sounds, demo clips: alerted | **96 %** (296/309), median 0.5 s |
| (a) incl. real recordings | 78 % (361/460) |
| (b) Background noise only: false alert | **0 %** (0/58) |
| (b) Other real-world sounds (49 ESC-50 classes): false alert | **36 %** (173/482); 18 % with the 2-window debounce |
| (b) of which a false **smoke** alert (= false evacuation) | 1.5 % (7/482) |
| (c) Smoke alarm missed, demo clip | **0 %** (0/51), pre-empts in 0.5 s every time |
| (c) Smoke alarm missed, real recording | 53 % (27/51) |
| (d) YAMNet top-1 on all 2,000 clean ESC-50 clips | **40 %** (top-3 71 %) across 50 classes |

Per sound (demo clips): doorbell 98 %, kettle 100 %, oven timer 100 %, smoke alarm 100 %. Knock 62 %: lost
under tv / vacuum / party noise (1/5, 0/5, 2/5).

## Diagnosis: the hand-tuned spectral rules cause almost all false alerts
Same 1,000 trials, three hearing configurations:

| configuration | demo clips alerted | real-world sounds false alert | smoke missed (demo / real) |
|---|---|---|---|
| YAMNet + spectral rules (today) | 96 % | 36 % | 0 % / 53 % |
| YAMNet only (`NUDGE_HEARING_FALLBACK=0`) | 10 % | **0.4 %** (2/482) | 45 % / 49 % |
| today + tonal smoke rule (`NUDGE_SMOKE_TONAL=1`) | 96 % | 37 % (179/482), **false smoke 5.8 % (28/482)** | 0 % / **4 %** (2/51) |

- **Spectral rules fired 172 of the 173 false alerts** (YAMNet 2). They score energy in fixed bands: the doorbell
  at 480-2050 Hz, the knock at 110-600 Hz, the kettle as a tone at 2.15-2.4 kHz. Birds, crickets, babies, cows
  and creaking doors put energy there too. The false alerts come out as doorbell 69, kettle 66, knock 50,
  oven timer 18, smoke 7.
- **YAMNet alone almost never false-alerts** (it knows the 521 AudioSet classes, so a baby is "Crying", not a
  doorbell). But it does not recognise our *synthesized* placeholder tones (doorbell 0 %, kettle 5 %, oven 0 %).
  The demo works only because of the spectral rules.
- **YAMNet does know the real sounds:** real Wikimedia doorbell 100 % in a quiet room (YAMNet only). Household
  classes on clean clips: keyboard typing 100 %, toilet flush 98 %, crying baby 82 %, alarm clock 78 % top-1
  (98 % top-3), glass breaking 82 % top-3. So adding a tier-2/3 sound (baby crying, alarm clock, glass breaking) is
  mostly a registry entry, not a new detector.
- **The 1-window mission trigger in `main.py` doubles the false alerts** compared with the 2-window `SoundEvents`
  debounce (36 % -> 18 %), and costs only 1 % of demo-clip detections (96 % -> 95 %).

## Worst failures
1. **Real-world sounds trigger doorbell / kettle / knock missions** (36 % of trials). The worst classes: hen 9/12,
   crying baby 8/14, door creaks 8/14, snoring 8/14, rooster 8/16, chirping birds 7/10, cow 7/13, thunderstorm 7/14
   (knock).
2. **False smoke alerts** (7/482 = false evacuations), all from the spectral 3.1 kHz rule: chirping birds,
   crickets x2, sneezing, pig, crying baby, can opening.
3. **The real smoke-alarm recording is missed 53 %** in the default configuration: it beeps at 2.8 kHz, outside
   the synthetic 3.15 kHz band. While missed it also reads as "kettle" in 34 trials.
5. **The opt-in tonal smoke rule (`NUDGE_SMOKE_TONAL=1`) trades misses for false evacuations.** It catches the real
   alarm 49/51 (missed 27 -> 2), but false smoke alerts on real-world sounds go from 7 to **28/482 (5.8 %)**:
   car horn 4, crickets 4, chirping birds 4, rooster 2, crow 2, crying baby 2, plus 10 single ones (siren,
   washing machine, glass breaking, church bells, ...). Mostly quiet / fan noise. Ricktho's sweep found 0 false
   windows because it only played noise and registry sounds. **Input for the team decision: keep it opt-in for
   the demo.** If it is turned on, gate it on YAMNet not naming an animal / vehicle / bell.
4. **Knock is weak**: synthesized 62 %, real 32 %, ESC-50 knocks 24 %, and lost under tv / vacuum / party noise.

## Follow-ups (for TODO.md, Eden's file)
- Ricktho: real-world sounds false-alert in 36 % of trials via the spectral band rules (doorbell 69, kettle 66, knock 50,
  oven 18, smoke 7 of 482). Idea: only accept a spectral score when YAMNet's top class isn't something else (Animal,
  Speech, Crying, Vehicle, Bell, ...), or move to real recordings + YAMNet-led detection. Re-check with
  `pixi run hearing-eval`.
- Ricktho: 7/482 false smoke alerts (false evacuations) from the 3.1 kHz rule (birds, crickets, sneezing, pig, crying
  baby, can opening); 28/482 with `NUDGE_SMOKE_TONAL=1`.
- Ricktho: the knock is weak: demo clip 62 % (lost under tv / vacuum / party), real recordings 24-32 %.
- Team decision `NUDGE_SMOKE_TONAL`: real alarm missed 53 % without it, 4 % with it, but 4x the false smoke alerts.
- Finn (`main.py`): missions start on ONE window >= 0.5. The 2-window `SoundEvents` debounce halves real-world false
  alerts (36 % -> 18 %) for a 1 % detection cost. Consider it for non-danger sounds; smoke stays at 1 window.

## What this means for the pitch (honest version)
- Say: "In 1,000 randomised trials across the house and 5 noise profiles, Nudge caught the smoke alarm 51/51
  times within 0.5 s and the doorbell, kettle and oven timer 98-100 %. Background noise alone never raised an
  alert (0/58)."
- Say: "Its classifier (YAMNet, on-device) knows 521 sound classes. On a public benchmark of 2,000 everyday
  sounds (ESC-50) it names the class right 40 % first try, 71 % in its top 3. Baby crying 82 %, alarm clock 78 %."
- Don't claim "ignores everything else". With real-world sounds the band rules false-alert in about a third of
  trials. Frame it as the next step: real recordings + YAMNet-led detection (YAMNet alone: 0.4 % false alerts).

## Files
- `scripts/test_hearing_eval.py` (+ `pixi run hearing-eval`), `scripts/fetch_datasets.py` (+ `pixi run fetch-esc50`)
- `docs/hearing-eval.md`, `docs/img/hearing-eval.png` (generated)
- `.gitignore`: `assets/datasets/`, `assets/eval/`
- Read-only use of `hearing.py` / `sensors.py` (Ricktho's lane): **no changes**. The eval drives
  `SimSensors.synthesize_stereo_audio` without a MuJoCo model and patches `sim.sensors.SOUND_SOURCES` in its own
  process to place the extra sound.
