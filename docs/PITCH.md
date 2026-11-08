# 5-minute pitch: outline (Eden, draft 12:15 on 2026-11-08; [TBC] = confirm after the 12:45 freeze)

Judging weights to hit: perception-driven action 25 %, adapting to scene changes 25 %, relevance & impact
20 % (critical failure handled on screen), operational autonomy / 100 % local 15 %, ambition & engineering 15 %.
**One line to repeat:** *rules for safety, local Gemma for the unknown.*

| Time | Slide / moment | Say (key points) | Criterion |
|---|---|---|---|
| 0:00–0:30 | **The person** | A deafblind person can't hear the doorbell, the kettle or the smoke alarm, and can't see where to go. Guide dogs can't hear for you; hearing dogs can't guide. Nudge does both. | Relevance |
| 0:30–1:30 | **Demo video (≤ 60 s)** | "Everything you'll see runs on this laptop." Short sped-up clips, each a different situation (format decided after the freeze). | Perception, adapting |
| 1:30–2:10 | **How it decides** (diagram: ears + camera + LiDAR → where things are → reflex layer / Gemma thinking layer → robot) | Plain-code reflexes in milliseconds that the LLM can never override; local Gemma for judgement, choosing from a menu built from live perception, with a reason on screen; **0.19 s** per decision on this laptop. | Engineering, perception |
| 2:10–3:10 | **It adapts** (4 quick examples, 15 s each) | **Hears where you are:** the owner speaks from another room, Nudge goes straight there. **Is it real?** A doorbell on the TV vs the real door, told apart by direction ("That was the TV"). **Unknown sounds [TBC]:** glass breaking, a dog, a baby: Gemma reasons what to do. **A new home:** it explores a house it has never seen and builds its own map (99.8 % of the floor, finds the TV and the sofa itself). | Adapting (25 %) |
| 3:10–3:45 | **When things go wrong** (screenshots: safety rejection + "999 CALLED") | In rehearsal Gemma tried to lead the owner past the fire; the safety layer rejected it and took the safe exit. If the owner doesn't respond, it alerts their emergency contact and stays; in a fire it asks "Shall I call 999?" while still leading them out. | Critical failure handled |
| 3:45–4:15 | **Built for the person** | Deaf / Blind / Deafblind profiles choose the outputs (vibration code by priority, light, screen, speech; speech only when near). Teach it in one sentence ("every night at 10, take me to bed"); it refuses what it can't do. Short, plain messages: for many BSL users English is a second language. | Relevance & impact |
| 4:15–4:40 | **100 % local, measured** | YAMNet, YOLOv8n, Gemma 4 E2B, Whisper, Piper, all on-device. Hearing: 96 % of household sounds alerted in 1,000 random trials, **smoke alarm never missed**, 0 % false alerts from noise. Owner seen in 97–98 % of camera views. [TBC] full test suite + adaptability benchmark pass rate. | Autonomy, engineering |
| 4:40–5:00 | **What's next** | A real Go2: [TBC: LiDAR localisation already estimates the pose / else: SLAM instead of the sim pose], a learned gait, more homes, more sounds. | Ambition |

## Q&A prep (likely questions)
- **"What if the LLM is wrong?"** It only chooses from options the safety layer already allows; the answer is schema-constrained; every choice is re-validated; a 4 s watchdog hands over to rules. We show a real rejection.
- **"Why not a cloud model?"** Privacy in the home, no internet dependency in an emergency, latency; it's also a judging criterion.
- **"Is the perception real or scripted?"** Sounds are classified from simulated stereo audio by YAMNet + rules, with direction from the two ears; the owner is found by YOLO on the rendered head camera; we measured both (1,000 random hearing trials; 97–98 % owner detection). Honest: [TBC] the pose comes from the sim (a real robot needs localisation).
- **"How would a deafblind person actually use it?"** Handle vibration codes (felt), nudges, guiding by the handle; screen / light for deaf owners, speech for blind owners; the profile chooses.
- **"What if it hears something it doesn't know?"** [TBC] Gemma reasons about it (what, where, owner's state) and picks a safe action; anything danger-like is always top priority.
- **"Does it work in a different house?"** A second home (apartment) runs with the same code; the robot can map a new house by itself.
- **"What if the owner doesn't respond?"** After unanswered nudges: emergency contact + stays; in a fire: 999 + leads them out anyway; a fallen owner → "Are you OK?" → 999.

## Assets to prepare
- [ ] Demo video (montage tooling: `scripts/record_clips.py`, `scripts/make_montage.py`, `docs/VIDEO_HOWTO.md` [TBC])
- [ ] Architecture diagram (one slide)
- [ ] Screenshots: safety rejection; "999 CALLED (no answer)" banner (`ask_45.png`); night + dock; Deaf/Blind switch
- [ ] Exploration clip (robot maps the house, sped up) — Teo
- [ ] Numbers slide (from `docs/TECHNICAL_SUMMARY.md`); hearing chart `docs/img/hearing-eval.png` (Teo)
