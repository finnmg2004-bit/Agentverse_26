# 5-minute pitch: outline (draft, Eden)

Judging weights to hit: perception-driven action 25 %, adapting to scene changes 25 %, relevance & impact
20 % (critical failure handled on screen), operational autonomy / 100 % local 15 %, ambition & engineering 15 %.

| Time | Slide / moment | Say (key points) | Criterion |
|---|---|---|---|
| 0:00–0:30 | **The person** | A deafblind person can't hear the doorbell, the kettle or the smoke alarm, and can't see where to go. Guide dogs can't hear for you; hearing dogs can't guide. Nudge does both. | Relevance |
| 0:30–1:30 | **Demo video (≤ 60 s)** | Let it play. One line before: "Everything you'll see runs on this laptop." | Perception, adapting |
| 1:30–2:15 | **How it decides** (diagram: ears + camera + LiDAR → owner-location belief → reflex layer / Gemma thinking layer → robot) | Two layers: plain-code reflexes in milliseconds (danger first, obstacles, speed cap) that the LLM can never override; local Gemma for judgement, choosing one option from a menu built from live perception, with a reason on screen. | Engineering, perception |
| 2:15–2:50 | **The safety moment** (screenshot: "LLM choice rejected by safety… terrace passes 1.6 m from the fire") | In rehearsal Gemma tried to lead the owner past the fire; the safety layer rejected it and the rules took the safe exit. That's the design: the LLM proposes, plain code disposes. | Critical failure handled |
| 2:50–3:30 | **Teaching it** (carpet rule clip, voice) | The owner says "I've got a new carpet, don't step on it"; local Whisper + Gemma turn it into a rule the planner enforces. Danger sounds can't be overruled by any rule. | Adapting, relevance |
| 3:30–4:05 | **Accessibility by design** | One event, every channel: vibration code by priority tier, light, screen, speech. Short, plain messages because for many BSL users English is a second language. Voice is one input, never the only one. | Relevance & impact |
| 4:05–4:35 | **100 % local, measured** | YAMNet, YOLOv8n, Gemma 4 E2B, Whisper, Piper all on-device. Gemma picks the right option 97.8 % of the time, in ~0.5 s; 9/9 end-to-end scenario tests pass; sim at real time. | Autonomy, engineering |
| 4:35–5:00 | **What's next + ask** | Real robot (SLAM instead of sim pose), learned gait, real household recordings, teachable new jobs ("every night at 10, take me to bed"). | Ambition |

## Q&A prep (likely questions)
- **"What if the LLM is wrong?"** It only chooses from options the safety layer already allows, the answer is schema-constrained, every choice is re-validated, and a 4 s watchdog hands over to rules. We show a real rejection.
- **"Why not a cloud model?"** Privacy in the home, no internet dependency in an emergency, and latency; it's also a judging requirement.
- **"Is the perception real or scripted?"** Sounds are classified from synthesized stereo audio by YAMNet + spectral rules with a bearing; the owner is found by YOLO on the rendered head camera; the robot's pose comes from the sim (honest: a real robot needs SLAM).
- **"How would a deafblind person actually use it?"** Handle vibration codes (felt), nudges, guiding by the handle; the screen/voice help deaf or blind owners respectively.
- **"What happens if the owner doesn't respond?"** It nudges again (up to 3), and in a fire leads them out anyway and can call emergency services.

## Assets to prepare
- [ ] Demo video (see `docs/DEMO_STORYBOARD.md`)
- [ ] Architecture diagram (one slide)
- [ ] Screenshot: safety rejection in the GUI event log (from the rehearsal)
- [ ] Short carpet-rule clip (voice → "Nudge says" → shaded no-go area)
- [ ] Numbers slide (from `docs/TECHNICAL_SUMMARY.md`)

## Pitch note: "Teaching it" slide, variant B (experimental)
Replace the carpet clip (2:50-3:30) or add 20 s: the owner says "every night at 10, take me to bed"; Gemma turns it into a task program (trigger + steps from existing skills only), the Tasks tab shows it, the clock jumps to 21:59 and Nudge acts. Say: "New jobs are data, not code; the validator only allows known skills, and a smoke alarm beats every task." Unsupported requests get an honest "Sorry, I can't do that yet."
