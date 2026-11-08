# Agent evals: save every run, golden sets, gate

The agent (`nudge-brain/agent.py`) shows Gemma a lettered menu of legal options per event and
Gemma picks one. This folder checks that a prompt, model or menu change does not make those
choices worse, using real decisions saved from earlier runs.

## 1. Every run is saved (`nudge-brain/run_log.py`)
The brain appends one JSON line per decision to `runs/<date>_<time>_ep<episode>.jsonl`
(gitignored, a new file per brain start and per `[R]` reset). Each line holds the event, the
exact menu text and options, the chosen letter and tool, the source
(`llm` / `rules` / `watchdog` / `only option`), latency, Gemma's raw reply and reason, what the
rules would have done (and its letter), the owner message, mission, episode and sim time.
`late_llm` lines keep answers that came after the watchdog had already decided. `command` lines
keep owner sentences: what the parser made of them and the result (rule, task or refusal).

Writing happens on a background thread from a bounded queue, so the 20 Hz loop never waits.
Errors switch the log off after printing one line. `NUDGE_RUN_LOG=0` turns it off and
`NUDGE_RUN_DIR=<dir>` writes somewhere else.

## 2. Three eval sets
| set | file | what it checks |
|---|---|---|
| `golden` | `golden.jsonl` | **real** decisions harvested from scenario runs, replayed with the exact menu the model saw, scored against the reviewed answer (+ `call_emergency_services` in a fire) |
| `teach` | `teach_golden.jsonl` | owner sentence → taught task / "can't do that", through the real `rules.CommandParser` (Gemma teach form + keyword merge) and `rules.validate` |
| `synthetic` | `eval_cases.py` | 56 hand-checked situations rendered live through today's `build_menu`, so menu-builder changes are caught too (45 where the LLM is asked) |

Golden labels are usually the rules' choice. When we disagree with the rules, the case has a `note`
explaining why, and `expect` may list several acceptable letters.

## 3. The gate: run it after ANY prompt / model / menu change
```bash
pixi run llm-eval --gate                     # all sets, gemma4:e2b; exit 1 if any set < 90 %
pixi run llm-eval --gate --sets golden teach # the two golden sets only (~1-2 min)
pixi run llm-eval --models gemma4:e2b nudge-gemma   # compare models
pixi run llm-eval --check                    # validate the case files, no Ollama
```
It prints accuracy and LLM latency (p50 / p95, share within the `NUDGE_WATCHDOG_S` watchdog) per
set. Per-case answers go to `results/<model>_<date>_<time>[_gate].json`. Latency depends on the GPU:
measure on the recording laptop before deciding anything about speed.

## 4. Continuous evals: every mistake becomes a test
1. Run the brain or the scenarios as usual. Runs are saved automatically.
2. `pixi run llm-harvest` lists the unique LLM decisions and owner sentences in `runs/*.jsonl`
   (writes `runs/candidates.jsonl`). It flags decisions where Gemma (or its late answer) disagreed
   with the rules.
3. When a logged decision was wrong, add it with the right answer:
   ```bash
   pixi run llm-harvest --add runs/2026-11-08_090047_ep0.jsonl:3 --expect A --note "owner seen 1 s ago: scan, don't leave"
   ```
   `--expect` is optional (default: the rules' letter). A `command` line goes to
   `teach_golden.jsonl`, and `--expect-json '{"type": "refuse", "say_contains": "can'\''t do that"}'`
   overrides its label. Situations already in the set (same event + menu with numbers masked) are refused.
4. Re-run `pixi run llm-eval --gate`. Fix the prompt or menu until it passes. Never relabel a case just to pass.

## Files
- `run_eval.py`: the eval runner and gate. `golden_sets.py`: golden and teach scoring.
- `scenarios.py` + `eval_cases.py`: the synthetic set (real `WorldModel`, planner and evacuation planner).
- `harvest_runs.py`: run logs → candidate cases, `--add`.
- `results/`: saved eval runs.
