#!/usr/bin/env python3
"""
LLM eval harness for Nudge's menu-based decision layer (`nudge-brain/agent.py`).

For every case in `eval_cases.py` it builds the real world model, the real menu (`build_menu` +
`_menu_text`) and the real JSON schema (`_schema`), and sends it through the agent's own
`NudgeAgent._chat()` — the same system prompt, `format` schema, `think: False`, temperature 0,
num_predict (48, or 96 with `say`) and NUDGE_LLM_GPU_LAYERS handling as at runtime. Scores:

  accuracy    chosen letter == the labelled option (deterministic_policy's call, except three
              documented rules quirks, see scenarios.sensible_label), and, in a fire,
              call_emergency_services == what the situation text asks for
  choice_acc  letter only;  esc_acc  escalation flag only (fire cases)
  valid       the chosen option passes safety.validate_action (else the agent falls back)
  say_ok      when asked, a non-empty owner message <= 90 chars (content is not scored)
  latency     wall time per request, p50 / p95 / max, % under the watchdog and under 1.0 s

The adoption gate (AGENTS.md §6.1) uses the cases where the LLM is really asked (>1 option).

Three sets (see README.md):
  synthetic  eval_cases.py: 56 hand-checked decision points rendered LIVE through today's
             build_menu (catches menu-builder changes)
  golden     golden.jsonl: REAL decisions from saved runs (runs/*.jsonl, harvest_runs.py), replayed
             with the exact menu the model saw (catches prompt / model changes on real situations)
  teach      teach_golden.jsonl: owner sentences -> taught task / "can't do that", through the real
             rules.CommandParser (Gemma teach form + keyword merge) and rules.validate

  pixi run llm-eval                               # gemma4:e2b, all sets
  pixi run llm-eval --gate                        # exit 1 if any set < 90 % (run after ANY change)
  pixi run llm-eval --sets golden teach           # just these sets
  pixi run llm-eval --models gemma4:e2b nudge-gemma
  pixi run llm-eval --check                       # only verify cases (no Ollama)

Results: scripts/llm_eval/results/<model>_<date>[_tag].json (per-case answers + summary).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import scenarios  # noqa: E402  (sets up sys.path for nudge-brain + sim)
from eval_cases import CASES  # noqa: E402

import agent  # noqa: E402
import safety  # noqa: E402

import golden_sets  # noqa: E402

RESULTS_DIR = HERE / "results"
SETS = ("synthetic", "golden", "teach")


class Probe(agent.NudgeAgent):
    """NudgeAgent without the probe thread; `_chat` is the runtime code path, the raw HTTP response
    is kept for token counts."""

    def __init__(self, model: str) -> None:  # noqa: D401  (no super().__init__: no background thread)
        self.model = model
        self.enabled = True       # rules.CommandParser checks these before asking the LLM
        self.llm_online = bool(model)
        self.last_resp: Dict[str, Any] = {}

    def _http(self, path, payload, timeout):
        self.last_resp = super()._http(path, payload, timeout)
        return self.last_resp


def check_cases() -> int:
    bad = 0
    ids = set()
    for sc in CASES:
        assert sc["id"] not in ids, f"duplicate id {sc['id']}"
        ids.add(sc["id"])
        r = scenarios.render(sc)
        if r["label"] != sc["expect"]:
            bad += 1
            print(f"[MISMATCH] {sc['id']}: expected {scenarios.call_str(sc['expect'])}, "
                  f"label -> {scenarios.call_str(r['label'])} (rules {scenarios.call_str(r['rule'])})")
        if r["override"]:
            print(f"[override] {sc['id']}: rules {scenarios.call_str(r['rule'])} -> label "
                  f"{scenarios.call_str(r['label'])} ({r['override']})")
        if r["choice"] is None:
            bad += 1
            print(f"[MISMATCH] {sc['id']}: the label's call is not on the menu")
        if r["escalate"] != sc.get("esc"):
            bad += 1
            print(f"[MISMATCH] {sc['id']}: esc expected {sc.get('esc')}, situation says {r['escalate']}")
    n_llm = sum(scenarios.render(sc)["llm_asked"] for sc in CASES)
    events = sorted({sc["event"] for sc in CASES})
    print(f"[check] {len(CASES)} cases ({n_llm} where the LLM is asked), events: {', '.join(events)}; "
          f"{bad} mismatches")
    return bad


def pct(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    v = sorted(values)
    k = (len(v) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def gpu_name() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip().splitlines()[0]
    except Exception:
        return "unknown"


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    if not n:
        return {"n": 0}
    lat = [r["latency_s"] for r in rows if r["latency_s"] is not None]
    esc_rows = [r for r in rows if r["esc_expected"] is not None]
    return {
        "n": n,
        "accuracy": round(sum(r["correct"] for r in rows) / n, 3),
        "choice_acc": round(sum(r["choice_ok"] for r in rows) / n, 3),
        "esc_acc": round(sum(r["esc_ok"] for r in esc_rows) / len(esc_rows), 3) if esc_rows else None,
        "valid_rate": round(sum(r["valid"] for r in rows) / n, 3),
        "say_ok_rate": round(sum(r["say_ok"] for r in rows) / n, 3),
        "p50_s": round(pct(lat, 0.50), 3), "p95_s": round(pct(lat, 0.95), 3),
        "max_s": round(max(lat), 3) if lat else None,
        "watchdog_s": safety.WATCHDOG_S,
        "under_watchdog": round(sum(x <= safety.WATCHDOG_S for x in lat) / n, 3),
        "under_2s": round(sum(x <= 2.0 for x in lat) / n, 3),
        "under_1s": round(sum(x <= 1.0 for x in lat) / n, 3),
        # what the agent would actually use: correct, valid and inside the watchdog (NUDGE_WATCHDOG_S)
        "llm_decides_correctly": round(sum(r["correct"] and r["valid"] and (r["latency_s"] or 99) <= safety.WATCHDOG_S
                                           for r in rows) / n, 3),
    }


def unload_others(model: str) -> None:
    """Free the GPU for a fair timing: unload every other model Ollama has resident."""
    try:
        with urllib.request.urlopen(agent.OLLAMA_URL + "/api/ps", timeout=5) as resp:
            running = [m["name"] for m in json.loads(resp.read().decode()).get("models", [])]
    except Exception:
        return
    for name in running:
        if name != model and name != f"{model}:latest":
            body = json.dumps({"model": name, "keep_alive": 0}).encode()
            req = urllib.request.Request(agent.OLLAMA_URL + "/api/generate", data=body,
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=60).read()
            print(f"[unload] {name}")


def run_model(model: str, timeout: float, sets=SETS) -> Dict[str, Any]:
    print(f"\n=== {model} @ {agent.OLLAMA_URL} (NUDGE_LLM_GPU_LAYERS={os.environ.get('NUDGE_LLM_GPU_LAYERS')}) ===")
    unload_others(model)
    probe = Probe(model)
    t0 = time.perf_counter()  # warm-up exactly like NudgeAgent._probe_loop; not timed
    probe._chat("Situation:\n- warm-up\n\nOptions:\nA) get_state() - nothing to do", agent._schema(1, False),
                timeout=300)
    print(f"[warmup] {time.perf_counter() - t0:.1f} s")

    out: Dict[str, Any] = {"model": model, "date": dt.datetime.now().isoformat(timespec="seconds"),
                           "host": {"gpu": gpu_name(), "platform": platform.platform(),
                                    "gpu_layers": os.environ.get("NUDGE_LLM_GPU_LAYERS")},
                           "sets": {}}
    if "golden" in sets:
        print(f"\n--- golden: {golden_sets.GOLDEN.name} (real logged decisions, exact menus) ---")
        g = golden_sets.run_golden(probe, timeout)
        out["sets"]["golden"] = {"summary": golden_sets.set_summary(g), "cases": g}
    if "teach" in sets:
        print(f"\n--- teach: {golden_sets.TEACH_GOLDEN.name} (owner sentence -> task / can't do that) ---")
        t = golden_sets.run_teach(probe)
        out["sets"]["teach"] = {"summary": golden_sets.set_summary(t), "cases": t}
    if "synthetic" not in sets:
        return out
    print("\n--- synthetic: eval_cases.py (rendered through today's build_menu) ---")
    rows = []
    for sc in CASES:
        r = scenarios.render(sc)
        row: Dict[str, Any] = {"id": sc["id"], "event": sc["event"], "mission": sc.get("mission"),
                               "llm_asked": r["llm_asked"], "tags": sc.get("tags", []),
                               "n_options": len(r["options"]), "expected": r["choice"],
                               "expected_call": scenarios.call_str(r["label"]), "override": r["override"],
                               "rule_call": scenarios.call_str(r["rule"]), "esc_expected": r["escalate"]}
        t0 = time.perf_counter()
        try:
            got = probe._chat(r["prompt"], r["schema"], timeout=timeout)
            row["latency_s"] = round(time.perf_counter() - t0, 3)
        except Exception as exc:
            got = {"error": str(exc)}
            row["latency_s"] = None if "timed out" in str(exc) else round(time.perf_counter() - t0, 3)
        resp = probe.last_resp or {}
        row["raw"] = (resp.get("message") or {}).get("content", got.get("error") if isinstance(got, dict) else "")
        row["prompt_tokens"], row["output_tokens"] = resp.get("prompt_eval_count"), resp.get("eval_count")
        probe.last_resp = {}
        choice = got.get("choice") if isinstance(got, dict) else None
        row["choice"], row["reason"] = choice, got.get("reason") if isinstance(got, dict) else None
        row["esc"] = got.get("call_emergency_services") if isinstance(got, dict) else None
        row["say"] = got.get("say") if isinstance(got, dict) else None
        row["say_ok"] = (not r["ask_say"]) or (isinstance(row["say"], str) and 0 < len(row["say"].strip()) <= 90)
        action = scenarios.option_action(r, choice) if isinstance(choice, str) else None
        row["call"] = scenarios.call_str(action)
        row["choice_ok"] = choice == r["choice"]
        row["esc_ok"] = r["escalate"] is None or row["esc"] is r["escalate"]
        row["correct"] = row["choice_ok"] and row["esc_ok"]
        row["valid"] = bool(action and scenarios.is_valid(action, sc)[0])
        mark = "ok " if row["correct"] else "BAD"
        lat = "  err" if row["latency_s"] is None else f"{row['latency_s']:5.2f}"
        esc = "" if r["escalate"] is None else f" esc {r['escalate']}/{row['esc']}"
        print(f"  {mark} {lat}s  {sc['id']:<24} want {r['choice']} {row['expected_call']:<38} "
              f"got {choice} {row['call']}{esc}{'' if r['llm_asked'] else '  (not asked at runtime)'}"
              f"{'' if row['say_ok'] else '  [say missing]'}")
        rows.append(row)

    gate = [r for r in rows if r["llm_asked"]]
    out.update({"summary": {"all": summarize(rows), "llm_asked": summarize(gate),
                            "fire": summarize([r for r in gate if r["mission"] == "smoke_alarm"]),
                            "doorbell_knock": summarize([r for r in gate if r["mission"] != "smoke_alarm"])},
                "cases": rows})
    out["sets"]["synthetic"] = {"summary": golden_sets.set_summary(gate)}
    return out


def print_table(results: List[Dict[str, Any]]) -> None:
    results = [r for r in results if "summary" in r]
    if not results:
        return
    print("\nSynthetic cases where the LLM is asked at runtime:")
    print(f"{'model':<22} {'n':>3} {'acc':>6} {'choice':>6} {'esc':>6} {'valid':>6} "
          f"{'p50':>6} {'p95':>6} {'max':>6} {'<wd':>5} {'<1s':>5} {'used':>5}")
    for r in results:
        s = r["summary"]["llm_asked"]
        esc = "-" if s["esc_acc"] is None else f"{s['esc_acc']:.0%}"
        print(f"{r['model']:<22} {s['n']:>3} {s['accuracy']:>6.1%} {s['choice_acc']:>6.1%} {esc:>6} "
              f"{s['valid_rate']:>6.1%} {s['p50_s']:>6.2f} {s['p95_s']:>6.2f} {s['max_s']:>6.2f} "
              f"{s['under_watchdog']:>5.0%} {s['under_1s']:>5.0%} {s['llm_decides_correctly']:>5.0%}")
    print("  acc = right letter (and right call_emergency_services in a fire);\n"
          f"  <wd = answered within the watchdog (NUDGE_WATCHDOG_S = {safety.WATCHDOG_S:.1f} s); "
          "used = acc AND valid AND within the watchdog")
    for r in results:
        f, d = r["summary"]["fire"], r["summary"]["doorbell_knock"]
        print(f"  {r['model']}: doorbell/knock acc {d['accuracy']:.1%} (n={d['n']}), "
              f"fire acc {f['accuracy']:.1%} (n={f['n']})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["gemma4:e2b"])
    ap.add_argument("--check", action="store_true", help="only verify cases against the rules")
    ap.add_argument("--timeout", type=float, default=30.0, help="per-request timeout (s) for measuring")
    ap.add_argument("--tag", default="", help="suffix for the results file name")
    ap.add_argument("--sets", nargs="+", choices=SETS, default=list(SETS), help="which eval sets to run")
    ap.add_argument("--gate", action="store_true",
                    help="exit 1 if any set scores below --min (run after ANY prompt / model / menu change)")
    ap.add_argument("--min", type=float, default=0.90, help="gate threshold per set (default 0.90 = 18/20)")
    args = ap.parse_args()

    bad = check_cases() if "synthetic" in args.sets else 0
    bad += golden_sets.check_golden()
    if args.check:
        sys.exit(1 if bad else 0)
    if bad:
        sys.exit("eval cases disagree with the rules / menus - fix them first")
    RESULTS_DIR.mkdir(exist_ok=True)
    results = []
    for model in args.models:
        res = run_model(model, args.timeout, args.sets)
        results.append(res)
        name = model.replace(":", "_").replace("/", "_")
        tag = args.tag or ("gate" if args.gate else "")
        out = RESULTS_DIR / f"{name}_{dt.datetime.now():%Y-%m-%d_%H%M}{('_' + tag) if tag else ''}.json"
        out.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
        print(f"[saved] {out.relative_to(HERE.parents[1])}")
    print_table(results)

    failed = False
    print(f"\nGate (each set >= {args.min:.0%}; latency = LLM answers only; watchdog {safety.WATCHDOG_S:.1f} s):")
    for res in results:
        for name in args.sets:
            s = (res["sets"].get(name) or {}).get("summary")
            if not s or not s["n"]:
                print(f"  {res['model']:<14} {name:<10} (no cases)")
                continue
            ok = s["accuracy"] >= args.min
            failed |= not ok
            lat = (f"p50 {s['p50_s']:.2f} s  p95 {s['p95_s']:.2f} s  within watchdog {s['under_watchdog']:.0%}"
                   if s["p50_s"] is not None else "no LLM calls")
            print(f"  {res['model']:<14} {name:<10} {s['correct']:>3}/{s['n']:<3} {s['accuracy']:6.1%}  {lat}  "
                  f"{'PASS' if ok else 'FAIL'}")
    if args.gate:
        print("GATE " + ("FAILED: do not ship this prompt / model / menu change" if failed else "PASSED"))
        sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
