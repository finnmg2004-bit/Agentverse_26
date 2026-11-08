#!/usr/bin/env python3
"""
The two golden sets of the LLM eval (run_eval.py --sets golden teach):

  golden.jsonl        REAL agent decisions harvested from saved runs (harvest_runs.py). Each case is
                      replayed with the exact menu text the model saw, through NudgeAgent._chat (same
                      system prompt, JSON schema, think: false, num_predict) and scored against the
                      reviewed answer letter(s) + call_emergency_services in a fire.
  teach_golden.jsonl  owner sentences -> taught task / refusal, through the real rules.CommandParser
                      (Gemma teach form merged with the keyword parser) and rules.validate.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import agent  # noqa: E402  (sys.path set up by scenarios.py, imported first by run_eval.py)
import safety  # noqa: E402

HERE = Path(__file__).resolve().parent
GOLDEN = HERE / "golden.jsonl"
TEACH_GOLDEN = HERE / "teach_golden.jsonl"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pct(values: List[float], q: float) -> float:
    v = sorted(values)
    k = (len(v) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def set_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    lat = [r["latency_s"] for r in rows if r.get("latency_s") is not None]
    good = sum(bool(r["correct"]) for r in rows)
    return {"n": n, "correct": good, "accuracy": round(good / n, 3) if n else None,
            "p50_s": round(pct(lat, 0.50), 3) if lat else None,
            "p95_s": round(pct(lat, 0.95), 3) if lat else None,
            "under_watchdog": round(sum(x <= safety.WATCHDOG_S for x in lat) / len(lat), 3) if lat else None}


# --------------------------------------------------------------------------- golden decisions
def check_golden() -> int:
    """Golden cases must be well-formed: unique ids, expected letters on the menu, a real choice."""
    bad, ids = 0, set()
    cases = read_jsonl(GOLDEN)
    for c in cases:
        letters = [o["letter"] for o in c["options"]]
        if c["id"] in ids or not c["expect"] or any(e not in letters for e in c["expect"]) or len(letters) < 2:
            bad += 1
            print(f"[golden BAD] {c['id']}: expect {c['expect']} vs menu {letters}")
        ids.add(c["id"])
    print(f"[check] golden.jsonl: {len(cases)} real decisions, {bad} malformed; "
          f"teach_golden.jsonl: {len(read_jsonl(TEACH_GOLDEN))} sentences")
    return bad


def _call(opt: Optional[Dict[str, Any]]) -> str:
    return f"{opt['tool']}({', '.join(str(v) for v in opt['args'].values())})" if opt else "-"


def run_golden(probe, timeout: float) -> List[Dict[str, Any]]:
    from run_log import text_hash
    sp = text_hash(agent.SYSTEM_PROMPT)
    rows = []
    for c in read_jsonl(GOLDEN):
        schema = agent._schema(len(c["options"]), c["ask_esc"], c["ask_say"])
        t0 = time.perf_counter()
        try:
            got: Any = probe._chat(c["prompt"], schema, timeout=timeout)
            lat: Optional[float] = round(time.perf_counter() - t0, 3)
        except Exception as exc:
            got, lat = {"error": str(exc)}, None
        got = got if isinstance(got, dict) else {}
        choice = got.get("choice")
        esc_ok = c.get("expect_esc") is None or got.get("call_emergency_services") is c["expect_esc"]
        say_ok = (not c["ask_say"]) or (isinstance(got.get("say"), str) and 0 < len(got["say"].strip()) <= 90)
        row = {"id": c["id"], "event": c["event"], "expected": c["expect"], "expect_call": c["expect_call"],
               "choice": choice, "call": _call(next((o for o in c["options"] if o["letter"] == choice), None)),
               "reason": got.get("reason"), "esc": got.get("call_emergency_services"),
               "esc_expected": c.get("expect_esc"), "say": got.get("say"), "say_ok": say_ok,
               "choice_ok": choice in c["expect"], "esc_ok": esc_ok, "latency_s": lat,
               "error": got.get("error"),
               "prompt_changed_since_logged": (c.get("logged") or {}).get("system_prompt") not in (None, sp)}
        row["correct"] = row["choice_ok"] and esc_ok
        print(f"  {'ok ' if row['correct'] else 'BAD'} {'  err' if lat is None else f'{lat:5.2f}'}s  "
              f"{c['id']:<46} want {'/'.join(c['expect'])} {c['expect_call']:<38} got {choice} {row['call']}"
              + ("" if esc_ok else f"  [esc want {c['expect_esc']} got {got.get('call_emergency_services')}]"))
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- teach sentences
def _norm(t: Any) -> str:
    return str(t or "").lower().replace("’", "'")


def teach_ok(expect: Dict[str, Any], instr: Optional[Dict[str, Any]], say: str) -> bool:
    if expect["type"] == "refuse":
        return instr is None and _norm(expect.get("say_contains")) in _norm(say)
    if instr is None or instr.get("type") != expect["type"]:
        return False
    if expect["type"] != "teach":
        return True
    task = instr.get("task") or {}
    trig = task.get("trigger") or {}
    if any(trig.get(k) != v for k, v in (expect.get("trigger") or {}).items()):
        return False
    steps = task.get("steps") or []
    if "skills" in expect and [st.get("skill") for st in steps] != expect["skills"]:
        return False
    if "target" in expect and not any(st.get("skill") == "guide_owner_to"
                                      and (st.get("args") or {}).get("target") == expect["target"] for st in steps):
        return False
    if "text_contains" in expect and not any(_norm(expect["text_contains"]) in _norm((st.get("args") or {}).get("text"))
                                             for st in steps):
        return False
    return True


def run_teach(probe) -> List[Dict[str, Any]]:
    import rules
    parser = rules.CommandParser(probe)
    rows = []
    for c in read_jsonl(TEACH_GOLDEN):
        parser._work({"text": c["text"], "source": "gui", "confidence": 1.0})  # synchronous here
        res = parser.poll()[0]
        instr, say = rules.validate(res["parsed"], False, c["text"])
        ok = teach_ok(c["expect"], instr, say)
        task = (instr or {}).get("task") or {}
        got = (f"{task.get('trigger')} -> {[st['skill'] for st in task.get('steps') or []]}"
               + "".join(f" {st['args'].get('target')}" for st in task.get("steps") or []
                         if st["skill"] == "guide_owner_to")
               if task else ((instr or {}).get("type") or f"refused: {say}"))
        llm_used = res["source"] != "keywords"
        rows.append({"id": c["id"], "text": c["text"], "expect": c["expect"], "parser": res["source"],
                     "llm_raw": res.get("raw"), "instr": instr, "say": say, "correct": ok,
                     "latency_s": res["latency"] if llm_used else None})
        print(f"  {'ok ' if ok else 'BAD'} {res['latency']:5.2f}s  {c['id']:<22} [{res['source']}] "
              f"\"{c['text']}\" -> {got}")
    return rows
