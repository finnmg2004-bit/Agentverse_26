#!/usr/bin/env python3
"""
Turn saved brain runs (runs/*.jsonl, written by nudge-brain/run_log.py) into eval cases.

  pixi run llm-harvest                       # all runs -> runs/candidates.jsonl + a table
  pixi run llm-harvest runs/2026-11-08_*.jsonl
  pixi run llm-harvest --add runs/<file>.jsonl:17                 # append that decision to golden.jsonl
  pixi run llm-harvest --add runs/<file>.jsonl:17 --expect C --note "owner holds the handle: keep leading"
  pixi run llm-harvest --add runs/<file>.jsonl:4                  # a `command` line -> teach_golden.jsonl
  pixi run llm-harvest --add runs/<file>.jsonl:4 --expect-json '{"type": "refuse", "say_contains": "can't do that"}'

Decision cases keep the EXACT menu the model saw (prompt text + options + schema flags), so the
golden set replays real situations, independent of how the world model / menu builder evolves.
Only decisions where the LLM is really asked are harvested (a mission and > 1 option).

Labels: by default the rules' choice (`rule.letter` in the log). When the logged decision was
wrong, pass the right letter(s) with --expect (comma list = several acceptable answers) and say
why in --note. In a fire, the expected `call_emergency_services` is what the situation text asks
for ("should be called"), like scripts/llm_eval/scenarios.escalation_expected.

Dedupe: two decisions are the same case when event + menu text match with walk metres, seconds
and counts masked and owner probabilities rounded to 10 points, so a run that repeats a situation
adds no new case.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN_DIR = ROOT / "runs"
GOLDEN = HERE / "golden.jsonl"
TEACH_GOLDEN = HERE / "teach_golden.jsonl"
CANDIDATES = RUN_DIR / "candidates.jsonl"


# --------------------------------------------------------------------------- reading
def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def iter_log(paths: Iterable[Path]) -> Iterable[Tuple[str, int, Dict[str, Any]]]:
    """(origin 'runs/<file>:<line>', line number, record) for every parsable line."""
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                yield f"{_rel(p)}:{i}", i, rec


def _rel(p: Path) -> str:
    try:
        return Path(p).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def is_llm_decision(rec: Dict[str, Any]) -> bool:
    return (rec.get("kind") == "decision" and bool(rec.get("prompt")) and rec.get("mission") is not None
            and len(rec.get("options") or []) > 1)


def dedupe_key(rec: Dict[str, Any]) -> str:
    if rec.get("kind") == "command":
        return "command|" + re.sub(r"\W+", " ", str(rec.get("text", "")).lower()).strip()
    # Probabilities decide searches ("prefer the closer room only when within ~10 points"), so they
    # stay, in 10-point buckets; metres, seconds and other counts are masked.
    text = re.sub(r"(\d+)%", lambda m: f"P{int(round(int(m.group(1)) / 10.0))}", rec.get("prompt") or "")
    masked = re.sub(r"\d+(\.\d+)?", "#", text)
    return f"{rec.get('event')}|{masked}"


def expected_esc(rec: Dict[str, Any]) -> Optional[bool]:
    if not rec.get("ask_esc"):
        return None
    return "should be called" in (rec.get("prompt") or "")


def _call(opt: Optional[Dict[str, Any]]) -> str:
    if not opt:
        return "-"
    return f"{opt['tool']}({', '.join(str(v) for v in (opt.get('args') or {}).values())})"


def option(rec: Dict[str, Any], letter: Optional[str]) -> Optional[Dict[str, Any]]:
    return next((o for o in rec.get("options") or [] if o.get("letter") == letter), None)


# --------------------------------------------------------------------------- cases
def decision_case(rec: Dict[str, Any], origin: str, expect: Optional[List[str]] = None,
                  note: str = "", case_id: str = "") -> Dict[str, Any]:
    rule = (rec.get("rule") or {}).get("letter")
    expect = expect or ([rule] if rule else [])
    if not expect:
        raise SystemExit(f"{origin}: the rules' choice is not on the menu; pass --expect")
    letters = [o["letter"] for o in rec["options"]]
    bad = [e for e in expect if e not in letters]
    if bad:
        raise SystemExit(f"{origin}: --expect {bad} not on the menu ({', '.join(letters)})")
    mission = rec.get("mission") or {}
    return {
        "id": case_id or f"{rec.get('event')}_{mission.get('sound')}_{Path(origin.split(':')[0]).stem}_{origin.split(':')[1]}",
        "origin": origin, "event": rec.get("event"), "mission": mission, "emergency": rec.get("emergency"),
        "prompt": rec["prompt"], "options": rec["options"], "ask_esc": bool(rec.get("ask_esc")),
        "ask_say": bool(rec.get("ask_say")),
        "expect": expect, "expect_call": " | ".join(_call(option(rec, e)) for e in expect),
        "expect_esc": expected_esc(rec), "rule": rule,
        "logged": {k: rec.get(k) for k in ("source", "choice", "latency", "model", "reason", "system_prompt")},
        "note": note,
    }


def teach_expect_from(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Expected outcome of a logged `command` line = what the brain did (override with --expect-json)."""
    res = rec.get("result")
    if not res:
        return {"type": "refuse", "say_contains": str(rec.get("say") or "")[:40]}
    if res.get("type") != "teach":
        return {"type": res.get("type")}
    task = res.get("task") or {}
    out: Dict[str, Any] = {"type": "teach", "trigger": {k: v for k, v in (task.get("trigger") or {}).items()
                                                         if k in ("kind", "at", "repeat", "sound", "minutes", "room")},
                           "skills": [s.get("skill") for s in task.get("steps") or []]}
    tgt = next((s["args"].get("target") for s in task.get("steps") or [] if s.get("skill") == "guide_owner_to"), None)
    if tgt:
        out["target"] = tgt
    return out


def candidates(paths: List[Path]) -> List[Dict[str, Any]]:
    late: Dict[Tuple[str, Any], Dict[str, Any]] = {}
    recs = list(iter_log(paths))
    for origin, _, rec in recs:
        if rec.get("kind") == "late_llm":
            late[(origin.split(":")[0], rec.get("req_id"))] = rec
    seen: Dict[str, Dict[str, Any]] = {}
    for origin, _, rec in recs:
        if not (is_llm_decision(rec) or (rec.get("kind") == "command" and rec.get("text"))):
            continue
        key = dedupe_key(rec)
        if key in seen:
            seen[key]["seen"] += 1
            continue
        cand = {"origin": origin, "kind": rec["kind"], "seen": 1, "key": key}
        if rec["kind"] == "decision":
            rule = (rec.get("rule") or {}).get("letter")
            lt = late.get((origin.split(":")[0], rec.get("req_id")))
            late_choice = None
            if lt:
                try:
                    late_choice = json.loads(lt.get("llm_raw") or "{}").get("choice")
                except Exception:
                    late_choice = None
            llm_choice = rec.get("choice") if rec.get("source") == "llm" else late_choice
            cand.update(event=rec.get("event"), mission=(rec.get("mission") or {}).get("sound"),
                        n_options=len(rec["options"]), rule=rule, rule_call=_call(option(rec, rule)),
                        source=rec.get("source"), choice=rec.get("choice"), llm_choice=llm_choice,
                        llm_call=_call(option(rec, llm_choice)), latency=rec.get("latency"),
                        disagree=bool(llm_choice and llm_choice != rule))
        else:
            cand.update(text=rec.get("text"), parser=rec.get("parser"), latency=rec.get("latency"),
                        expect=teach_expect_from(rec), say=rec.get("say"))
        seen[key] = cand
    return list(seen.values())


def print_table(cands: List[Dict[str, Any]]) -> None:
    dec = [c for c in cands if c["kind"] == "decision"]
    cmd = [c for c in cands if c["kind"] == "command"]
    print(f"{len(dec)} unique LLM decisions, {len(cmd)} unique owner sentences")
    for c in dec:
        flag = "  <-- LLM disagrees with the rules: review" if c["disagree"] else ""
        print(f"  {c['origin']:<48} {c['event']:<18} {str(c['mission']):<12} rules {c['rule']} {c['rule_call']:<40} "
              f"{c['source']:<9} llm {c['llm_choice'] or '-'} x{c['seen']}{flag}")
    for c in cmd:
        print(f"  {c['origin']:<48} \"{c['text']}\" -> {json.dumps(c['expect'])}")


# --------------------------------------------------------------------------- --add
def add(spec: str, expect: Optional[str], note: str, expect_json: Optional[str], case_id: str, force: bool) -> None:
    file, _, line = spec.rpartition(":")
    path = Path(file)
    if not path.is_absolute():
        path = (Path.cwd() / path) if (Path.cwd() / path).exists() else ROOT / path
    lines = path.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[int(line) - 1])
    origin = f"{_rel(path)}:{line}"
    if rec.get("kind") == "decision":
        if not is_llm_decision(rec):
            raise SystemExit(f"{origin}: not a decision the LLM is asked about (no mission or only one option)")
        existing = {dedupe_key(c): c["id"] for c in read_jsonl(GOLDEN)}
        if dedupe_key(rec) in existing and not force:
            raise SystemExit(f"{origin}: same situation already in golden.jsonl as {existing[dedupe_key(rec)]} "
                             "(--force to add anyway)")
        case = decision_case(rec, origin, [e.strip() for e in expect.split(",")] if expect else None, note, case_id)
        target = GOLDEN
    elif rec.get("kind") == "command":
        exp = json.loads(expect_json) if expect_json else teach_expect_from(rec)
        case = {"id": case_id or re.sub(r"\W+", "_", str(rec.get("text", "")).lower()).strip("_")[:40],
                "text": rec.get("text"), "expect": exp, "origin": origin, "note": note}
        if any(c.get("text") == case["text"] for c in read_jsonl(TEACH_GOLDEN)) and not force:
            raise SystemExit(f"{origin}: sentence already in teach_golden.jsonl (--force to add anyway)")
        target = TEACH_GOLDEN
    else:
        raise SystemExit(f"{origin}: kind {rec.get('kind')!r} can't become an eval case")
    with open(target, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"[added] {case['id']} -> {target.relative_to(ROOT).as_posix()} "
          f"(expect {case.get('expect_call') or json.dumps(case['expect'])})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="run logs (default: runs/*.jsonl)")
    ap.add_argument("--add", metavar="FILE:LINE", help="append this logged decision / sentence to the golden set")
    ap.add_argument("--expect", help="correct letter(s) for --add, comma list (default: the rules' letter)")
    ap.add_argument("--expect-json", help="expected outcome for a `command` line, JSON (default: what the brain did)")
    ap.add_argument("--note", default="", help="why this label (required when it differs from the rules)")
    ap.add_argument("--id", default="", help="case id for --add")
    ap.add_argument("--force", action="store_true", help="--add even if the same situation is already golden")
    ap.add_argument("--out", default=str(CANDIDATES), help="candidates file (default runs/candidates.jsonl)")
    args = ap.parse_args()
    if args.add:
        add(args.add, args.expect, args.note, args.expect_json, args.id, args.force)
        return
    paths = [Path(p) for f in (args.files or [str(RUN_DIR / "*.jsonl")]) for p in sorted(glob.glob(f))]
    paths = [p for p in paths if p.name != CANDIDATES.name]
    if not paths:
        sys.exit("no run logs found (runs/*.jsonl): run the brain or `pixi run test-scenarios` first")
    cands = candidates(paths)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cands), encoding="utf-8")
    print_table(cands)
    print(f"[saved] {_rel(out)} - add the ones worth keeping with --add <origin> (and --expect/--note if the rules are wrong)")


if __name__ == "__main__":
    main()
