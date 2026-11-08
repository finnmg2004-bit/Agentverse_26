#!/usr/bin/env python3
"""
Soak test: run scenarios N rounds through scripts/locked_test.py (one locked run per round, so other
sessions on this PC can get the test lock in between) and print a pass-rate table.

    python scripts/soak.py doorbell smoke carpet_rule demo            # 3 rounds (default)
    python scripts/soak.py -n 5 --md soak.md --logs ~/nudge-overnight/soak doorbell

Each round's full output is saved to --logs (default ~/nudge-overnight/soak/<stamp>/round<k>.log).
For every failure the table section lists the lines that explain it (missing / forbidden / FAILED / ...).
Exit code 0 only if every run passed.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
SUMMARY_RE = re.compile(r"^\s+(PASS|FAIL)\s+(\S+)\s*$")
WHY_KEYS = ("missing", "forbidden", "SCENARIO FAILED", "did not report", "hotkeys never", "Traceback",
            "Error", "sim never connected", "FIRE PROXIMITY")


def split_by_scenario(text: str) -> Dict[str, List[str]]:
    """Output lines grouped by the `=== name:` header that precedes them."""
    out: Dict[str, List[str]] = {}
    cur = None
    for line in text.splitlines():
        m = re.match(r"^=== (\S+?):", line)
        if m:
            cur = m.group(1)
            out[cur] = []
        elif cur is not None:
            out[cur].append(line)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenarios", nargs="+")
    ap.add_argument("-n", "--rounds", type=int, default=3)
    ap.add_argument("--logs", default=None, help="directory for per-round logs")
    ap.add_argument("--md", default=None, help="also write the markdown table to this file")
    ap.add_argument("--wait", type=float, default=2400.0, help="max wait for the test lock per round (s)")
    args = ap.parse_args()

    stamp = time.strftime("%Y%m%d-%H%M")
    logdir = Path(args.logs).expanduser() if args.logs else Path.home() / "nudge-overnight" / "soak" / stamp
    logdir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, List[str]] = {s: [] for s in args.scenarios}  # "PASS" / "FAIL" / "?" per round
    why: Dict[str, List[str]] = {s: [] for s in args.scenarios}
    durations: Dict[str, List[float]] = {s: [] for s in args.scenarios}
    t_start = time.time()
    for k in range(1, args.rounds + 1):
        t0 = time.time()
        proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "locked_test.py"), "--wait", str(args.wait),
                               *args.scenarios], cwd=str(ROOT), capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
        text = (proc.stdout or "") + (proc.stderr or "")
        log = logdir / f"round{k}.log"
        log.write_text(text, encoding="utf-8")
        got = {m.group(2): m.group(1) for m in map(SUMMARY_RE.match, text.splitlines()) if m}
        per = split_by_scenario(text)
        for s in args.scenarios:
            r = got.get(s, "?")
            results[s].append(r)
            for line in per.get(s, []):
                m = re.match(rf"^--> {re.escape(s)}: \w+ \(([\d.]+)s\)", line)
                if m:
                    durations[s].append(float(m.group(1)))
            if r != "PASS":
                lines = [l.strip() for l in per.get(s, []) if any(key in l for key in WHY_KEYS)]
                why[s].append(f"round {k}: " + (" | ".join(lines[:4]) if lines else f"no summary (see {log})"))
        print(f"[soak] round {k}/{args.rounds} done in {time.time() - t0:.0f}s: "
              + ", ".join(f"{s} {results[s][-1]}" for s in args.scenarios), flush=True)

    rows = ["| scenario | pass rate | rounds | avg time |", "|---|---|---|---|"]
    for s in args.scenarios:
        n_pass = results[s].count("PASS")
        avg = f"{sum(durations[s]) / len(durations[s]):.0f} s" if durations[s] else "-"
        rows.append(f"| `{s}` | {n_pass}/{len(results[s])} | {' '.join(results[s])} | {avg} |")
    flaky = [s for s in args.scenarios if 0 < results[s].count("PASS") < len(results[s])]
    broken = [s for s in args.scenarios if results[s].count("PASS") == 0]
    md = [f"Soak {stamp}: {args.rounds} rounds in {(time.time() - t_start) / 60:.0f} min, logs `{logdir}`", ""] + rows
    if flaky or broken:
        md.append("")
        for s in flaky + broken:
            md.append(f"- `{s}` ({'flaky' if s in flaky else 'always fails'}):")
            md += [f"  - {w[:400]}" for w in why[s]]
    table = "\n".join(md)
    print("\n" + table)
    if args.md:
        Path(args.md).expanduser().write_text(table + "\n", encoding="utf-8")
    return 0 if not flaky and not broken else 1


if __name__ == "__main__":
    sys.exit(main())
