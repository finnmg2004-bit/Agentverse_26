#!/usr/bin/env python3
"""
Pre-sleep check for the overnight run (overnight/CHECKLIST.md). Run on each PC, in each lane's checkout:

    pixi run python scripts/overnight_check.py --lane sean --setup      # first time: branch + guard + push + doorbell test
    pixi run python scripts/overnight_check.py --lane sean              # re-check only
    pixi run python scripts/overnight_check.py --uninstall-hook         # morning: remove the push guard

--setup = --branch (switch to / create overnight/<lane> from origin/main) + --install-hook (pre-push guard:
only overnight/* may be pushed) + --push (real push of the branch, proves pushing works with no prompt)
+ --scenario doorbell (one locked scenario run, ~2 min). Exit code 0 = no FAIL lines.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GUARD_SRC = REPO / "scripts" / "overnight_pre_push.sh"
LANES = ("sean", "ricktho", "integration", "eden-tasks", "eden-gui", "eden-mapping")
MODELS = ("yolov8n.onnx", "yamnet.tflite", "yamnet_class_map.csv", "whisper-base.en", "en_US-ljspeech-medium.onnx")
NO_PROMPT = dict(os.environ, GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="never")

results: list[tuple[str, str, str]] = []


def report(level: str, what: str, detail: str = "") -> None:
    results.append((level, what, detail))
    print(f"[{level:4}] {what}" + (f"  ({detail})" if detail else ""), flush=True)


def git(*args: str, timeout: float = 60.0, env=None) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(REPO), capture_output=True, text=True,
                          timeout=timeout, env=env or os.environ)


def hooks_dir() -> Path:
    common = git("rev-parse", "--git-common-dir").stdout.strip()
    p = Path(common)
    return (p if p.is_absolute() else REPO / p) / "hooks"


def install_hook() -> None:
    dst = hooks_dir() / "pre-push"
    if dst.exists() and "overnight guard" not in dst.read_text(errors="ignore"):
        report("FAIL", "push guard", f"{dst} exists and is not ours; not overwritten")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(GUARD_SRC.read_text().replace("\r\n", "\n").encode())  # LF, or sh chokes on \r
    try:
        dst.chmod(0o755)
    except OSError:
        pass
    report("ok", "push guard installed", str(dst))


def uninstall_hook() -> None:
    dst = hooks_dir() / "pre-push"
    if dst.exists() and "overnight guard" in dst.read_text(errors="ignore"):
        dst.unlink()
        report("ok", "push guard removed", str(dst))
    else:
        report("ok", "no push guard installed")


def switch_branch(lane: str) -> None:
    branch = f"overnight/{lane}"
    git("fetch", "-q", "origin", timeout=120, env=NO_PROMPT)
    cur = git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    if cur == branch:
        return
    if git("status", "--porcelain", "--untracked-files=no").stdout.strip():
        report("FAIL", f"switch to {branch}", "uncommitted changes to tracked files; commit or stash them first")
        return
    remote = git("rev-parse", "--verify", "-q", f"origin/{branch}").returncode == 0
    local = git("rev-parse", "--verify", "-q", branch).returncode == 0
    if local:
        r = git("switch", branch)
    elif remote:
        r = git("switch", "-c", branch, f"origin/{branch}")
    else:
        r = git("switch", "-c", branch, "origin/main")
    report("ok" if r.returncode == 0 else "FAIL", f"on branch {branch}", r.stderr.strip()[-200:])


def check_git(lane: str | None, do_push: bool) -> None:
    if git("rev-parse", "--git-dir").returncode != 0:
        report("FAIL", "git repo", "not a git checkout")
        return
    name, email = git("config", "user.name").stdout.strip(), git("config", "user.email").stdout.strip()
    report("ok" if name and email else "FAIL", "git identity", f"{name} <{email}>")
    f = git("fetch", "-q", "origin", timeout=120, env=NO_PROMPT)
    report("ok" if f.returncode == 0 else "FAIL", "git fetch without a prompt", f.stderr.strip()[-200:])
    has_plan = git("cat-file", "-e", "HEAD:overnight/README.md").returncode == 0
    report("ok" if has_plan else "FAIL", "checkout contains the overnight plan",
           "" if has_plan else "git pull origin main (or re-run with --branch)")
    behind = git("rev-list", "--count", "HEAD..origin/main").stdout.strip()
    report("ok" if behind in ("0", "") else "WARN", "up to date with origin/main", f"{behind} commits behind")
    cur = git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    if lane:
        want = f"overnight/{lane}"
        report("ok" if cur == want else "FAIL", f"on branch {want}", f"currently {cur}; use --branch")
    elif cur == "main":
        report("WARN", "on main", "overnight sessions must work on overnight/<lane>")
    dirty = git("status", "--porcelain", "--untracked-files=no").stdout.strip()
    report("ok" if not dirty else "WARN", "no uncommitted tracked changes", dirty.replace("\n", "; ")[:200])

    hook = hooks_dir() / "pre-push"
    guarded = hook.exists() and "overnight guard" in hook.read_text(errors="ignore")
    report("ok" if guarded else "FAIL", "push guard hook installed", "" if guarded else "use --install-hook")
    if guarded:
        r = git("push", "--dry-run", "origin", "HEAD:refs/heads/main", timeout=90, env=NO_PROMPT)
        blocked = r.returncode != 0 and "overnight guard" in (r.stderr + r.stdout)
        report("ok" if blocked else "FAIL", "push to main is blocked", (r.stderr or r.stdout).strip()[-200:])

    if lane:
        target = f"HEAD:refs/heads/overnight/{lane}"
        args = ["push", "-u", "origin", target] if do_push else ["push", "--dry-run", "origin", target]
        r = git(*args, timeout=120, env=NO_PROMPT)
        report("ok" if r.returncode == 0 else "FAIL",
               f"{'push' if do_push else 'dry-run push'} to overnight/{lane} without a prompt",
               (r.stderr or r.stdout).strip()[-200:])


def check_ollama() -> None:
    base = os.environ.get("NUDGE_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
    model = os.environ.get("NUDGE_LLM_MODEL", "gemma4:e2b")
    try:
        with urllib.request.urlopen(f"{base}/api/tags", timeout=5) as resp:
            names = [m.get("name", "") for m in json.load(resp).get("models", [])]
    except Exception as exc:  # noqa: BLE001
        report("FAIL", "Ollama running", f"{base}: {exc}; start Ollama")
        return
    have = any(n == model or n.startswith(model) for n in names)
    report("ok" if have else "FAIL", f"Ollama has {model}", "" if have else f"ollama pull {model}")
    if not have:
        return
    body = json.dumps({"model": model, "prompt": "Reply with OK.", "stream": False, "think": False,
                       "keep_alive": "8h", "options": {"num_predict": 4}}).encode()
    times = []
    for _ in range(2):  # first call may load the model
        t0 = time.time()
        try:
            req = urllib.request.Request(f"{base}/api/generate", body, {"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=120).read()
        except Exception as exc:  # noqa: BLE001
            report("FAIL", "Gemma answers", str(exc))
            return
        times.append(time.time() - t0)
    report("ok" if times[-1] < 2.0 else "WARN", "Gemma answers (kept loaded 8 h)",
           f"warm {times[-1]:.2f} s, cold {times[0]:.2f} s")


def check_files() -> None:
    mdir = REPO / "assets" / "models"
    missing = [m for m in MODELS if not (mdir / m).exists()]
    report("ok" if not missing else "FAIL", "local models present",
           "" if not missing else f"missing {missing}: pixi run fetch-models")
    xml = REPO / "worlds" / "nerv-world" / "build" / "house-go2.xml"
    report("ok" if xml.exists() else "FAIL", "nerv-world house present", "" if xml.exists() else "pixi run fetch-world")
    free_gb = shutil.disk_usage(str(REPO)).free / 1e9
    report("ok" if free_gb > 5 else "WARN", "free disk space", f"{free_gb:.0f} GB")


def check_ports() -> None:
    busy = []
    for port in range(5555, 5560):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            busy.append(port)
        finally:
            s.close()
    report("ok" if not busy else "FAIL", "ports 5555-5559 free",
           "" if not busy else f"busy: {busy}; close sim/brain/GUI/voice before sleeping")


def _powercfg_ac(sub: str, setting: str) -> int | None:
    try:
        out = subprocess.run(["powercfg", "/query", "SCHEME_CURRENT", sub, setting],
                             capture_output=True, text=True, timeout=20).stdout
    except Exception:  # noqa: BLE001
        return None
    m = re.search(r"AC Power Setting Index:\s*0x([0-9a-fA-F]+)", out)
    return int(m.group(1), 16) if m else None


def check_power() -> None:
    if os.name != "nt":
        report("WARN", "power settings", "not Windows: check sleep/suspend by hand")
        return

    class SPS(ctypes.Structure):
        _fields_ = [("ACLineStatus", ctypes.c_byte), ("BatteryFlag", ctypes.c_byte),
                    ("BatteryLifePercent", ctypes.c_byte), ("SystemStatusFlag", ctypes.c_byte),
                    ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]
    sps = SPS()
    if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(sps)):
        report("ok" if sps.ACLineStatus == 1 else "FAIL", "plugged in", "" if sps.ACLineStatus == 1 else "on battery")
    for sub, setting, label, fix in (
            ("SUB_SLEEP", "STANDBYIDLE", "never sleeps on AC", "powercfg /change standby-timeout-ac 0"),
            ("SUB_SLEEP", "HIBERNATEIDLE", "never hibernates on AC", "powercfg /change hibernate-timeout-ac 0"),
            ("SUB_BUTTONS", "LIDACTION", "closing the lid does nothing on AC",
             "Control Panel > Power Options > Choose what closing the lid does > Do nothing (plugged in)")):
        v = _powercfg_ac(sub, setting)
        if v is None:
            report("WARN", label, f"couldn't read; check by hand ({fix})")
        else:
            report("ok" if v == 0 else "FAIL", label, "" if v == 0 else f"fix: {fix}")
    report("WARN", "Windows Update paused", "check by hand: Settings > Windows Update > Pause updates")


def check_claude() -> None:
    exe = shutil.which("claude")
    report("ok" if exe else "WARN", "claude CLI on PATH", exe or "only needed for the headless watchdog (Eden's PC)")


def run_scenario(name: str) -> None:
    print(f"[ .. ] running scenario '{name}' under the test lock (a few minutes)...", flush=True)
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "locked_test.py"), name], cwd=str(REPO))
    report("ok" if r.returncode == 0 else "FAIL", f"scenario {name}", f"exit {r.returncode}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lane", choices=LANES)
    ap.add_argument("--setup", action="store_true", help="--branch --install-hook --push --scenario doorbell")
    ap.add_argument("--branch", action="store_true")
    ap.add_argument("--install-hook", action="store_true")
    ap.add_argument("--uninstall-hook", action="store_true")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--scenario", default=None)
    a = ap.parse_args()
    if a.uninstall_hook:
        uninstall_hook()
        return 0
    if a.setup:
        a.branch = a.install_hook = a.push = True
        a.scenario = a.scenario or "doorbell"
    if (a.branch or a.push) and not a.lane:
        ap.error("--lane is needed for --branch/--push/--setup")

    print(f"Nudge overnight check  {time.strftime('%Y-%m-%d %H:%M')}  lane={a.lane}  repo={REPO}\n")
    if a.branch:
        switch_branch(a.lane)
    if a.install_hook:
        install_hook()
    check_git(a.lane, a.push)
    check_files()
    check_ports()
    check_ollama()
    check_power()
    check_claude()
    if a.scenario:
        run_scenario(a.scenario)
    (Path.home() / "nudge-overnight").mkdir(exist_ok=True)

    fails = [r for r in results if r[0] == "FAIL"]
    warns = [r for r in results if r[0] == "WARN"]
    print(f"\n{len(fails)} FAIL, {len(warns)} WARN. " + ("Fix the FAIL lines, then re-run." if fails else
          "Ready: start the session (overnight/CHECKLIST.md step 4)."))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
