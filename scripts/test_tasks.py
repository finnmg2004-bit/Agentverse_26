#!/usr/bin/env python3
"""Unit tests for nudge-brain/tasks.py (teachable tasks, overnight contract C1/C3).

No sim, no brain, no ports: a fake clock, fake SoundEvents and a fake mission adapter.
    pixi run python scripts/test_tasks.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "nudge-brain")]

import tasks as T  # noqa: E402

BEDTIME = {"name": "bedtime",
           "trigger": {"kind": "time", "at": "22:00", "repeat": "daily"},
           "steps": [{"skill": "find_owner", "args": {}},
                     {"skill": "say", "args": {"text": "It's 10 pm. Time for bed."}},
                     {"skill": "guide_owner_to", "args": {"target": "guest"}}],
           "until": None}

FAILS = []


def check(cond, what):
    print(("  ok   " if cond else "  FAIL ") + what)
    if not cond:
        FAILS.append(what)


class FakeAdapter:
    """Stands in for main.py's mission machinery."""

    def __init__(self):
        self.started, self.cancelled = [], []
        self.is_busy = self.is_emergency = False
        self.status = "running"

    def busy(self):
        return self.is_busy

    def emergency(self):
        return self.is_emergency

    def start_job(self, job, task):
        self.started.append(job)
        self.status = "running"

    def job_status(self, job):
        return self.status

    def cancel_job(self, job):
        self.cancelled.append(job)


def test_validator():
    print("validator")
    task, why = T.validate_task(BEDTIME)
    check(task is not None and why == "", "bedtime task is valid")
    check(task["steps"][2]["args"]["target"] == "guest", "guest kept as the bed")
    alias = dict(BEDTIME, steps=[{"skill": "guide_owner_to", "args": {"target": "bed"}}])
    check(T.validate_task(alias)[0]["steps"][0]["args"]["target"] == "guest", "'bed' -> guest")
    smoke = dict(BEDTIME, trigger={"kind": "sound", "sound": "smoke_alarm"})
    task, why = T.validate_task(smoke)
    check(task is None and "danger" in why, f"smoke-alarm trigger refused ({why})")
    check(T.validate_task(dict(BEDTIME, trigger={"kind": "sound", "sound": "kettle"}))[0] is not None,
          "kettle trigger allowed")
    check(T.validate_task(dict(BEDTIME, steps=[{"skill": "make_tea", "args": {}}]))[0] is None, "unknown skill refused")
    check(T.validate_task(dict(BEDTIME, steps=[{"skill": "say", "args": {"text": "hi"}}] * 9))[0] is None,
          "9 steps refused")
    check(T.validate_task(dict(BEDTIME, steps=[{"skill": "guide_owner_to", "args": {"target": "moon"}}]))[0] is None,
          "unknown place refused")
    check(T.validate_task(dict(BEDTIME, trigger={"kind": "time", "at": "25:00"}))[0] is None, "bad time refused")
    check(T.validate_task("take me to bed")[0] is None, "non-dict refused, no exception")
    check(T.validate_task(dict(BEDTIME, trigger={"kind": "idle", "minutes": 120}))[0] is not None, "idle 120 min ok")
    ev = dict(BEDTIME, trigger={"kind": "event", "event": "owner_entered_room", "room": "kitchen"})
    check(T.validate_task(ev)[0] is not None, "owner_entered_room kitchen ok")
    conf = T.confirmation(T.validate_task(BEDTIME)[0])
    check(conf == "OK. Every day at 10 pm I'll take you to bed.", f"confirmation: {conf}")


def test_compile():
    print("compile")
    jobs = T.compile_jobs(T.validate_task(BEDTIME)[0]["steps"])
    check(jobs == [{"kind": "mission", "say": "It's 10 pm. Time for bed.", "target": "guest"}], f"bedtime -> {jobs}")
    remind = T.validate_task(dict(BEDTIME, steps=[{"skill": "find_owner"}, {"skill": "alert_owner", "args": {"text": "Pills!"}},
                                                  {"skill": "wait", "args": {"s": 5}}, {"skill": "say", "args": {"text": "Bye"}}]))[0]
    jobs = T.compile_jobs(remind["steps"])
    check([j["kind"] for j in jobs] == ["find_alert", "wait", "say"], f"reminder -> {[j['kind'] for j in jobs]}")


def test_watcher():
    print("trigger watcher")
    task = dict(T.validate_task(BEDTIME)[0], state="idle")
    w = T.TriggerWatcher()
    check(w.update([task], clock="08:00", day=0) == [], "first clock reading never fires")
    check(w.update([task], clock="21:59", day=0) == [], "jump to 21:59: not yet")
    check(w.update([task], clock="22:00", day=0) == ["bedtime"], "22:00 fires")
    check(w.update([task], clock="22:01", day=0) == [], "fires once")
    check(w.next_run(task) == "day 1 22:00", f"next run tomorrow ({w.next_run(task)})")
    w2 = T.TriggerWatcher()
    w2.update([task], clock="08:00", day=0)
    check(w2.update([task], clock="23:00", day=0) == [], "big jump past the time (stale) does not fire")
    w3 = T.TriggerWatcher()
    w3.update([task], clock="21:50", day=0)
    check(w3.update([task], clock="22:10", day=0) == ["bedtime"], "crossing within grace fires")
    check(w3.update([task], clock="08:00", day=0) == [] and w3.update([task], clock="22:00", day=0) == ["bedtime"],
          "clock reset (R) then 22:00 fires again")
    w4 = T.TriggerWatcher()
    mid = dict(T.validate_task(dict(BEDTIME, trigger={"kind": "time", "at": "00:05"}))[0], state="idle")
    w4.update([mid], clock="23:58", day=0)
    check(w4.update([mid], clock="00:06", day=1) == [mid["name"]], "fires across midnight")

    kettle = dict(T.validate_task(dict(BEDTIME, name="tea", trigger={"kind": "sound", "sound": "kettle"}))[0], state="idle")
    ws = T.TriggerWatcher()
    check(ws.update([kettle], sound_events=[{"sound": "kettle", "phase": "stop"}]) == [], "sound stop event ignored")
    check(ws.update([kettle], sound_events=[{"sound": "doorbell", "phase": "start"}]) == [], "other sound ignored")
    check(ws.update([kettle], sound_events=[{"sound": "kettle", "phase": "start"}]) == ["tea"], "kettle start fires")

    idle = dict(T.validate_task(dict(BEDTIME, name="check", trigger={"kind": "idle", "minutes": 2}))[0], state="idle")
    wi = T.TriggerWatcher()
    fires = [wi.update([idle], owner_idle_s=s) for s in (30, 119, 121, 300, 10, 130)]
    check(fires == [[], [], ["check"], [], [], ["check"]], f"idle fires once per still stretch {fires}")

    ev = dict(T.validate_task(dict(BEDTIME, name="hello", trigger={"kind": "event", "event": "owner_entered_room",
                                                                   "room": "kitchen"}))[0], state="idle")
    check(T.TriggerWatcher().update([ev], events=[{"event": "owner_entered_room", "room": "kitchen"}]) == ["hello"],
          "owner_entered_room fires")


def test_store_and_runner(tmp):
    print("store + runner")
    path = os.path.join(tmp, "tasks.json")
    store = T.TaskStore(path)
    check(len(store) == 0, "empty store when file missing")
    store.add(T.validate_task(BEDTIME)[0])
    check(T.TaskStore(path).tasks.get("bedtime") is not None, "task persisted and reloaded")
    with open(path, "w", encoding="utf-8") as f:
        f.write("{not json")
    check(len(T.TaskStore(path)) == 0, "corrupt file -> empty store, no exception")
    store.save()

    logs = []
    ad = FakeAdapter()
    run = T.TaskRunner(store, ad, log=logs.append)
    run.tick(0.0, clock="21:59", day=0)
    run.tick(1.0, clock="22:00", day=0)
    check(len(ad.started) == 1 and ad.started[0]["kind"] == "mission" and ad.started[0]["target"] == "guest",
          "22:00 starts the bedtime mission")
    check(run.status()[0]["state"] == "running", "status says running")
    ad.status = "ok"
    run.tick(2.0, clock="22:00", day=0)
    st = run.status()[0]
    check(st["state"] == "idle" and st["last_run"]["result"] == "done", f"done -> idle again (daily) {st['last_run']}")
    for key in ("name", "trigger", "steps", "next_run", "source", "state", "last_run"):
        check(key in st, f"C3 field {key}")

    # busy: a built-in mission is running -> the task waits, then runs
    ad2 = FakeAdapter()
    ad2.is_busy = True
    run2 = T.TaskRunner(store, ad2, log=logs.append)
    run2.fire("bedtime")
    run2.tick(0.0)
    check(ad2.started == [] and run2.status()[0]["queued"], "waits while another mission is active")
    ad2.is_busy = False
    run2.tick(1.0)
    check(len(ad2.started) == 1, "starts once the mission is over")

    # smoke pre-empts a running task
    ad2.is_emergency = True
    run2.tick(2.0)
    st = run2.status()[0]
    check(st["state"] == "failed" and len(ad2.cancelled) == 1, f"danger sound fails + cancels the task ({st['last_run']})")
    ad2.is_emergency = False

    # waits are timed on the sim clock; once-tasks end as done
    once = T.validate_task({"name": "wave", "trigger": {"kind": "time", "at": "09:00", "repeat": "once"},
                            "steps": [{"skill": "wait", "args": {"s": 3}}, {"skill": "say", "args": {"text": "Hello"}}]})[0]
    store.add(once)
    ad3 = FakeAdapter()
    run3 = T.TaskRunner(store, ad3, log=logs.append)
    run3.fire("wave")
    run3.tick(10.0)
    run3.tick(12.0)
    check(ad3.started == [], "wait not over after 2 s")
    run3.tick(13.1)
    check(ad3.started == [{"kind": "say", "text": "Hello"}], "say after the 3 s wait")
    ad3.status = "ok"
    run3.tick(13.2)
    wave = [s for s in run3.status() if s["name"] == "wave"][0]
    check(wave["state"] == "done" and wave["next_run"] is None, "once task -> done")

    # E1.4: a sound task replaces the built-in response to its sound (never tier 1)
    check(run3.sound_task("kettle") is None, "no kettle task -> built-in kettle response")
    tea = T.validate_task({"name": "tea", "trigger": {"kind": "sound", "sound": "kettle"},
                           "steps": [{"skill": "find_owner"}, {"skill": "alert_owner", "args": {"text": "Kettle."}}]})[0]
    store.add(tea)
    check(run3.sound_task("kettle") == "tea", "kettle task replaces the built-in kettle response")
    check(run3.sound_task("doorbell") is None, "other sounds keep the built-in response")
    store.tasks["sneaky"] = dict(tea, name="sneaky", trigger={"kind": "sound", "sound": "smoke_alarm"})
    check(run3.sound_task("smoke_alarm") is None, "a smoke task (even unvalidated) never replaces the alarm")
    del store.tasks["sneaky"]
    run3.remove("tea")
    check(run3.sound_task("kettle") is None, "forgotten kettle task -> built-in response again")

    # forget
    check(run3.remove("bed") is not None and "bedtime" not in store.tasks, "'forget the bed task' removes bedtime")
    check(run3.remove("nothing") is None, "forgetting an unknown task is harmless")


def _said(text, emergency=False):
    """Keyword path end to end: sentence -> rules.keyword_parse -> rules.validate."""
    import rules as R
    return R.validate(R.keyword_parse(text), emergency, text)


def test_teach_parsing():
    """E1.2: teach / forget / can't-do sentences in rules.py (keyword fallback = no Ollama needed)."""
    import rules as R
    print("teach parsing (rules.py, keywords)")
    instr, say = _said("Nudge, every night at 10 pm take me to bed")
    check(instr is not None and instr["type"] == "teach", "bedtime sentence -> teach")
    task = instr["task"] if instr else {}
    check(task.get("name") == "bedtime" and task.get("trigger") == {"kind": "time", "at": "22:00", "repeat": "daily"},
          f"bedtime trigger 22:00 daily ({task.get('name')}, {task.get('trigger')})")
    check([s["skill"] for s in task.get("steps", [])] == ["find_owner", "say", "guide_owner_to"]
          and task["steps"][-1]["args"] == {"target": "guest"}, "bedtime steps find -> say -> guide to guest")
    check(say == "OK. Every day at 10 pm I'll take you to bed.", f"confirmation: {say!r}")
    check(R.keyword_parse("every night at 10 take me to bed")["teach"]["at"] == "22:00", "'every night at 10' = 22:00")
    instr, _ = _said("Every night at 10 pm take me to bed")
    check(R.keyword_parse("take me to the bedroom")["kind"] == "go_to", "'take me to the bedroom' (no time) stays go_to now")

    instr, say = _said("When the kettle boils, just tell me")
    t = instr["task"] if instr else {}
    check(t.get("trigger", {}).get("sound") == "kettle" and [s["skill"] for s in t.get("steps", [])]
          == ["find_owner", "alert_owner"], f"kettle -> sound trigger + tell ({say!r})")
    check(say.startswith("OK. When I hear the kettle"), "kettle confirmation")

    instr, say = _said("Remind me to take my pills at 8 every morning")
    t = instr["task"] if instr else {}
    check(t.get("trigger", {}).get("at") == "08:00" and "your pills" in str(t.get("steps")),
          f"pills reminder 08:00 with 'your pills' ({say!r})")

    instr, say = _said("If I haven't moved for two hours, come and check on me")
    t = instr["task"] if instr else {}
    check(t.get("trigger") == {"kind": "idle", "minutes": 120.0}, f"idle 2 h ({t.get('trigger')})")

    instr, say = _said("When I go into the kitchen remind me to switch off the oven")
    t = instr["task"] if instr else {}
    check(t.get("trigger", {}).get("room") == "kitchen", f"room-entry trigger ({say!r})")

    for bad in ("When the smoke alarm goes, don't wake me", "If there's a fire just tell me"):
        instr, say = _said(bad)
        check(instr is None and "smoke alarm" in say, f"danger refused: {bad!r} -> {say!r}")
    instr, say = _said("Every day at 9 make me a coffee")
    check(instr is None and "can't" in say, f"unsupported teach -> can't ({say!r})")
    instr, say = _said("Please do the washing up")
    check(instr is None and "can't" in say, f"cant_do_that sentence -> can't ({say!r})")
    instr, say = _said("Every night take me to bed")
    check(instr is None and say == R.NO_TIME, "no time given -> asks when")
    instr, say = _said("Every night at 10 pm take me to bed", emergency=True)
    check(instr is None and "fire" in say, "no teaching during a fire")

    instr, _ = _said("Forget the bedtime task")
    check(instr == {"type": "forget_task", "name": "bedtime"}, f"forget -> forget_task ({instr})")
    instr, _ = _said("cancel the bed task")
    check(instr and instr["type"] == "forget_task", "'cancel the bed task' is forget, not stop")
    check(_said("stop")[0] == {"type": "stop"}, "'stop' still stops")
    check(R.forget_reply(None, "kettle") == "I don't have a kettle task.", "forget unknown reply")

    # LLM form merged with keywords (what CommandParser does when Gemma answers)
    llm = {"trigger": "time", "at": "10:00", "repeat": "daily", "sound": "none", "minutes": 0, "room": "none",
           "action": "guide", "target": "bed", "text": "", "name": "bedtime"}
    flat = R.merge_teach(llm, R.keyword_teach("every night at 10 pm take me to bed"))
    task, say = R.build_task(flat, "every night at 10 pm take me to bed")
    check(task is not None and task["trigger"]["at"] == "22:00", "regex clock beats the LLM's 10:00")
    llm_bad = dict(llm, trigger="sound", sound="doorbell")
    task, say = R.build_task(R.merge_teach(llm_bad, R.keyword_teach("when the smoke alarm goes tell me")),
                             "when the smoke alarm goes tell me")
    check(task is None and say == R.DANGER_TEACH, "LLM mapping smoke onto the doorbell is still refused")
    check(R.teach_schema()["properties"]["action"]["enum"] == R.TEACH_ACTIONS, "teach schema enum")


def test_ask():
    """ask.py (C5 brain side, E1.5): yes/no matching and the question life cycle."""
    import ask as A
    yn = ["yes", "no"]
    for text, want in [("yes", "yes"), ("Yes please.", "yes"), ("okay", "yes"), ("Nudge, yeah", "yes"),
                       ("no", "no"), ("No thanks", "no"), ("don't", "no"), ("please don't", "no"),
                       ("ok no", "no"), ("maybe", None), ("stop", None), ("", None),
                       ("no, take me to the kitchen right now", None)]:
        got = A.match_choice(text, yn)
        check(got == want, f"match_choice({text!r}) = {got!r}")
    got = []
    a = A.AskOwner()
    qid = a.ask("Every day at 10 pm I'll take you to bed. OK?", 0.0, got.append, timeout_s=10)
    check(a.status() == {"id": qid, "question": "Every day at 10 pm I'll take you to bed. OK?",
                         "choices": ["yes", "no"], "timeout_s": 10.0}, "ask status = C5 shape")
    check(not a.answer({"text": "take me to the kitchen"}, plain_command=True), "a command is not an answer")
    check(not a.answer({"id": "other", "text": "yes"}), "user_answer for another id ignored")
    check(a.answer({"id": qid, "text": "uh", "choice": "no"}) and got == ["no"], "user_answer choice wins")
    check(a.status() is None, "answered -> ask cleared")
    a.ask("Q?", 0.0, got.append, timeout_s=10)
    a.tick(5.0)
    check(len(got) == 1, "no timeout before timeout_s")
    a.tick(10.0)
    check(got[-1] is None and a.status() is None, "timeout -> callback(None), cleared")
    a.ask("Q1?", 0.0, got.append)
    a.ask("Q2?", 1.0, got.append)
    check(got[-1] is None and a.status()["question"] == "Q2?", "a new question times out the old one")
    check(a.answer({"text": "yes"}, plain_command=True) and got[-1] == "yes", "plain 'yes' answers")
    a.ask("Q3?", 0.0, got.append)
    n = len(got)
    a.clear()
    a.tick(100.0)
    check(len(got) == n and a.status() is None, "clear() drops silently (reset / fire)")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        test_teach_parsing()
        test_validator()
        test_compile()
        test_watcher()
        test_store_and_runner(tmp)
        test_ask()
    if FAILS:
        print(f"\n{len(FAILS)} FAILED")
        sys.exit(1)
    print("\ntest_tasks: all passed")


if __name__ == "__main__":
    main()
