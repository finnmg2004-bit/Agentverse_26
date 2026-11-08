"""Sim clock (overnight contract C2): wall-clock time of day inside the simulated house.

The sim owns it; the brain, GUI and tests read `state.sim_clock` ("HH:MM") and `state.sim_day` (int,
0 = the start day). It advances 1:1 with MuJoCo sim time (`data.time`), starts at `--clock HH:MM`
(default 08:00), returns to the start time on [R] reset, and can be set live with the `gui` action
`{"action": "clock", "time": "21:59"}` (keeps the current day).
"""

from __future__ import annotations

MIN_PER_DAY = 24 * 60


def parse_hhmm(text: str) -> float:
    """"HH:MM" -> minutes after midnight. Raises ValueError on bad input."""
    hh, mm = str(text).strip().split(":")
    h, m = int(hh), int(mm)
    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise ValueError(f"time out of range: {text!r}")
    return float(h * 60 + m)


class SimClock:
    def __init__(self, start: str = "08:00", sim_t: float = 0.0) -> None:
        self.start_min = parse_hhmm(start)
        self._base_min = self.start_min  # absolute minutes since day 0 00:00 at sim time _base_t
        self._base_t = float(sim_t)

    def _abs_min(self, sim_t: float) -> float:
        return self._base_min + (float(sim_t) - self._base_t) / 60.0

    def reset(self, sim_t: float) -> None:
        """Back to the start time on day 0 ([R] reset)."""
        self._base_min = self.start_min
        self._base_t = float(sim_t)

    def set(self, text: str, sim_t: float) -> None:
        """Jump to HH:MM on the current day (GUI / tests)."""
        day = self.day(sim_t)
        self._base_min = day * MIN_PER_DAY + parse_hhmm(text)
        self._base_t = float(sim_t)

    def day(self, sim_t: float) -> int:
        return int(self._abs_min(sim_t) // MIN_PER_DAY)

    def hhmm(self, sim_t: float) -> str:
        m = int(self._abs_min(sim_t) % MIN_PER_DAY)
        return f"{m // 60:02d}:{m % 60:02d}"


if __name__ == "__main__":  # quick self-check: python sim/sim_clock.py
    c = SimClock("08:00", 0.0)
    assert (c.hhmm(0.0), c.day(0.0)) == ("08:00", 0)
    assert c.hhmm(59.9) == "08:00" and c.hhmm(60.0) == "08:01"
    c.set("21:59", 100.0)
    assert c.hhmm(100.0) == "21:59" and c.hhmm(160.0) == "22:00"
    c.set("23:59", 200.0)
    assert (c.hhmm(260.0), c.day(260.0)) == ("00:00", 1)
    c.set("07:00", 300.0)  # keeps day 1
    assert (c.hhmm(300.0), c.day(300.0)) == ("07:00", 1)
    c.reset(400.0)
    assert (c.hhmm(400.0), c.day(400.0)) == ("08:00", 0)
    for bad in ("24:00", "8", "ab:cd", "12:60"):
        try:
            parse_hhmm(bad)
            raise AssertionError(bad)
        except ValueError:
            pass
    print("sim_clock OK")
