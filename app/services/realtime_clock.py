from __future__ import annotations

import time


class RealtimeFrameClock:
    def __init__(self) -> None:
        self._start = time.perf_counter()
        self._last = 0.0

    def next_timestamp_seconds(self, now: float | None = None) -> float:
        current = time.perf_counter() if now is None else float(now)
        elapsed = max(0.0, current - self._start)
        if elapsed < self._last:
            return self._last
        self._last = elapsed
        return elapsed
