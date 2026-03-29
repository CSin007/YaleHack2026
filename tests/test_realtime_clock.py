import pytest

from app.services.realtime_clock import RealtimeFrameClock


def test_realtime_clock_returns_elapsed_seconds() -> None:
    clock = RealtimeFrameClock()
    clock._start = 100.0

    t1 = clock.next_timestamp_seconds(now=100.20)
    t2 = clock.next_timestamp_seconds(now=100.65)

    assert t1 == pytest.approx(0.20, abs=1e-9)
    assert t2 == pytest.approx(0.65, abs=1e-9)


def test_realtime_clock_clamps_when_time_goes_backwards() -> None:
    clock = RealtimeFrameClock()
    clock._start = 50.0

    t1 = clock.next_timestamp_seconds(now=50.40)
    t2 = clock.next_timestamp_seconds(now=50.10)
    t3 = clock.next_timestamp_seconds(now=50.90)

    assert t1 == pytest.approx(0.40, abs=1e-9)
    assert t2 == pytest.approx(0.40, abs=1e-9)
    assert t3 == pytest.approx(0.90, abs=1e-9)
