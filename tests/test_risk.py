from app.models import FrameMetric
from app.services.risk import StampedeRiskPredictor


def test_predictor_flags_high_density_pattern() -> None:
    predictor = StampedeRiskPredictor()
    metrics = [
        FrameMetric(
            timestamp_seconds=0.0,
            people_count=12,
            density_per_square_meter=3.8,
            motion_turbulence=0.5,
            occupancy_ratio=0.24,
            alert_triggered=False,
        ),
        FrameMetric(
            timestamp_seconds=1.0,
            people_count=16,
            density_per_square_meter=4.6,
            motion_turbulence=1.1,
            occupancy_ratio=0.33,
            alert_triggered=True,
        ),
        FrameMetric(
            timestamp_seconds=2.0,
            people_count=20,
            density_per_square_meter=5.9,
            motion_turbulence=1.7,
            occupancy_ratio=0.48,
            alert_triggered=True,
        ),
    ]

    result = predictor.assess(metrics, threshold_breach_frames=2)

    assert result.level in {"high", "critical"}
    assert result.score >= 0.6
    assert any("density" in reason.lower() for reason in result.rationale)


def test_predictor_uses_shared_default_density_profile(monkeypatch) -> None:
    monkeypatch.delenv("CROWD_DENSITY_ALERT", raising=False)
    monkeypatch.delenv("CROWD_DENSITY_DANGER", raising=False)
    monkeypatch.delenv("CROWD_DENSITY_CRITICAL", raising=False)
    monkeypatch.delenv("CROWD_DENSITY_SEVERE", raising=False)

    predictor = StampedeRiskPredictor()

    assert predictor.alert_density == predictor.danger_density == 1.5
    assert predictor.danger_density < predictor.critical_density < predictor.severe_density


def test_predictor_density_profile_respects_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("CROWD_DENSITY_ALERT", "2.0")
    monkeypatch.setenv("CROWD_DENSITY_DANGER", "1.8")
    monkeypatch.setenv("CROWD_DENSITY_CRITICAL", "2.9")
    monkeypatch.setenv("CROWD_DENSITY_SEVERE", "4.4")

    predictor = StampedeRiskPredictor()

    assert predictor.alert_density == 2.0
    assert predictor.danger_density == 1.8
    assert predictor.critical_density == 2.9
    assert predictor.severe_density == 4.4
