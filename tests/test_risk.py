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
