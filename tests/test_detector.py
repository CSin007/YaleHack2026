import cv2
import numpy as np

from app.services.detector import CrowdAnalyzer, LightweightPersonTracker, NormalizedZone


def test_adaptive_sample_step_stays_base_on_calm_signal() -> None:
    step = CrowdAnalyzer._adaptive_sample_step(
        base_step=12,
        density=0.8,
        density_threshold=2.5,
        occupancy_ratio=0.10,
        turbulence=0.20,
    )
    assert step == 12


def test_adaptive_sample_step_tightens_on_medium_pressure() -> None:
    step = CrowdAnalyzer._adaptive_sample_step(
        base_step=12,
        density=1.7,
        density_threshold=2.5,
        occupancy_ratio=0.16,
        turbulence=0.40,
    )
    assert step == 6


def test_adaptive_sample_step_tightens_aggressively_on_high_pressure() -> None:
    step = CrowdAnalyzer._adaptive_sample_step(
        base_step=12,
        density=2.2,
        density_threshold=2.5,
        occupancy_ratio=0.30,
        turbulence=0.30,
    )
    assert step == 3


def _make_hybrid_analyzer() -> CrowdAnalyzer:
    analyzer = CrowdAnalyzer()
    analyzer.detector_mode = "hybrid"
    analyzer.yolo_model = object()
    analyzer.hybrid_occ_hint_min = 0.34
    analyzer.hybrid_confidence_max = 0.32
    return analyzer


def test_hybrid_fallback_requires_low_confidence_trend() -> None:
    analyzer = _make_hybrid_analyzer()
    use_fallback = analyzer._should_enable_hybrid_fallback(
        yolo_detections=[(10, 10, 40, 120, 0.52)],
        yolo_confidence=0.52,
        occ_hint=0.40,
        recent_yolo_confidence=0.45,
    )
    assert use_fallback is False


def test_hybrid_fallback_enables_when_confidence_trend_is_low() -> None:
    analyzer = _make_hybrid_analyzer()
    use_fallback = analyzer._should_enable_hybrid_fallback(
        yolo_detections=[(10, 10, 40, 120, 0.20)],
        yolo_confidence=0.20,
        occ_hint=0.38,
        recent_yolo_confidence=0.28,
    )
    assert use_fallback is True


def test_hybrid_fallback_requires_occupancy_gate() -> None:
    analyzer = _make_hybrid_analyzer()
    use_fallback = analyzer._should_enable_hybrid_fallback(
        yolo_detections=[(10, 10, 40, 120, 0.20)],
        yolo_confidence=0.20,
        occ_hint=0.20,
        recent_yolo_confidence=0.24,
    )
    assert use_fallback is False


def _synthetic_camera_shift_frames() -> tuple[np.ndarray, np.ndarray]:
    base = np.zeros((180, 260), dtype=np.uint8)
    for y in range(20, 170, 30):
        for x in range(20, 250, 30):
            cv2.circle(base, (x, y), 4, 255, -1)
            cv2.rectangle(base, (x - 8, y - 8), (x - 3, y - 3), 160, -1)
    matrix = np.float32([[1, 0, 9], [0, 1, 6]])
    shifted = cv2.warpAffine(base, matrix, (base.shape[1], base.shape[0]), flags=cv2.INTER_LINEAR)
    return base, shifted


def _make_motion_analyzer() -> CrowdAnalyzer:
    analyzer = object.__new__(CrowdAnalyzer)
    analyzer.motion_compensation_enabled = True
    analyzer.motion_comp_min_inlier_ratio = 0.0
    analyzer.motion_comp_max_corners = 180
    analyzer.motion_comp_quality_level = 0.01
    analyzer.motion_comp_min_distance = 7.0
    analyzer.motion_comp_ransac_thresh = 3.0
    analyzer.motion_comp_max_translation_ratio = 0.25
    return analyzer


def test_global_motion_compensation_reduces_frame_misalignment() -> None:
    prev_gray, curr_gray = _synthetic_camera_shift_frames()
    stabilized, inlier_ratio = CrowdAnalyzer._compensate_global_motion(
        prev_gray=prev_gray,
        curr_gray=curr_gray,
        max_corners=180,
        quality_level=0.01,
        min_distance=7.0,
        ransac_thresh=3.0,
        max_translation_ratio=0.25,
    )
    before = float(np.mean(cv2.absdiff(prev_gray, curr_gray)))
    after = float(np.mean(cv2.absdiff(prev_gray, stabilized)))
    assert inlier_ratio >= 0.25
    assert after < before * 0.70


def test_motion_turbulence_drops_with_camera_compensation() -> None:
    analyzer = _make_motion_analyzer()
    prev_gray, curr_gray = _synthetic_camera_shift_frames()

    analyzer.motion_compensation_enabled = False
    raw_turbulence = analyzer._estimate_motion_turbulence(prev_gray, curr_gray)

    analyzer.motion_compensation_enabled = True
    compensated_turbulence = analyzer._estimate_motion_turbulence(prev_gray, curr_gray)

    assert compensated_turbulence < raw_turbulence


def test_realtime_session_default_threshold_uses_shared_profile(monkeypatch) -> None:
    monkeypatch.delenv("CROWD_DENSITY_ALERT", raising=False)
    monkeypatch.setenv("CROWD_DENSITY_DANGER", "1.7")
    monkeypatch.setenv("CROWD_DENSITY_CRITICAL", "2.8")
    monkeypatch.setenv("CROWD_DENSITY_SEVERE", "4.2")

    analyzer = CrowdAnalyzer()
    session = analyzer.create_realtime_session(
        area_value=100.0,
        area_unit="square_meters",
        zone=NormalizedZone(),
        alert_density_threshold=None,
    )
    assert analyzer.default_density_threshold == analyzer.risk_predictor.alert_density
    assert session.density_threshold == analyzer.risk_predictor.alert_density
    assert session.density_threshold == analyzer.risk_predictor.danger_density


def test_lightweight_tracker_stabilizes_count_across_small_jitter() -> None:
    tracker = LightweightPersonTracker(max_missed=2, min_confirmed_hits=2, max_match_distance_ratio=0.25)
    frame_shape = (220, 320)

    detections_1 = [(30, 40, 30, 80, 0.86), (170, 46, 32, 82, 0.89)]
    detections_2 = [(33, 42, 30, 80, 0.84), (167, 45, 32, 82, 0.90)]

    c1 = tracker.update(detections_1, frame_shape)
    c2 = tracker.update(detections_2, frame_shape)

    assert c1 == 2
    assert c2 == 2


def test_lightweight_tracker_expires_tracks_after_missed_frames() -> None:
    tracker = LightweightPersonTracker(max_missed=1, min_confirmed_hits=1, max_match_distance_ratio=0.25)
    frame_shape = (220, 320)

    c1 = tracker.update([(40, 50, 30, 80, 0.9)], frame_shape)
    c2 = tracker.update([], frame_shape)
    c3 = tracker.update([], frame_shape)

    assert c1 == 1
    assert c2 == 1
    assert c3 == 0


def test_people_count_uses_tracker_anchor_when_detections_drop() -> None:
    no_tracking = CrowdAnalyzer._estimate_people_count(
        detections=[],
        area_sq_m=10.0,
        occ_ratio=0.12,
        roi_shape=(180, 240),
        tracked_count=None,
    )
    with_tracking = CrowdAnalyzer._estimate_people_count(
        detections=[],
        area_sq_m=10.0,
        occ_ratio=0.12,
        roi_shape=(180, 240),
        tracked_count=6,
    )
    assert no_tracking == 0
    assert with_tracking >= 6
