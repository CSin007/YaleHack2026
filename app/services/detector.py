from __future__ import annotations

import base64
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.models import DetectionBox, FrameMetric, RealtimeAnalysisResult, VideoAnalysisResult
from app.services.heatmap import CrowdHeatmap
from app.services.risk import StampedeRiskPredictor

SQUARE_FEET_TO_SQUARE_METERS = 0.092903


@dataclass(slots=True)
class NormalizedZone:
    x: float = 0.0
    y: float = 0.0
    width: float = 1.0
    height: float = 1.0

    def validate(self) -> None:
        values = (self.x, self.y, self.width, self.height)
        if any(v < 0.0 or v > 1.0 for v in values):
            raise ValueError("Zone values must be between 0.0 and 1.0.")
        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError("Zone width and height must be greater than zero.")
        if self.x + self.width > 1.0 or self.y + self.height > 1.0:
            raise ValueError("Zone rectangle must stay inside the video frame.")

    def as_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}


@dataclass(slots=True)
class TrackState:
    cx: float
    cy: float
    w: float
    h: float
    hits: int = 1
    missed: int = 0


class LightweightPersonTracker:
    def __init__(
        self,
        max_missed: int = 2,
        min_confirmed_hits: int = 2,
        max_match_distance_ratio: float = 0.18,
    ) -> None:
        self.max_missed = max(1, int(max_missed))
        self.min_confirmed_hits = max(1, int(min_confirmed_hits))
        self.max_match_distance_ratio = float(np.clip(max_match_distance_ratio, 0.05, 0.6))
        self._tracks: dict[int, TrackState] = {}
        self._next_id = 1

    def update(
        self,
        detections: list[tuple[int, int, int, int, float]],
        frame_shape: tuple[int, int],
    ) -> int:
        if not self._tracks and not detections:
            return 0

        h, w = frame_shape
        frame_diag = float(max(np.hypot(h, w), 1.0))
        max_dist = frame_diag * self.max_match_distance_ratio

        det_points = [(idx, x + bw * 0.5, y + bh * 0.5, bw, bh, conf) for idx, (x, y, bw, bh, conf) in enumerate(detections)]
        det_points.sort(key=lambda item: item[5], reverse=True)

        matched_track_ids: set[int] = set()
        matched_det_ids: set[int] = set()

        for det_idx, dcx, dcy, dw, dh, _conf in det_points:
            best_track_id = -1
            best_dist = float("inf")
            for track_id, track in self._tracks.items():
                if track_id in matched_track_ids:
                    continue
                dist = float(np.hypot(dcx - track.cx, dcy - track.cy))
                if dist <= max_dist and dist < best_dist:
                    best_dist = dist
                    best_track_id = track_id
            if best_track_id < 0:
                continue
            track = self._tracks[best_track_id]
            track.cx = dcx
            track.cy = dcy
            track.w = dw
            track.h = dh
            track.hits += 1
            track.missed = 0
            matched_track_ids.add(best_track_id)
            matched_det_ids.add(det_idx)

        for det_idx, dcx, dcy, dw, dh, _conf in det_points:
            if det_idx in matched_det_ids:
                continue
            track_id = self._next_id
            self._tracks[track_id] = TrackState(cx=dcx, cy=dcy, w=dw, h=dh)
            self._next_id += 1
            matched_track_ids.add(track_id)

        stale_ids: list[int] = []
        for track_id, track in self._tracks.items():
            if track_id in matched_track_ids:
                continue
            track.missed += 1
            if track.missed > self.max_missed:
                stale_ids.append(track_id)
        for track_id in stale_ids:
            self._tracks.pop(track_id, None)

        stable = 0
        for track in self._tracks.values():
            is_confirmed = track.hits >= self.min_confirmed_hits
            is_recent = track.missed == 0
            if is_confirmed or is_recent:
                stable += 1
        return stable


class CrowdAnalyzer:
    def __init__(self) -> None:
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.risk_predictor = StampedeRiskPredictor()
        self.default_density_threshold = self.risk_predictor.alert_density

        self.yolo_model = None
        self.detector_backend = "hog"
        self.detector_mode = os.getenv("CROWD_DETECTOR_MODE", "yolo").strip().lower()
        if self.detector_mode not in {"yolo", "hybrid", "hog"}:
            self.detector_mode = "yolo"
        self.hybrid_occ_hint_min = float(np.clip(float(os.getenv("HYBRID_OCC_HINT_MIN", "0.34")), 0.0, 1.0))
        self.hybrid_confidence_max = float(np.clip(float(os.getenv("HYBRID_YOLO_CONF_FALLBACK_MAX", "0.32")), 0.0, 1.0))
        self.hybrid_confidence_window = max(2, int(os.getenv("HYBRID_YOLO_CONF_WINDOW", "8")))
        self.motion_compensation_enabled = os.getenv("MOTION_COMPENSATION", "1").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        self.motion_comp_min_inlier_ratio = float(
            np.clip(float(os.getenv("MOTION_COMP_MIN_INLIER_RATIO", "0.42")), 0.0, 1.0)
        )
        self.motion_comp_max_corners = max(40, int(os.getenv("MOTION_COMP_MAX_CORNERS", "180")))
        self.motion_comp_quality_level = float(
            np.clip(float(os.getenv("MOTION_COMP_QUALITY_LEVEL", "0.01")), 0.001, 0.2)
        )
        self.motion_comp_min_distance = float(max(2.0, float(os.getenv("MOTION_COMP_MIN_DISTANCE", "7.0"))))
        self.motion_comp_ransac_thresh = float(max(0.5, float(os.getenv("MOTION_COMP_RANSAC_THRESH", "3.0"))))
        self.motion_comp_max_translation_ratio = float(
            np.clip(float(os.getenv("MOTION_COMP_MAX_TRANSLATION_RATIO", "0.18")), 0.02, 0.5)
        )
        self.tracker_enabled = os.getenv("CROWD_TRACKER_ENABLED", "1").strip().lower() in {"1", "true", "yes"}
        self.tracker_max_missed = max(1, int(os.getenv("CROWD_TRACKER_MAX_MISSED", "2")))
        self.tracker_min_confirmed_hits = max(1, int(os.getenv("CROWD_TRACKER_MIN_CONFIRMED_HITS", "2")))
        self.tracker_max_match_distance_ratio = float(
            np.clip(float(os.getenv("CROWD_TRACKER_MAX_MATCH_DISTANCE_RATIO", "0.18")), 0.05, 0.6)
        )
        self._setup_yolo()

    def analyze_video(
        self,
        video_path: Path,
        area_value: float,
        area_unit: str,
        zone: NormalizedZone,
        alert_density_threshold: float | None = None,
    ) -> VideoAnalysisResult:
        if area_value <= 0:
            raise ValueError("The monitored area must be greater than zero.")

        zone.validate()
        area_sq_m = self._to_square_meters(area_value, area_unit)
        density_thresh = alert_density_threshold or self.default_density_threshold
        if density_thresh <= 0:
            raise ValueError("The alert density threshold must be greater than zero.")

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError("Unable to open the uploaded video.")

        frame_metrics: list[FrameMetric] = []
        prev_gray: np.ndarray | None = None
        base_sample_step = self._sample_step(capture)
        sample_step = base_sample_step
        frame_index = threshold_breach = 0
        next_process_index = 0
        yolo_conf_history: deque[float] = deque(maxlen=self.hybrid_confidence_window)
        heatmap_max_density = float(os.getenv("HEATMAP_MAX_DENSITY", "4.0"))
        heatmap: CrowdHeatmap | None = None

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index < next_process_index:
                    frame_index += 1
                    continue

                roi = self._extract_roi(frame, zone)
                recent_yolo_conf = self._rolling_confidence(yolo_conf_history)
                detections, yolo_confidence = self._detect_people_with_meta(
                    roi,
                    recent_yolo_confidence=recent_yolo_conf,
                )
                if self.yolo_model is not None and self.detector_mode in {"yolo", "hybrid"}:
                    yolo_conf_history.append(yolo_confidence)
                occ_ratio = self._estimate_occupancy_ratio(roi)

                if heatmap is None:
                    heatmap = CrowdHeatmap(
                        frame_shape=(roi.shape[0], roi.shape[1]),
                        decay=0.55,
                        blur_ksize=31,
                        alpha=0.40,
                    )
                heatmap.update(roi, detections)

                det_conf = self._detection_confidence(detections)
                count = heatmap.estimate_count(area_sq_m, heatmap_max_density)
                density = count / area_sq_m
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                turb = self._estimate_motion_turbulence(prev_gray, gray)
                prev_gray = gray

                alert = density >= density_thresh
                if alert:
                    threshold_breach += 1

                sample_step = self._adaptive_sample_step(
                    base_step=base_sample_step,
                    density=density,
                    density_threshold=density_thresh,
                    occupancy_ratio=occ_ratio,
                    turbulence=turb,
                )
                next_process_index = frame_index + sample_step

                frame_metrics.append(
                    FrameMetric(
                        timestamp_seconds=round(capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, 2),
                        people_count=count,
                        density_per_square_meter=round(density, 3),
                        motion_turbulence=round(turb, 3),
                        occupancy_ratio=round(occ_ratio, 3),
                        detection_confidence=round(det_conf, 3),
                        alert_triggered=alert,
                    )
                )
                frame_index += 1
        finally:
            capture.release()

        if not frame_metrics:
            raise ValueError("No frames could be analyzed from the video.")

        avg_count = sum(frame.people_count for frame in frame_metrics) / len(frame_metrics)
        peak_count = max(frame.people_count for frame in frame_metrics)
        avg_density = sum(frame.density_per_square_meter for frame in frame_metrics) / len(frame_metrics)
        peak_density = max(frame.density_per_square_meter for frame in frame_metrics)
        risk = self.risk_predictor.assess(frame_metrics, threshold_breach)

        return VideoAnalysisResult(
            frame_sample_count=len(frame_metrics),
            avg_people_count=round(avg_count, 2),
            peak_people_count=peak_count,
            avg_density_per_square_meter=round(avg_density, 3),
            peak_density_per_square_meter=round(peak_density, 3),
            density_threshold_per_square_meter=round(density_thresh, 3),
            threshold_breach_frames=threshold_breach,
            monitored_area_square_meters=round(area_sq_m, 3),
            monitored_area_original_unit=area_unit,
            monitored_area_original_value=area_value,
            zone=zone.as_dict(),
            frame_metrics=frame_metrics,
            risk_assessment=risk,
        )

    def create_realtime_session(
        self,
        area_value: float,
        area_unit: str,
        zone: NormalizedZone,
        alert_density_threshold: float | None = None,
        history_size: int = 20,
    ) -> "RealtimeCrowdSession":
        if area_value <= 0:
            raise ValueError("The monitored area must be greater than zero.")

        zone.validate()
        area_sq_m = self._to_square_meters(area_value, area_unit)
        density_thresh = alert_density_threshold or self.default_density_threshold
        if density_thresh <= 0:
            raise ValueError("The alert density threshold must be greater than zero.")

        return RealtimeCrowdSession(
            analyzer=self,
            area_square_meters=area_sq_m,
            area_unit=area_unit,
            area_value=area_value,
            zone=zone,
            density_threshold=density_thresh,
            history_size=history_size,
        )

    @staticmethod
    def _to_square_meters(area_value: float, area_unit: str) -> float:
        unit = area_unit.strip().lower()
        if unit == "square_meters":
            return area_value
        if unit == "square_feet":
            return area_value * SQUARE_FEET_TO_SQUARE_METERS
        raise ValueError("Area unit must be 'square_feet' or 'square_meters'.")

    @staticmethod
    def _sample_step(capture: cv2.VideoCapture) -> int:
        fps = capture.get(cv2.CAP_PROP_FPS)
        return max(int(fps // 2), 1) if fps > 0 else 5

    def _build_tracker(self) -> LightweightPersonTracker | None:
        if not self.tracker_enabled:
            return None
        return LightweightPersonTracker(
            max_missed=self.tracker_max_missed,
            min_confirmed_hits=self.tracker_min_confirmed_hits,
            max_match_distance_ratio=self.tracker_max_match_distance_ratio,
        )

    @staticmethod
    def _adaptive_sample_step(
        base_step: int,
        density: float,
        density_threshold: float,
        occupancy_ratio: float,
        turbulence: float,
    ) -> int:
        base_step = max(int(base_step), 1)
        if base_step == 1:
            return 1

        high_pressure = (
            density >= density_threshold * 0.85
            or occupancy_ratio >= 0.28
            or turbulence >= 0.85
        )
        medium_pressure = (
            density >= density_threshold * 0.60
            or occupancy_ratio >= 0.18
            or turbulence >= 0.45
        )

        if high_pressure:
            return max(1, base_step // 4)
        if medium_pressure:
            return max(1, base_step // 2)
        return base_step

    @staticmethod
    def _extract_roi(frame: np.ndarray, zone: NormalizedZone) -> np.ndarray:
        h, w = frame.shape[:2]
        x0, y0 = int(zone.x * w), int(zone.y * h)
        x1, y1 = int((zone.x + zone.width) * w), int((zone.y + zone.height) * h)
        return frame[y0:y1, x0:x1]

    def _setup_yolo(self) -> None:
        model_path = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")
        try:
            from ultralytics import YOLO

            self.yolo_model = YOLO(model_path)
            self.detector_backend = "yolo"
        except Exception:
            self.yolo_model = None
            self.detector_backend = "hog"

    def _detect_people_yolo(self, roi: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        if self.yolo_model is None:
            return []

        try:
            conf_thresh = float(os.getenv("YOLO_CONFIDENCE", "0.25"))
            iou_thresh = float(os.getenv("YOLO_IOU", "0.45"))
            imgsz = int(os.getenv("YOLO_IMGSZ", "640"))
            aspect_min = float(os.getenv("YOLO_ASPECT_MIN", "0.95"))
            aspect_max = float(os.getenv("YOLO_ASPECT_MAX", "5.2"))
            area_min_ratio = float(os.getenv("YOLO_AREA_MIN_RATIO", "0.00030"))
            area_max_ratio = float(os.getenv("YOLO_AREA_MAX_RATIO", "0.48"))
            edge_margin = float(os.getenv("YOLO_EDGE_CONF_MARGIN", "0.18"))
            use_edge_filter = os.getenv("YOLO_EDGE_FILTER", "0").strip().lower() in {"1", "true", "yes"}

            results = self.yolo_model.predict(
                source=roi,
                classes=[0],
                conf=conf_thresh,
                iou=iou_thresh,
                imgsz=imgsz,
                verbose=False,
                device="cpu",
            )
            if not results:
                return []

            boxes = getattr(results[0], "boxes", None)
            if boxes is None:
                return []

            out: list[tuple[int, int, int, int, float]] = []
            h, w = roi.shape[:2]
            roi_area = float(max(h * w, 1))

            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy()) if hasattr(box.conf[0], "cpu") else float(box.conf[0])

                x1 = int(max(0, min(w - 1, xyxy[0])))
                y1 = int(max(0, min(h - 1, xyxy[1])))
                x2 = int(max(0, min(w, xyxy[2])))
                y2 = int(max(0, min(h, xyxy[3])))

                bw = max(1, x2 - x1)
                bh = max(1, y2 - y1)
                area_ratio = (bw * bh) / roi_area
                aspect = bh / max(float(bw), 1.0)

                # Person-like shape/size filtering to suppress false positives.
                if aspect < aspect_min or aspect > aspect_max:
                    continue
                if area_ratio < area_min_ratio or area_ratio > area_max_ratio:
                    continue

                # Downweight clipped edge boxes unless confidence is strong.
                touching_edge = x1 <= 1 or y1 <= 1 or x2 >= (w - 1) or y2 >= (h - 1)
                if use_edge_filter and touching_edge and conf < (conf_thresh + edge_margin):
                    continue

                out.append((x1, y1, bw, bh, conf))

            if not out:
                return []

            box_np = np.array([[x, y, bw, bh, conf] for x, y, bw, bh, conf in out], dtype=np.float32)
            kept = self._non_max_suppression(box_np, 0.45)
            return [(int(b[0]), int(b[1]), int(b[2]), int(b[3]), float(b[4])) for b in kept]
        except Exception:
            return []

    def _enhance_for_detection(self, roi: np.ndarray) -> np.ndarray:
        ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        y = clahe.apply(y)
        merged = cv2.merge((y, cr, cb))
        return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

    def _detect_people(self, roi: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        detections, _ = self._detect_people_with_meta(roi, recent_yolo_confidence=None)
        return detections

    @staticmethod
    def _rolling_confidence(history: deque[float]) -> float | None:
        if not history:
            return None
        return float(sum(history) / len(history))

    def _should_enable_hybrid_fallback(
        self,
        yolo_detections: list[tuple[int, int, int, int, float]],
        yolo_confidence: float,
        occ_hint: float,
        recent_yolo_confidence: float | None,
    ) -> bool:
        if self.detector_mode != "hybrid" or self.yolo_model is None:
            return False
        if len(yolo_detections) > 1:
            return False
        if occ_hint < self.hybrid_occ_hint_min:
            return False

        trend_conf = recent_yolo_confidence if recent_yolo_confidence is not None else yolo_confidence
        return trend_conf <= self.hybrid_confidence_max or yolo_confidence <= self.hybrid_confidence_max

    def _detect_people_with_meta(
        self,
        roi: np.ndarray,
        recent_yolo_confidence: float | None,
    ) -> tuple[list[tuple[int, int, int, int, float]], float]:
        if self.detector_mode == "hog":
            hog_only = self._detect_people_hog(roi)
            self.detector_backend = "hog"
            return hog_only, 0.0

        yolo_detections = self._detect_people_yolo(roi)
        yolo_confidence = self._detection_confidence(yolo_detections)
        if self.detector_mode == "yolo" and self.yolo_model is not None:
            self.detector_backend = "yolo"
            return yolo_detections, yolo_confidence

        occ_hint = self._estimate_occupancy_ratio(roi)
        if self._should_enable_hybrid_fallback(
            yolo_detections=yolo_detections,
            yolo_confidence=yolo_confidence,
            occ_hint=occ_hint,
            recent_yolo_confidence=recent_yolo_confidence,
        ):
            if occ_hint >= self.hybrid_occ_hint_min:
                hog_hints = self._detect_people_hog(roi)
                if hog_hints:
                    max_hog = max(1, int(os.getenv("HYBRID_HOG_MAX_ADD", "2")))
                    yolo_detections = self._merge_detections(yolo_detections, hog_hints[:max_hog])

        if yolo_detections:
            self.detector_backend = "yolo"
            return yolo_detections, yolo_confidence

        hog_detections = self._detect_people_hog(roi)
        if hog_detections:
            self.detector_backend = "hog"
            return hog_detections, yolo_confidence

        self.detector_backend = "hog"
        return [], yolo_confidence

    def _detect_people_hog(self, roi: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        self.detector_backend = "hog"
        roi_enhanced = self._enhance_for_detection(roi)
        rects, weights = self.hog.detectMultiScale(
            roi_enhanced,
            winStride=(4, 4),
            padding=(8, 8),
            scale=1.04,
        )
        if len(rects) == 0:
            return []

        # Raise weight threshold significantly to suppress false positives in non-busy areas.
        # Filter by aspect ratio: person boxes must be taller than wide (HOG window is 64x128).
        weighted = [
            [*r, float(w)]
            for r, w in zip(rects, weights)
            if float(w) >= 0.55 and r[3] >= r[2] * 1.1
        ]
        if not weighted:
            return []

        boxes = np.array(weighted)
        kept = self._non_max_suppression(boxes, 0.42)
        return [(int(b[0]), int(b[1]), int(b[2]), int(b[3]), float(b[4])) for b in kept]

    @staticmethod
    def _iou(a: tuple[int, int, int, int, float], b: tuple[int, int, int, int, float]) -> float:
        ax1, ay1, aw, ah, _ = a
        bx1, by1, bw, bh, _ = b
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh

        xx1 = max(ax1, bx1)
        yy1 = max(ay1, by1)
        xx2 = min(ax2, bx2)
        yy2 = min(ay2, by2)

        iw = max(0, xx2 - xx1)
        ih = max(0, yy2 - yy1)
        inter = float(iw * ih)
        if inter <= 0.0:
            return 0.0
        area_a = float(max(1, aw * ah))
        area_b = float(max(1, bw * bh))
        return inter / (area_a + area_b - inter + 1e-6)

    def _merge_detections(
        self,
        primary: list[tuple[int, int, int, int, float]],
        secondary: list[tuple[int, int, int, int, float]],
        iou_thresh: float = 0.35,
    ) -> list[tuple[int, int, int, int, float]]:
        merged = list(primary)
        for cand in secondary:
            overlap = any(self._iou(cand, existing) >= iou_thresh for existing in merged)
            if not overlap:
                # Slightly downweight HOG confidence so YOLO remains dominant in scoring.
                merged.append((cand[0], cand[1], cand[2], cand[3], min(0.78, cand[4])))
        return merged

    @staticmethod
    def _estimate_occupancy_ratio(roi: np.ndarray) -> float:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 60, 140)
        edge_ratio = float(np.count_nonzero(edges)) / float(edges.size)

        lap = cv2.Laplacian(gray, cv2.CV_32F)
        texture_ratio = min(float(np.mean(np.abs(lap))) / 28.0, 1.0)

        fg = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7)
        fg_ratio = float(np.count_nonzero(fg)) / float(fg.size)

        return float(min(max(edge_ratio * 0.35 + texture_ratio * 0.35 + fg_ratio * 0.30, 0.0), 0.95))

    @staticmethod
    def _estimate_people_count(
        detections: list[tuple[int, int, int, int, float]],
        area_sq_m: float,
        occ_ratio: float,
        roi_shape: tuple[int, int],
        tracked_count: int | None = None,
    ) -> int:
        box_count = len(detections)

        # Primary: use tracker as a lower bound (handles momentary missed detections).
        base = max(box_count, tracked_count) if (tracked_count is not None and tracked_count > 0) else box_count

        # No detections at all — very conservatively fall back to occupancy signal.
        if base == 0:
            if occ_ratio < 0.15:
                return 0
            occ_estimate = occ_ratio * 2.5 * area_sq_m
            return int(round(np.clip(occ_estimate, 0.0, max(8.0, area_sq_m * 6.0))))

        # Small upward adjustment for lower-confidence detections (partial occlusion).
        confs = np.array([conf for *_, conf in detections], dtype=np.float32) if detections else np.array([0.5])
        mean_conf = float(np.mean(confs))
        scale = 1.0 + (1.0 - mean_conf) * 0.25
        estimate = max(base, int(round(base * scale)))

        hard_cap = int(max(30.0, area_sq_m * 12.0))
        return int(np.clip(estimate, base, hard_cap))

    @staticmethod
    def _detection_confidence(detections: list[tuple[int, int, int, int, float]]) -> float:
        if not detections:
            return 0.0
        confs = [max(0.0, min(1.0, conf)) for *_, conf in detections]
        return float(np.mean(confs))

    @staticmethod
    def _non_max_suppression(boxes: np.ndarray, overlap_thresh: float) -> list[np.ndarray]:
        if len(boxes) == 0:
            return []

        boxes = boxes.astype("float")
        picked = []
        x1, y1 = boxes[:, 0], boxes[:, 1]
        x2, y2 = boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]
        scores = boxes[:, 4]
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(scores)

        while len(idxs):
            last = len(idxs) - 1
            i = idxs[last]
            picked.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            overlap = (w * h) / area[idxs[:last]]
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

        return [boxes[i] for i in picked]

    @staticmethod
    def _compensate_global_motion(
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        max_corners: int,
        quality_level: float,
        min_distance: float,
        ransac_thresh: float,
        max_translation_ratio: float,
    ) -> tuple[np.ndarray, float]:
        prev_pts = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance,
            blockSize=7,
        )
        if prev_pts is None or len(prev_pts) < 10:
            return curr_gray, 0.0

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            curr_gray,
            prev_pts,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        if curr_pts is None or status is None:
            return curr_gray, 0.0

        status = status.reshape(-1).astype(bool)
        prev_valid = prev_pts.reshape(-1, 2)[status]
        curr_valid = curr_pts.reshape(-1, 2)[status]
        if len(prev_valid) < 8 or len(curr_valid) < 8:
            return curr_gray, 0.0

        matrix, inliers = cv2.estimateAffinePartial2D(
            prev_valid,
            curr_valid,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_thresh,
            maxIters=2000,
            confidence=0.99,
            refineIters=12,
        )
        if matrix is None:
            return curr_gray, 0.0

        tx = float(matrix[0, 2])
        ty = float(matrix[1, 2])
        h, w = curr_gray.shape[:2]
        max_shift = max(h, w) * max_translation_ratio
        if abs(tx) > max_shift or abs(ty) > max_shift:
            return curr_gray, 0.0

        inlier_ratio = float(np.mean(inliers)) if inliers is not None and len(inliers) else 0.0
        inv = cv2.invertAffineTransform(matrix)
        stabilized = cv2.warpAffine(
            curr_gray,
            inv,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )
        return stabilized, inlier_ratio

    def _estimate_motion_turbulence(self, prev_gray: np.ndarray | None, curr_gray: np.ndarray) -> float:
        if prev_gray is None:
            return 0.0

        flow_curr = curr_gray
        if self.motion_compensation_enabled:
            stabilized, inlier_ratio = self._compensate_global_motion(
                prev_gray=prev_gray,
                curr_gray=curr_gray,
                max_corners=self.motion_comp_max_corners,
                quality_level=self.motion_comp_quality_level,
                min_distance=self.motion_comp_min_distance,
                ransac_thresh=self.motion_comp_ransac_thresh,
                max_translation_ratio=self.motion_comp_max_translation_ratio,
            )
            if inlier_ratio >= self.motion_comp_min_inlier_ratio:
                flow_curr = stabilized

        flow = cv2.calcOpticalFlowFarneback(prev_gray, flow_curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return float(np.std(ang) * 0.15 + np.mean(mag))


class RealtimeCrowdSession:
    def __init__(
        self,
        analyzer: CrowdAnalyzer,
        area_square_meters: float,
        area_unit: str,
        area_value: float,
        zone: NormalizedZone,
        density_threshold: float,
        history_size: int,
    ) -> None:
        self.analyzer = analyzer
        self.area_square_meters = area_square_meters
        self.area_unit = area_unit
        self.area_value = area_value
        self.zone = zone
        self.density_threshold = density_threshold
        self.history_size = history_size
        self.previous_gray: np.ndarray | None = None
        self.frame_metrics: list[FrameMetric] = []
        self.threshold_breach_frames = 0
        self.heatmap: CrowdHeatmap | None = None
        self._smoothed_count: float | None = None
        self._smoothed_density: float | None = None
        self._smoothed_occ: float | None = None
        self._smoothed_turb: float | None = None
        self._smoothed_conf: float | None = None
        self._alert_on_streak = 0
        self._alert_off_streak = 0
        self._alert_active = False
        self._yolo_conf_history: deque[float] = deque(maxlen=self.analyzer.hybrid_confidence_window)
        self._tracker = self.analyzer._build_tracker()

    @staticmethod
    def _ema(prev: float | None, cur: float, alpha_up: float, alpha_down: float) -> float:
        if prev is None:
            return cur
        alpha = alpha_up if cur >= prev else alpha_down
        return prev * (1.0 - alpha) + cur * alpha

    def process_frame(self, frame: np.ndarray, timestamp_seconds: float | None = None) -> dict:
        roi = self.analyzer._extract_roi(frame, self.zone)
        if roi.size == 0:
            raise ValueError("The selected zone does not produce a valid frame region.")

        fh, fw = frame.shape[:2]
        roi_x = int(self.zone.x * fw)
        roi_y = int(self.zone.y * fh)

        recent_yolo_conf = self.analyzer._rolling_confidence(self._yolo_conf_history)
        detections, yolo_confidence = self.analyzer._detect_people_with_meta(
            roi,
            recent_yolo_confidence=recent_yolo_conf,
        )
        if self.analyzer.yolo_model is not None and self.analyzer.detector_mode in {"yolo", "hybrid"}:
            self._yolo_conf_history.append(yolo_confidence)
        occ_ratio = self.analyzer._estimate_occupancy_ratio(roi)
        raw_det_conf = self.analyzer._detection_confidence(detections)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        raw_turb = self.analyzer._estimate_motion_turbulence(self.previous_gray, gray)
        self.previous_gray = gray

        # Initialise heatmap on first frame.
        if self.heatmap is None:
            self.heatmap = CrowdHeatmap(
                frame_shape=(roi.shape[0], roi.shape[1]),
                decay=0.55,
                blur_ksize=31,
                alpha=0.40,
            )

        # Update heatmap first so estimate_count reflects the current frame.
        self.heatmap.update(roi, detections)

        # Derive count and density purely from the heatmap heat map.
        heatmap_max_density = float(os.getenv("HEATMAP_MAX_DENSITY", "4.0"))
        raw_count = self.heatmap.estimate_count(self.area_square_meters, heatmap_max_density)
        raw_density = raw_count / self.area_square_meters

        # Smooth core metrics for seamless UX without changing heatmap behavior.
        self._smoothed_count = self._ema(self._smoothed_count, float(raw_count), alpha_up=0.52, alpha_down=0.34)
        self._smoothed_density = self._ema(self._smoothed_density, raw_density, alpha_up=0.50, alpha_down=0.32)
        self._smoothed_occ = self._ema(self._smoothed_occ, occ_ratio, alpha_up=0.45, alpha_down=0.28)
        self._smoothed_turb = self._ema(self._smoothed_turb, raw_turb, alpha_up=0.50, alpha_down=0.30)
        self._smoothed_conf = self._ema(self._smoothed_conf, raw_det_conf, alpha_up=0.55, alpha_down=0.35)

        count = int(max(0, round(self._smoothed_count or 0.0)))
        density = float(self._smoothed_density or 0.0)
        occ_ratio = float(np.clip(self._smoothed_occ or 0.0, 0.0, 1.0))
        turb = max(0.0, float(self._smoothed_turb or 0.0))
        det_conf = float(np.clip(self._smoothed_conf or 0.0, 0.0, 1.0))

        # Hysteresis to avoid alert flicker on threshold edges.
        on_threshold = self.density_threshold
        off_threshold = max(self.density_threshold * 0.90, self.density_threshold - 0.18)
        if density >= on_threshold:
            self._alert_on_streak += 1
            self._alert_off_streak = 0
        elif density <= off_threshold:
            self._alert_off_streak += 1
            self._alert_on_streak = 0

        if self._alert_on_streak >= 2:
            self._alert_active = True
        elif self._alert_off_streak >= 2:
            self._alert_active = False

        alert = self._alert_active
        if alert:
            self.threshold_breach_frames += 1
        heatmap_roi = self.heatmap.get_overlay(roi)
        density_grid = self.heatmap.get_density_grid(rows=4, cols=6).tolist()

        _, buf = cv2.imencode('.jpg', heatmap_roi, [cv2.IMWRITE_JPEG_QUALITY, 72])
        heatmap_b64 = base64.b64encode(buf).decode('utf-8')

        frame_metric = FrameMetric(
            timestamp_seconds=round(timestamp_seconds or 0.0, 2),
            people_count=count,
            density_per_square_meter=round(density, 3),
            motion_turbulence=round(turb, 3),
            occupancy_ratio=round(occ_ratio, 3),
            detection_confidence=round(det_conf, 3),
            alert_triggered=alert,
        )

        self.frame_metrics.append(frame_metric)
        if len(self.frame_metrics) > self.history_size:
            self.frame_metrics = self.frame_metrics[-self.history_size:]

        risk = self.analyzer.risk_predictor.assess(
            self.frame_metrics,
            threshold_breach_frames=min(self.threshold_breach_frames, len(self.frame_metrics)),
        )

        detection_boxes = [
            DetectionBox(
                x=round((roi_x + bx) / fw, 4),
                y=round((roi_y + by) / fh, 4),
                width=round(bw / fw, 4),
                height=round(bh / fh, 4),
                confidence=round(conf, 3),
            )
            for bx, by, bw, bh, conf in detections
        ]

        result = RealtimeAnalysisResult(
            frame_metric=frame_metric,
            detections=detection_boxes,
            threshold_breach_frames=self.threshold_breach_frames,
            monitored_area_square_meters=round(self.area_square_meters, 3),
            density_threshold_per_square_meter=round(self.density_threshold, 3),
            zone=self.zone.as_dict(),
            frames_processed=len(self.frame_metrics),
            risk_assessment=risk,
        )
        out = result.model_dump()
        out['heatmap_b64'] = heatmap_b64
        out['density_grid'] = density_grid
        out['detector_backend'] = self.analyzer.detector_backend
        return out

    def process_frame_bytes(self, frame_bytes: bytes, timestamp_seconds: float | None = None) -> dict:
        arr = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Unable to decode the incoming frame.")
        return self.process_frame(frame=frame, timestamp_seconds=timestamp_seconds)
