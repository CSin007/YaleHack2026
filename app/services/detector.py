from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.models import DetectionBox, FrameMetric, RealtimeAnalysisResult, VideoAnalysisResult
from app.services.risk import StampedeRiskPredictor
from app.services.heatmap import CrowdHeatmap


SQUARE_FEET_TO_SQUARE_METERS = 0.092903
DEFAULT_DANGER_DENSITY_PER_SQUARE_METER = 2.5


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


class CrowdAnalyzer:
    def __init__(self) -> None:
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.risk_predictor = StampedeRiskPredictor()

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
        area_sq_m       = self._to_square_meters(area_value, area_unit)
        density_thresh  = alert_density_threshold or DEFAULT_DANGER_DENSITY_PER_SQUARE_METER
        if density_thresh <= 0:
            raise ValueError("The alert density threshold must be greater than zero.")

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError("Unable to open the uploaded video.")

        frame_metrics: list[FrameMetric] = []
        prev_gray: np.ndarray | None = None
        sample_step = self._sample_step(capture)
        frame_index = threshold_breach = 0

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index % sample_step != 0:
                    frame_index += 1
                    continue

                roi        = self._extract_roi(frame, zone)
                detections = self._detect_people(roi)
                occ_ratio  = self._estimate_occupancy_ratio(roi)
                count      = self._estimate_people_count(detections, area_sq_m, occ_ratio)
                density    = count / area_sq_m
                gray       = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                turb       = self._estimate_motion_turbulence(prev_gray, gray)
                prev_gray  = gray

                alert = density >= density_thresh
                if alert:
                    threshold_breach += 1

                frame_metrics.append(FrameMetric(
                    timestamp_seconds=round(capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, 2),
                    people_count=count,
                    density_per_square_meter=round(density, 3),
                    motion_turbulence=round(turb, 3),
                    occupancy_ratio=round(occ_ratio, 3),
                    alert_triggered=alert,
                ))
                frame_index += 1
        finally:
            capture.release()

        if not frame_metrics:
            raise ValueError("No frames could be analyzed from the video.")

        avg_count   = sum(f.people_count              for f in frame_metrics) / len(frame_metrics)
        peak_count  = max(f.people_count              for f in frame_metrics)
        avg_density = sum(f.density_per_square_meter  for f in frame_metrics) / len(frame_metrics)
        peak_density= max(f.density_per_square_meter  for f in frame_metrics)
        risk        = self.risk_predictor.assess(frame_metrics, threshold_breach)

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
        area_sq_m      = self._to_square_meters(area_value, area_unit)
        density_thresh = alert_density_threshold or DEFAULT_DANGER_DENSITY_PER_SQUARE_METER
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

    # ── static helpers ────────────────────────────────────────────────────

    @staticmethod
    def _to_square_meters(area_value: float, area_unit: str) -> float:
        u = area_unit.strip().lower()
        if u == "square_meters": return area_value
        if u == "square_feet":   return area_value * SQUARE_FEET_TO_SQUARE_METERS
        raise ValueError("Area unit must be 'square_feet' or 'square_meters'.")

    @staticmethod
    def _sample_step(capture: cv2.VideoCapture) -> int:
        fps = capture.get(cv2.CAP_PROP_FPS)
        return max(int(fps // 2), 1) if fps > 0 else 5

    @staticmethod
    def _extract_roi(frame: np.ndarray, zone: NormalizedZone) -> np.ndarray:
        h, w = frame.shape[:2]
        x0, y0 = int(zone.x * w), int(zone.y * h)
        x1, y1 = int((zone.x + zone.width) * w), int((zone.y + zone.height) * h)
        return frame[y0:y1, x0:x1]

    def _detect_people(self, roi: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        rects, weights = self.hog.detectMultiScale(roi, winStride=(4,4), padding=(8,8), scale=1.05)
        if len(rects) == 0:
            return []
        weighted = [[*r, float(w)] for r, w in zip(rects, weights) if float(w) >= 0.35]
        if not weighted:
            return []
        boxes = np.array(weighted)
        kept  = self._non_max_suppression(boxes, 0.45)
        return [(int(b[0]), int(b[1]), int(b[2]), int(b[3]), float(b[4])) for b in kept]

    @staticmethod
    def _estimate_occupancy_ratio(roi: np.ndarray) -> float:
        gray    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray    = cv2.GaussianBlur(gray, (5, 5), 0)
        edges   = cv2.Canny(gray, 60, 140)
        er      = float(np.count_nonzero(edges)) / float(edges.size)
        lap     = cv2.Laplacian(gray, cv2.CV_32F)
        tr      = min(float(np.mean(np.abs(lap))) / 28.0, 1.0)
        fg      = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 31, 7)
        fr      = float(np.count_nonzero(fg)) / float(fg.size)
        return float(min(max(er*0.35 + tr*0.35 + fr*0.3, 0.0), 0.95))

    @staticmethod
    def _estimate_people_count(
        detections: list[tuple[int, int, int, int, float]],
        area_sq_m: float,
        occ_ratio: float,
    ) -> int:
        box_count    = len(detections)
        occ_count    = int(round(occ_ratio * 7.5 * area_sq_m))
        if occ_ratio > 0.2:
            return max(box_count, occ_count)
        return max(box_count, int(round(occ_count * 0.6)))

    @staticmethod
    def _non_max_suppression(boxes: np.ndarray, overlap_thresh: float) -> list[np.ndarray]:
        if len(boxes) == 0:
            return []
        boxes   = boxes.astype("float")
        picked  = []
        x1, y1  = boxes[:,0], boxes[:,1]
        x2, y2  = boxes[:,0]+boxes[:,2], boxes[:,1]+boxes[:,3]
        scores  = boxes[:,4]
        area    = (x2-x1+1)*(y2-y1+1)
        idxs    = np.argsort(scores)
        while len(idxs):
            last = len(idxs) - 1
            i    = idxs[last]
            picked.append(i)
            xx1  = np.maximum(x1[i], x1[idxs[:last]])
            yy1  = np.maximum(y1[i], y1[idxs[:last]])
            xx2  = np.minimum(x2[i], x2[idxs[:last]])
            yy2  = np.minimum(y2[i], y2[idxs[:last]])
            w    = np.maximum(0, xx2-xx1+1)
            h    = np.maximum(0, yy2-yy1+1)
            overlap = (w*h) / area[idxs[:last]]
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))
        return [boxes[i] for i in picked]

    @staticmethod
    def _estimate_motion_turbulence(
        prev_gray: np.ndarray | None, curr_gray: np.ndarray
    ) -> float:
        if prev_gray is None:
            return 0.0
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
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
        self.analyzer           = analyzer
        self.area_square_meters = area_square_meters
        self.area_unit          = area_unit
        self.area_value         = area_value
        self.zone               = zone
        self.density_threshold  = density_threshold
        self.history_size       = history_size
        self.previous_gray: np.ndarray | None = None
        self.frame_metrics: list[FrameMetric] = []
        self.threshold_breach_frames          = 0
        self.heatmap: CrowdHeatmap | None     = None

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp_seconds: float | None = None,
    ) -> dict:
        roi = self.analyzer._extract_roi(frame, self.zone)
        if roi.size == 0:
            raise ValueError("The selected zone does not produce a valid frame region.")

        fh, fw      = frame.shape[:2]
        roi_x       = int(self.zone.x * fw)
        roi_y       = int(self.zone.y * fh)

        detections  = self.analyzer._detect_people(roi)
        occ_ratio   = self.analyzer._estimate_occupancy_ratio(roi)
        count       = self.analyzer._estimate_people_count(detections, self.area_square_meters, occ_ratio)
        density     = count / self.area_square_meters
        gray        = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        turb        = self.analyzer._estimate_motion_turbulence(self.previous_gray, gray)
        self.previous_gray = gray

        alert = density >= self.density_threshold
        if alert:
            self.threshold_breach_frames += 1

        # ── heatmap ───────────────────────────────────────────────────
        if self.heatmap is None:
            self.heatmap = CrowdHeatmap(
                frame_shape=(roi.shape[0], roi.shape[1]),
                decay=0.82,
                blur_ksize=31,
                alpha=0.42,
            )
        self.heatmap.update(roi, detections)
        heatmap_roi  = self.heatmap.get_overlay(roi)
        density_grid = self.heatmap.get_density_grid(rows=4, cols=6).tolist()

        _, buf       = cv2.imencode('.jpg', heatmap_roi, [cv2.IMWRITE_JPEG_QUALITY, 72])
        heatmap_b64  = base64.b64encode(buf).decode('utf-8')

        fm = FrameMetric(
            timestamp_seconds=round(timestamp_seconds or 0.0, 2),
            people_count=count,
            density_per_square_meter=round(density, 3),
            motion_turbulence=round(turb, 3),
            occupancy_ratio=round(occ_ratio, 3),
            alert_triggered=alert,
        )
        self.frame_metrics.append(fm)
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
            frame_metric=fm,
            detections=detection_boxes,
            threshold_breach_frames=self.threshold_breach_frames,
            monitored_area_square_meters=round(self.area_square_meters, 3),
            density_threshold_per_square_meter=round(self.density_threshold, 3),
            zone=self.zone.as_dict(),
            frames_processed=len(self.frame_metrics),
            risk_assessment=risk,
        )
        out = result.model_dump()
        out['heatmap_b64']  = heatmap_b64
        out['density_grid'] = density_grid
        return out

    def process_frame_bytes(
        self,
        frame_bytes: bytes,
        timestamp_seconds: float | None = None,
    ) -> dict:
        arr   = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Unable to decode the incoming frame.")
        return self.process_frame(frame=frame, timestamp_seconds=timestamp_seconds)