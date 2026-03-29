from __future__ import annotations

from pydantic import BaseModel, Field


class FrameMetric(BaseModel):
    timestamp_seconds: float
    people_count: int
    density_per_square_meter: float
    motion_turbulence: float
    occupancy_ratio: float
    detection_confidence: float = 0.0
    alert_triggered: bool


class DetectionBox(BaseModel):
    x: float
    y: float
    width: float
    height: float
    confidence: float


class RiskAssessment(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    level: str
    rationale: list[str]
    recommended_actions: list[str]


class VideoAnalysisResult(BaseModel):
    frame_sample_count: int
    avg_people_count: float
    peak_people_count: int
    avg_density_per_square_meter: float
    peak_density_per_square_meter: float
    density_threshold_per_square_meter: float
    threshold_breach_frames: int
    monitored_area_square_meters: float
    monitored_area_original_unit: str
    monitored_area_original_value: float
    zone: dict[str, float]
    frame_metrics: list[FrameMetric]
    risk_assessment: RiskAssessment


class RealtimeAnalysisResult(BaseModel):
    frame_metric: FrameMetric
    detections: list[DetectionBox]
    threshold_breach_frames: int
    monitored_area_square_meters: float
    density_threshold_per_square_meter: float
    zone: dict[str, float]
    frames_processed: int
    risk_assessment: RiskAssessment
