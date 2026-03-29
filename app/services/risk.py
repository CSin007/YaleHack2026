from __future__ import annotations

from statistics import mean

from app.models import FrameMetric, RiskAssessment
from app.services.thresholds import CrowdDensityThresholdProfile, load_density_threshold_profile


class StampedeRiskPredictor:
    OCC_WARN = 0.20
    OCC_CRITICAL = 0.40

    TURB_WARN = 0.40
    TURB_CRITICAL = 1.0

    def __init__(self, density_profile: CrowdDensityThresholdProfile | None = None) -> None:
        self.density_profile = density_profile or load_density_threshold_profile()
        self.alert_density = self.density_profile.alert
        self.danger_density = self.density_profile.danger
        self.critical_density = self.density_profile.critical
        self.severe_density = self.density_profile.severe

    def assess(
        self,
        frame_metrics: list[FrameMetric],
        threshold_breach_frames: int,
    ) -> RiskAssessment:
        if not frame_metrics:
            return RiskAssessment(
                score=0.0,
                level="low",
                rationale=["No analyzable frames."],
                recommended_actions=["Check video source and zone config."],
            )

        densities = [metric.density_per_square_meter for metric in frame_metrics]
        turbulences = [metric.motion_turbulence for metric in frame_metrics]
        occupancies = [metric.occupancy_ratio for metric in frame_metrics]
        yolo_confidences = [
            max(0.0, min(1.0, getattr(metric, "detection_confidence", 0.0))) for metric in frame_metrics
        ]

        avg_density = mean(densities)
        peak_density = max(densities)
        avg_turbulence = mean(turbulences)
        peak_turbulence = max(turbulences)
        avg_occupancy = mean(occupancies)
        peak_occupancy = max(occupancies)

        avg_yolo_conf = mean(yolo_confidences)
        sustained_ratio = threshold_breach_frames / max(len(frame_metrics), 1)
        density_accel = self._density_acceleration(densities)

        # YOLO confidence gates how much we trust detector-driven pressure metrics.
        # Low confidence still allows turbulence/occupancy to contribute, but at lower weight.
        yolo_support = min(avg_yolo_conf / 0.75, 1.0)

        density_component = min(peak_density / self.critical_density, 1.0) * (0.18 + 0.18 * yolo_support)
        occupancy_component = min(peak_occupancy / self.OCC_CRITICAL, 1.0) * 0.20
        turbulence_component = min(avg_turbulence / self.TURB_CRITICAL, 1.0) * 0.20
        sustained_component = min(sustained_ratio, 1.0) * 0.10
        acceleration_component = min(max(density_accel, 0.0) / 0.35, 1.0) * 0.08
        detector_component = yolo_support * 0.16

        score = min(
            density_component
            + occupancy_component
            + turbulence_component
            + sustained_component
            + acceleration_component
            + detector_component,
            1.0,
        )

        # Extreme-event uplift only when detector confidence is reasonably strong.
        if peak_density >= self.severe_density:
            score = min(1.0, score + 0.15)
        if peak_occupancy >= self.OCC_CRITICAL or peak_turbulence >= self.TURB_CRITICAL:
            score = min(1.0, score + 0.12)

        score = round(score, 3)

        rationale: list[str] = []
        actions: list[str] = []

        rationale.append(f"YOLO confidence support: {avg_yolo_conf:.2f} (higher means stronger detector certainty).")

        if peak_density >= self.severe_density:
            rationale.append(f"Peak density {peak_density:.1f} p/m2 indicates severe crush pressure.")
            actions.append("Stop inflow and trigger emergency egress protocol.")
        elif peak_density >= self.critical_density:
            rationale.append(f"Peak density {peak_density:.1f} p/m2 exceeds critical threshold.")
            actions.append("Halt gate entry and redirect nearby crowd flow.")
        elif peak_density >= self.danger_density:
            rationale.append(f"Peak density {peak_density:.1f} p/m2 shows elevated pressure.")
            actions.append("Throttle ingress and monitor aisle bottlenecks.")

        if peak_occupancy >= self.OCC_CRITICAL:
            rationale.append(f"Visual occupancy reached {peak_occupancy:.0%}, indicating packed frame regions.")
            actions.append("Dispatch floor staff to expand movement corridors.")
        elif peak_occupancy >= self.OCC_WARN:
            rationale.append(f"Occupancy at {peak_occupancy:.0%} suggests crowd buildup.")

        if avg_turbulence >= self.TURB_CRITICAL:
            rationale.append(f"High turbulence ({avg_turbulence:.2f}) suggests unstable movement behavior.")
            actions.append("Investigate push waves and counterflow behavior.")
        elif avg_turbulence >= self.TURB_WARN:
            rationale.append(f"Moderate turbulence ({avg_turbulence:.2f}) suggests movement instability.")

        if density_accel > 0.15:
            rationale.append(f"Density acceleration ({density_accel:.2f} p/m2/frame) indicates fast escalation.")
            actions.append("Pre-position medical and security near high-pressure sectors.")

        if sustained_ratio > 0.30:
            rationale.append(f"Density threshold breached in {sustained_ratio:.0%} of recent frames.")

        if avg_yolo_conf < 0.35:
            rationale.append("Detector confidence is low; risk may be conservative until visual certainty improves.")
            actions.append("Improve camera angle/lighting or switch to clearer feed for stronger confidence.")

        if not actions:
            actions.append("Continue passive monitoring and keep exits clear.")

        level = self._level_from_score(score)
        if level == "critical":
            actions.append("Activate venue-wide crowd response and emergency coordination.")
        elif level == "high":
            actions.append("Keep security and medical teams on immediate standby.")

        return RiskAssessment(
            score=score,
            level=level,
            rationale=self._dedupe(rationale),
            recommended_actions=self._dedupe(actions),
        )

    @staticmethod
    def _density_acceleration(densities: list[float]) -> float:
        if len(densities) < 3:
            return 0.0
        deltas = [densities[index] - densities[index - 1] for index in range(1, len(densities))]
        positive = [delta for delta in deltas if delta > 0]
        return mean(positive) if positive else 0.0

    @staticmethod
    def _level_from_score(score: float) -> str:
        if score >= 0.40:
            return "critical"
        if score >= 0.22:
            return "high"
        if score >= 0.10:
            return "moderate"
        return "low"

    @staticmethod
    def _dedupe(items: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for item in items:
            if item not in seen:
                out.append(item)
                seen.add(item)
        return out
