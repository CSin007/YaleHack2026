from __future__ import annotations

from statistics import mean
from app.models import FrameMetric, RiskAssessment


class StampedeRiskPredictor:

    # ── thresholds ────────────────────────────────────────────────────────
    DANGER_DENSITY   = 1.5
    CRITICAL_DENSITY = 3.0
    SEVERE_DENSITY   = 4.5

    OCC_WARN     = 0.20
    OCC_CRITICAL = 0.40

    TURB_WARN     = 0.40
    TURB_CRITICAL = 1.0

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

        densities   = [f.density_per_square_meter for f in frame_metrics]
        turbulences = [f.motion_turbulence         for f in frame_metrics]
        occupancies = [f.occupancy_ratio            for f in frame_metrics]

        avg_density     = mean(densities)
        peak_density    = max(densities)
        avg_turbulence  = mean(turbulences)
        peak_turbulence = max(turbulences)
        avg_occupancy   = mean(occupancies)
        peak_occupancy  = max(occupancies)
        density_accel   = self._density_acceleration(densities)
        sustained_ratio = threshold_breach_frames / max(len(frame_metrics), 1)

        # ── weighted components ───────────────────────────────────────
        density_component      = min(peak_density   / self.CRITICAL_DENSITY, 1.0) * 0.30
        occupancy_component    = min(peak_occupancy / self.OCC_CRITICAL,     1.0) * 0.28
        turbulence_component   = min(avg_turbulence / self.TURB_CRITICAL,    1.0) * 0.22
        sustained_component    = min(sustained_ratio,                         1.0) * 0.12
        acceleration_component = min(max(density_accel, 0.0) / 0.3,          1.0) * 0.08

        score = min(
            density_component + occupancy_component + turbulence_component
            + sustained_component + acceleration_component,
            1.0,
        )

        # ── occupancy override ────────────────────────────────────────
        if peak_occupancy >= self.OCC_CRITICAL:
            score = max(score, 0.82)
        elif peak_occupancy >= self.OCC_WARN:
            score = max(score, 0.50)

        # ── turbulence override ───────────────────────────────────────
        if peak_turbulence >= self.TURB_CRITICAL * 5:
            score = max(score, 0.88)
        elif peak_turbulence >= self.TURB_CRITICAL * 2:
            score = max(score, 0.72)
        elif peak_turbulence >= self.TURB_CRITICAL:
            score = max(score, 0.50)

        # ── combined override ─────────────────────────────────────────
        if peak_occupancy >= self.OCC_CRITICAL and peak_turbulence >= self.TURB_CRITICAL:
            score = max(score, 0.90)

        score = round(score, 3)

        # ── rationale + actions ───────────────────────────────────────
        rationale: list[str] = []
        actions:   list[str] = []

        if peak_density >= self.SEVERE_DENSITY:
            rationale.append(f"Peak density {peak_density:.1f} p/m² — life-threatening crush conditions.")
            actions.append("STOP ALL INFLOW IMMEDIATELY. Activate emergency egress plan.")
            actions.append("Alert emergency services and deploy all available staff.")
        elif peak_density >= self.CRITICAL_DENSITY:
            rationale.append(f"Peak density {peak_density:.1f} p/m² — above crush-risk threshold.")
            actions.append("Halt gate entry and redirect attendees away from this zone.")
            actions.append("Pre-stage medical teams at zone perimeter.")
        elif peak_density >= self.DANGER_DENSITY:
            rationale.append(f"Peak density {peak_density:.1f} p/m² — elevated crowd pressure.")
            actions.append("Throttle ingress. Redirect nearby attendees to lower-density areas.")

        if peak_occupancy >= self.OCC_CRITICAL:
            rationale.append(f"Frame visually saturated ({peak_occupancy:.0%} occupancy) — dense packing confirmed.")
            actions.append("Treat zone as critically congested regardless of area configuration.")
        elif peak_occupancy >= self.OCC_WARN:
            rationale.append(f"High visual occupancy ({peak_occupancy:.0%}) — zone filling rapidly.")
            actions.append("Verify exit clearance and barrier spacing in this zone.")

        if avg_turbulence >= self.TURB_CRITICAL:
            rationale.append(f"Extreme motion turbulence ({avg_turbulence:.2f}) — crowd instability or panic movement.")
            actions.append("Deploy staff immediately to stabilise crowd flow and check for incidents.")
        elif avg_turbulence >= self.TURB_WARN:
            rationale.append(f"Elevated turbulence ({avg_turbulence:.2f}) — unstable movement detected.")
            actions.append("Investigate for pushing, counterflow, or panic triggers.")

        if density_accel > 0.15:
            rationale.append(f"Density accelerating ({density_accel:.2f} p/m²/frame) — crowd pressure building.")
            actions.append("Dispatch staff to create flow separation and remove bottlenecks.")

        if sustained_ratio > 0.3:
            rationale.append(f"Threshold exceeded in {sustained_ratio:.0%} of observed frames.")
            actions.append("Keep zone under active monitoring.")

        if not rationale:
            rationale.append("Density and motion within safe operating limits.")
            actions.append("Continue passive monitoring.")

        level = self._level_from_score(score)
        if level == "critical":
            actions.append("Issue venue-wide response. Exits and corridors may be unsafe.")
        elif level == "high":
            actions.append("Pre-stage medical and security support near this zone.")

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
        deltas = [densities[i] - densities[i - 1] for i in range(1, len(densities))]
        pos    = [d for d in deltas if d > 0]
        return mean(pos) if pos else 0.0

    @staticmethod
    def _level_from_score(score: float) -> str:
        if score >= 0.65: return "critical"
        if score >= 0.40: return "high"
        if score >= 0.22: return "moderate"
        return "low"

    @staticmethod
    def _dedupe(items: list[str]) -> list[str]:
        seen: set[str] = set()
        out:  list[str] = []
        for item in items:
            if item not in seen:
                out.append(item)
                seen.add(item)
        return out