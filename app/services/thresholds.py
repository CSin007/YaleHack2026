from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CrowdDensityThresholdProfile:
    alert: float
    danger: float
    critical: float
    severe: float

    def validate(self) -> None:
        if min(self.alert, self.danger, self.critical, self.severe) <= 0.0:
            raise ValueError("Density thresholds must all be greater than zero.")
        if not (self.danger < self.critical < self.severe):
            raise ValueError("Density thresholds must satisfy: danger < critical < severe.")


def load_density_threshold_profile() -> CrowdDensityThresholdProfile:
    danger = float(os.getenv("CROWD_DENSITY_DANGER", "1.5"))
    critical = float(os.getenv("CROWD_DENSITY_CRITICAL", "3.0"))
    severe = float(os.getenv("CROWD_DENSITY_SEVERE", "4.5"))

    alert_raw = os.getenv("CROWD_DENSITY_ALERT")
    alert = danger if alert_raw in (None, "") else float(alert_raw)

    profile = CrowdDensityThresholdProfile(
        alert=alert,
        danger=danger,
        critical=critical,
        severe=severe,
    )
    profile.validate()
    return profile
