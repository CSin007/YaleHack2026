from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="CrowdShield MVP", version="0.1.0")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SECTIONS = ["A", "B", "C", "D", "E", "F"]

PHASES: List[Dict[str, Any]] = [
    {
        "id": "calm",
        "label": "Live Crowd Monitoring Active",
        "duration": 10,
        "base_risk": {"A": 8, "B": 12, "C": 10, "D": 9, "E": 11, "F": 8},
        "reports": [],
    },
    {
        "id": "detection",
        "label": "Anomaly detected in Section C",
        "duration": 12,
        "base_risk": {"A": 10, "B": 20, "C": 72, "D": 24, "E": 17, "F": 10},
        "reports": [
            {"seat": "C14", "message": "Sudden crowd surge", "severity": "high"},
            {"seat": "C18", "message": "Crowd pushing", "severity": "high"},
        ],
    },
    {
        "id": "individual_alert",
        "label": "Targeted attendee guidance active",
        "duration": 12,
        "base_risk": {"A": 12, "B": 26, "C": 79, "D": 30, "E": 24, "F": 11},
        "reports": [
            {"seat": "C12", "message": "Someone fell", "severity": "critical"},
            {"seat": "B12", "message": "Crowd pushing", "severity": "high"},
            {"seat": "C21", "message": "Panic movement", "severity": "high"},
        ],
    },
    {
        "id": "organizer",
        "label": "Seat-level incident map and response panel",
        "duration": 14,
        "base_risk": {"A": 14, "B": 32, "C": 83, "D": 38, "E": 29, "F": 14},
        "reports": [
            {"seat": "B12", "message": "Queue pressure at aisle", "severity": "high"},
            {"seat": "C10", "message": "Blocked movement", "severity": "high"},
            {"seat": "C12", "message": "Assistance needed", "severity": "critical"},
            {"seat": "D6", "message": "Unsafe compression", "severity": "high"},
        ],
    },
    {
        "id": "escalation",
        "label": "Escalation in progress",
        "duration": 14,
        "base_risk": {"A": 18, "B": 49, "C": 92, "D": 58, "E": 41, "F": 19},
        "reports": [
            {"seat": "B9", "message": "Heavy push wave", "severity": "critical"},
            {"seat": "C6", "message": "Multiple falls", "severity": "critical"},
            {"seat": "C22", "message": "People trapped near rail", "severity": "critical"},
            {"seat": "D8", "message": "Need medical support", "severity": "high"},
            {"seat": "D11", "message": "Evacuation guidance requested", "severity": "high"},
        ],
    },
    {
        "id": "resolution",
        "label": "Intervention prevents escalation",
        "duration": 9999,
        "base_risk": {"A": 9, "B": 20, "C": 33, "D": 24, "E": 16, "F": 9},
        "reports": [
            {"seat": "C12", "message": "Medical team arrived", "severity": "medium"},
            {"seat": "C14", "message": "Flow normalized", "severity": "low"},
        ],
    },
]


class ControlInput(BaseModel):
    action: str


class AttendeeAlertInput(BaseModel):
    seat: str
    message: str
    severity: str = "medium"


class DemoState:
    def __init__(self) -> None:
        self.lock = Lock()
        self.running = False
        self.start_time: datetime | None = None
        self.manual_phase_index = 0
        self.mode = "manual"
        self.user_reports: List[Dict[str, str]] = []

    def _phase_from_time(self) -> int:
        if not self.running or not self.start_time:
            return self.manual_phase_index

        elapsed = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        cursor = 0.0
        for idx, phase in enumerate(PHASES):
            cursor += phase["duration"]
            if elapsed < cursor:
                return idx
        return len(PHASES) - 1

    def get_phase_index(self) -> int:
        with self.lock:
            if self.mode == "auto":
                idx = self._phase_from_time()
                self.manual_phase_index = idx
                return idx
            return self.manual_phase_index

    def control(self, action: str) -> None:
        with self.lock:
            if action == "start":
                self.mode = "auto"
                self.running = True
                self.start_time = datetime.now(timezone.utc)
                self.manual_phase_index = 0
                return
            if action == "pause":
                self.mode = "manual"
                self.running = False
                self.start_time = None
                return
            if action == "next":
                self.mode = "manual"
                self.running = False
                self.start_time = None
                self.manual_phase_index = min(self.manual_phase_index + 1, len(PHASES) - 1)
                return
            if action == "back":
                self.mode = "manual"
                self.running = False
                self.start_time = None
                self.manual_phase_index = max(self.manual_phase_index - 1, 0)
                return
            if action == "escalate":
                self.mode = "manual"
                self.running = False
                self.start_time = None
                self.manual_phase_index = 4
                return
            if action == "resolve":
                self.mode = "manual"
                self.running = False
                self.start_time = None
                self.manual_phase_index = 5
                return
            if action == "reset":
                self.mode = "manual"
                self.running = False
                self.start_time = None
                self.manual_phase_index = 0
                self.user_reports = []
                return
            raise HTTPException(status_code=400, detail=f"Unknown action: {action}")

    def add_user_report(self, seat: str, message: str, severity: str) -> None:
        with self.lock:
            normalized = severity.lower().strip()
            if normalized not in {"low", "medium", "high", "critical"}:
                normalized = "medium"
            self.user_reports.insert(
                0,
                {
                    "seat": seat.upper().strip()[:8] or "UNK",
                    "message": message.strip()[:120] or "Attendee safety concern",
                    "severity": normalized,
                    "source": "attendee",
                    "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
                },
            )
            self.user_reports = self.user_reports[:12]


state = DemoState()
alerts_lock = Lock()
alerts: List[Dict[str, str]] = []


def risk_band(score: int) -> str:
    if score >= 75:
        return "critical"
    if score >= 45:
        return "high"
    if score >= 20:
        return "medium"
    return "low"


def ai_actions(phase_id: str, risks: Dict[str, int], reports: List[Dict[str, str]]) -> List[str]:
    hottest_section = max(risks, key=risks.get)
    peak = risks[hottest_section]
    actions = [
        f"Dispatch nearest security unit to Section {hottest_section}",
        f"Broadcast calm directional guidance away from Section {hottest_section}",
    ]
    if peak >= 70:
        actions.append("Close Gate 3 and redirect ingress to Exit A corridor")
    if phase_id in {"organizer", "escalation"}:
        actions.append("Open one-way aisle path from Rows 4-9 to reduce compression")
    if phase_id == "escalation":
        actions.append("Notify on-site medical team and pre-alert local emergency services")
    if reports:
        actions.append("Correlate seat reports to confirm spread radius before next announcement")
    return actions


def attendee_message(phase_id: str) -> Dict[str, str]:
    if phase_id in {"detection", "individual_alert", "organizer", "escalation"}:
        return {
            "title": "You are near a high-risk zone",
            "body": "Move calmly toward Exit A. Avoid Section C and follow staff guidance.",
            "direction": "-> Exit A",
            "tone": "alert",
        }
    if phase_id == "resolution":
        return {
            "title": "Risk levels are stabilizing",
            "body": "Continue walking to open space and wait for official updates.",
            "direction": "-> Keep distance from Section C",
            "tone": "recovery",
        }
    return {
        "title": "All clear",
        "body": "Live monitoring is active. Enjoy the show and keep aisles open.",
        "direction": "-> Nearest exit: A",
        "tone": "normal",
    }


def build_payload() -> Dict[str, Any]:
    idx = state.get_phase_index()
    phase = PHASES[idx]
    risks = phase["base_risk"]

    sections = []
    for section in SECTIONS:
        score = risks[section]
        sections.append(
            {
                "id": section,
                "risk_score": score,
                "risk_band": risk_band(score),
                "density_score": min(100, score + 7),
            }
        )

    reports = [
        {
            "seat": r["seat"],
            "message": r["message"],
            "severity": r["severity"],
            "source": "system",
            "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
        }
        for r in phase["reports"]
    ]
    reports = state.user_reports + reports

    return {
        "event": {
            "artist": "Taylor Swift",
            "venue": "Capital One Arena",
            "location": "Washington, DC",
            "show": "Eras Tour Safety Drill",
            "stadium_image": "/static/stadium-map.png",
        },
        "phase": {
            "index": idx,
            "id": phase["id"],
            "label": phase["label"],
            "mode": state.mode,
        },
        "sections": sections,
        "reports": reports,
        "attendee_alert": attendee_message(phase["id"]),
        "ai_actions": ai_actions(phase["id"], risks, reports),
        "system_metrics": {
            "anomaly_flag": phase["id"] not in {"calm", "resolution"},
            "max_risk": max(risks.values()),
            "active_reports": len(reports),
            "feed_status": "online",
        },
        "tagline": "Every person in the crowd becomes a sensor.",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/")
def root() -> FileResponse:
    return FileResponse("app/static/index.html")


@app.get("/organizer")
def organizer_page() -> FileResponse:
    return FileResponse("app/static/organizer.html")


@app.get("/attendee")
def attendee_page() -> FileResponse:
    return FileResponse("app/static/attendee.html")


@app.get("/api/state")
def get_state() -> Dict[str, Any]:
    return build_payload()


@app.post("/api/control")
def post_control(input_data: ControlInput) -> Dict[str, Any]:
    state.control(input_data.action.lower().strip())
    return build_payload()


@app.post("/api/attendee-alert")
def post_attendee_alert(input_data: AttendeeAlertInput) -> Dict[str, Any]:
    state.add_user_report(input_data.seat, input_data.message, input_data.severity)

    with alerts_lock:
        alerts.insert(
            0,
            {
                "seat": input_data.seat.upper().strip()[:8] or "UNK",
                "message": input_data.message.strip()[:120] or "Attendee safety concern",
                "severity": input_data.severity.lower().strip(),
                "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
            },
        )
        del alerts[32:]

    return build_payload()


@app.post("/alerts")
def send_alert(alert: AttendeeAlertInput):
    """Legacy endpoint for users to send alerts."""
    state.add_user_report(alert.seat, alert.message, alert.severity)

    with alerts_lock:
        alerts.insert(
            0,
            {
                "seat": alert.seat.upper().strip()[:8] or "UNK",
                "message": alert.message.strip()[:120] or "Attendee safety concern",
                "severity": alert.severity.lower().strip(),
                "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
            },
        )
        del alerts[32:]

    return {"status": "success"}


@app.get("/alerts")
def get_alerts():
    """Legacy alert feed endpoint kept for compatibility."""
    with alerts_lock:
        return {"alerts": list(alerts)}
