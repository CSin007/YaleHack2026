from __future__ import annotations

import os
import re
import shutil
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.services.detector import CrowdAnalyzer, NormalizedZone
from app.services.elevenlabs_tts import ElevenLabsTTSClient
from app.services.lava_commander import LavaIncidentCommander
from app.services.realtime_clock import RealtimeFrameClock

# ── App + middleware ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Hotspot", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir    = BASE_DIR / "static"
templates_dir = BASE_DIR / "templates"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# ── CV analyzer singleton ─────────────────────────────────────────────────────
_analyzer: CrowdAnalyzer | None = None


def get_analyzer() -> CrowdAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = CrowdAnalyzer()
    return _analyzer


# ── Demo scenario phases ──────────────────────────────────────────────────────
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
            {"seat": "D6",  "message": "Unsafe compression", "severity": "high"},
        ],
    },
    {
        "id": "escalation",
        "label": "Escalation in progress",
        "duration": 14,
        "base_risk": {"A": 18, "B": 49, "C": 92, "D": 58, "E": 41, "F": 19},
        "reports": [
            {"seat": "B9",  "message": "Heavy push wave", "severity": "critical"},
            {"seat": "C6",  "message": "Multiple falls", "severity": "critical"},
            {"seat": "C22", "message": "People trapped near rail", "severity": "critical"},
            {"seat": "D8",  "message": "Need medical support", "severity": "high"},
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


# ── Pydantic request models ───────────────────────────────────────────────────
class ControlInput(BaseModel):
    action: str


class AttendeeAlertInput(BaseModel):
    seat: str
    message: str
    severity: str = "medium"

class EmergencyPARequest(BaseModel):
    languages: List[str] = Field(default_factory=lambda: ["en", "es"])
    include_audio: bool = True
    force: bool = False


# -- Demo state machine -- ────────────────────────────────────────────────────────
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
            elif action == "pause":
                self.mode = "manual"
                self.running = False
                self.start_time = None
            elif action == "next":
                self.mode = "manual"
                self.running = False
                self.start_time = None
                self.manual_phase_index = min(self.manual_phase_index + 1, len(PHASES) - 1)
            elif action == "back":
                self.mode = "manual"
                self.running = False
                self.start_time = None
                self.manual_phase_index = max(self.manual_phase_index - 1, 0)
            elif action == "escalate":
                self.mode = "manual"
                self.running = False
                self.start_time = None
                self.manual_phase_index = 4
            elif action == "resolve":
                self.mode = "manual"
                self.running = False
                self.start_time = None
                self.manual_phase_index = 5
            elif action == "reset":
                self.mode = "manual"
                self.running = False
                self.start_time = None
                self.manual_phase_index = 0
                self.user_reports = []
            else:
                raise HTTPException(status_code=400, detail=f"Unknown action: {action}")

    def add_user_report(self, seat: str, message: str, severity: str) -> None:
        with self.lock:
            normalized = severity.lower().strip()
            if normalized not in {"low", "medium", "high", "critical"}:
                normalized = "medium"
            self.user_reports.insert(0, {
                "seat":      seat.upper().strip()[:8] or "UNK",
                "message":   message.strip()[:120] or "Attendee safety concern",
                "severity":  normalized,
                "source":    "attendee",
                "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
            })
            self.user_reports = self.user_reports[:12]


state = DemoState()
alerts_lock = Lock()
alerts: List[Dict[str, str]] = []
incident_commander = LavaIncidentCommander(
    api_key=os.getenv("LAVA_API_KEY") or os.getenv("LAVA_API"),
    base_url=os.getenv("LAVA_BASE_URL", "https://api.lava.so/v1"),
    model=os.getenv("LAVA_MODEL", "openai/gpt-4o-mini"),
)

elevenlabs_client = ElevenLabsTTSClient(
    api_key=os.getenv("ELEVENLABS_API_KEY") or os.getenv("ELEVEN_LABS_API"),
    model_id=os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2"),
    voice_map={
        "default": os.getenv("ELEVENLABS_VOICE_DEFAULT", "EXAVITQu4vr4xnSDxMaL"),
        "en": os.getenv("ELEVENLABS_VOICE_EN", os.getenv("ELEVENLABS_VOICE_DEFAULT", "EXAVITQu4vr4xnSDxMaL")),
        "es": os.getenv("ELEVENLABS_VOICE_ES", os.getenv("ELEVENLABS_VOICE_DEFAULT", "EXAVITQu4vr4xnSDxMaL")),
        "hi": os.getenv("ELEVENLABS_VOICE_HI", os.getenv("ELEVENLABS_VOICE_DEFAULT", "EXAVITQu4vr4xnSDxMaL")),
        "fr": os.getenv("ELEVENLABS_VOICE_FR", os.getenv("ELEVENLABS_VOICE_DEFAULT", "EXAVITQu4vr4xnSDxMaL")),
        "ar": os.getenv("ELEVENLABS_VOICE_AR", os.getenv("ELEVENLABS_VOICE_DEFAULT", "EXAVITQu4vr4xnSDxMaL")),
    },
)

realtime_lock = Lock()
latest_realtime: Dict[str, Any] = {
    "available": False,
    "frame_metric": {},
    "risk_assessment": {},
    "threshold_breach_frames": 0,
    "updated_at": None,
}



# ── State helpers ─────────────────────────────────────────────────────────────
def risk_band(score: int) -> str:
    if score >= 75: return "critical"
    if score >= 45: return "high"
    if score >= 20: return "medium"
    return "low"


def seat_to_section(seat: str) -> str | None:
    match = re.search(r"[A-F]", (seat or "").upper())
    return match.group(0) if match else None


def section_report_counts(reports: List[Dict[str, str]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for report in reports:
        section = seat_to_section(report.get("seat", ""))
        if not section:
            continue
        counts[section] = counts.get(section, 0) + 1
    return counts


def _language_text(lang: str, section: str, exits: str) -> str:
    if lang == "es":
        return (
            f"Instruccion de emergencia: evacue el estadio ahora por la salida segura mas cercana ({exits}). "
            f"Evite la Seccion {section}. Camine con calma, no corra y siga al personal de seguridad."
        )
    if lang == "hi":
        return (
            f"Emergency nirdesh: kripya turant venue se niklein aur sabse nazdeeki surakshit exit ({exits}) ka upyog karein. "
            f"Section {section} se door rahen. Shant rahen, bhaagen nahin, staff ke nirdesh maanein."
        )
    if lang == "fr":
        return (
            f"Instruction d'urgence: evacuez le site maintenant par la sortie sure la plus proche ({exits}). "
            f"Evitez la section {section}. Avancez calmement, ne courez pas et suivez le personnel."
        )
    if lang == "ar":
        return (
            f"Emergency instruction: evacuate now via nearest safe exit ({exits}). "
            f"Avoid section {section}. Move calmly and follow staff directions immediately."
        )
    return (
        f"Emergency instruction: exit the venue now using the nearest safe marked exit ({exits}). "
        f"Avoid Section {section}. Move calmly, do not run, and follow staff directions immediately."
    )

def _compose_multilingual_pa(section: str, exits: str, languages: List[str]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    seen: set[str] = set()
    for raw in (languages or ["en", "es"]):
        lang = (raw or "en").strip().lower()
        if not lang or lang in seen:
            continue
        seen.add(lang)
        out.append({"language": lang, "text": _language_text(lang, section, exits)})
    if not out:
        out.append({"language": "en", "text": _language_text("en", section, exits)})
    return out

def _evaluate_emergency_pa_gate(
    risks: Dict[str, int],
    reports: List[Dict[str, str]],
    realtime_snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    max_risk_threshold = int(os.getenv("PA_GATE_MAX_RISK", "75"))
    section_reports_threshold = int(os.getenv("PA_GATE_SECTION_REPORTS", "5"))
    density_threshold = float(os.getenv("PA_GATE_DENSITY", "2.8"))
    risk_score_threshold = float(os.getenv("PA_GATE_RISK_SCORE", "0.72"))
    freshness_seconds = float(os.getenv("PA_GATE_FRESH_SECONDS", "20"))

    counts = section_report_counts(reports)
    max_section = "-"
    max_section_reports = 0
    if counts:
        max_section, max_section_reports = max(counts.items(), key=lambda item: item[1])

    fm = realtime_snapshot.get("frame_metric", {})
    risk = realtime_snapshot.get("risk_assessment", {})
    updated_raw = realtime_snapshot.get("updated_at")
    is_fresh = False
    if isinstance(updated_raw, str) and updated_raw:
        try:
            updated_dt = datetime.fromisoformat(updated_raw)
            age = (datetime.now(timezone.utc) - updated_dt).total_seconds()
            is_fresh = age <= freshness_seconds
        except Exception:
            is_fresh = False

    checks = {
        f"state_max_risk_ge_{max_risk_threshold}": (max(risks.values()) if risks else 0) >= max_risk_threshold,
        f"single_section_reports_ge_{section_reports_threshold}": max_section_reports >= section_reports_threshold,
        f"density_ge_{density_threshold}": float(fm.get("density_per_square_meter", 0.0)) >= density_threshold,
        f"risk_score_ge_{risk_score_threshold}": float(risk.get("score", 0.0)) >= risk_score_threshold,
        # f"realtime_fresh_le_{int(freshness_seconds)}s": is_fresh,
    }

    enabled = bool(realtime_snapshot.get("available", False)) and all(checks.values())
    return {
        "enabled": enabled,
        "checks": checks,
        "max_section": max_section,
        "max_section_reports": max_section_reports,
        "realtime_available": bool(realtime_snapshot.get("available", False)),
        "frame_metric": fm,
        "risk_assessment": risk,
        "thresholds": {
            "max_risk": max_risk_threshold,
            "section_reports": section_reports_threshold,
            "density": density_threshold,
            "risk_score": risk_score_threshold,
            "fresh_seconds": freshness_seconds,
        },
    }

def _status_only_actions(phase_id: str, max_risk: int, total_reports: int) -> List[Dict[str, Any]]:
    phase_text = phase_id.replace("_", " ").title()
    return [
        {
            "action": "No intervention required",
            "target": "All sections",
            "priority": "low",
            "confidence": 0.96,
            "rationale": f"Traffic is flowing normally. Concert status: {phase_text}. Risk={max_risk}, reports={total_reports}.",
        }
    ]


def _threshold_actions(section: str, report_count: int, max_risk: int) -> List[Dict[str, Any]]:
    if report_count >= 5:
        return [
            {
                "action": "Dispatch multi-team surge response",
                "target": f"Section {section}",
                "priority": "critical",
                "confidence": 0.95,
                "rationale": f"{report_count} reports from one section indicates severe crowd stress. Send 12 security, 2 medical teams, and route supervisors.",
            },
            {
                "action": "Throttle inflow and open alternate egress",
                "target": "Gate 3 -> Exit A/C corridors",
                "priority": "critical",
                "confidence": 0.90,
                "rationale": "Immediate flow redistribution is required to reduce compression.",
            },
        ]
    if report_count >= 3:
        return [
            {
                "action": "Dispatch reinforced staff group",
                "target": f"Section {section}",
                "priority": "high",
                "confidence": 0.91,
                "rationale": f"{report_count} reports from Section {section}. Send 8 security/stewards plus 1 medical team.",
            },
            {
                "action": "Issue targeted calm reroute message",
                "target": f"Aisles around Section {section}",
                "priority": "high",
                "confidence": 0.84,
                "rationale": "Reduce local bottlenecking and counter-flow movement.",
            },
        ]
    if report_count >= 2:
        return [
            {
                "action": "Dispatch response staff",
                "target": f"Section {section}",
                "priority": "high" if max_risk >= 70 else "medium",
                "confidence": 0.87,
                "rationale": f"{report_count} reports from one section crossed intervention threshold. Send 4 security/stewards.",
            }
        ]
    return []


def build_incident_actions(
    phase_id: str, risks: Dict[str, int], reports: List[Dict[str, str]]
) -> Dict[str, Any]:
    counts = section_report_counts(reports)
    max_risk = max(risks.values()) if risks else 0
    total_reports = len(reports)

    hot_section = None
    hot_report_count = 0
    if counts:
        hot_section, hot_report_count = max(counts.items(), key=lambda item: item[1])

    threshold_queue = _threshold_actions(hot_section, hot_report_count, max_risk) if hot_section else []

    if threshold_queue:
        result_provider = "policy-threshold"
        result_error = None
        queue = threshold_queue
    elif max_risk >= 30 or total_reports >= 2:
        result = incident_commander.suggest_actions(
            phase_id=phase_id,
            risks=risks,
            reports=reports,
        )
        result_provider = result.provider
        result_error = result.error
        queue = result.actions
    else:
        result_provider = "policy-status"
        result_error = None
        queue = _status_only_actions(phase_id, max_risk, total_reports)

    list_lines = [incident_commander.format_action_line(action) for action in queue]
    return {
        "provider": result_provider,
        "error": result_error,
        "queue": queue,
        "list": list_lines,
    }

def attendee_message(phase_id: str) -> Dict[str, str]:
    if phase_id in {"detection", "individual_alert", "organizer", "escalation"}:
        return {
            "title":     "You are near a high-risk zone",
            "body":      "Move calmly toward Exit A. Avoid Section C and follow staff guidance.",
            "direction": "-> Exit A",
            "tone":      "alert",
        }
    if phase_id == "resolution":
        return {
            "title":     "Risk levels are stabilizing",
            "body":      "Continue walking to open space and wait for official updates.",
            "direction": "-> Keep distance from Section C",
            "tone":      "recovery",
        }
    return {
        "title":     "All clear",
        "body":      "Live monitoring is active. Enjoy the show and keep aisles open.",
        "direction": "-> Nearest exit: A",
        "tone":      "normal",
    }


def build_payload() -> Dict[str, Any]:
    idx = state.get_phase_index()
    phase = PHASES[idx]
    risks = phase["base_risk"]

    sections = [
        {
            "id":            s,
            "risk_score":    risks[s],
            "risk_band":     risk_band(risks[s]),
            "density_score": min(100, risks[s] + 7),
        }
        for s in SECTIONS
    ]

    reports = [
        {
            "seat":      r["seat"],
            "message":   r["message"],
            "severity":  r["severity"],
            "source":    "system",
            "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
        }
        for r in phase["reports"]
    ]
    reports = state.user_reports + reports
    commander = build_incident_actions(phase["id"], risks, reports)

    return {
        "event": {
            "artist":        "Taylor Swift",
            "venue":         "Capital One Arena",
            "location":      "Washington, DC",
            "show":          "Eras Tour Safety Drill",
            "stadium_image": "/static/stadium-map.png",
        },
        "phase": {
            "index": idx,
            "id":    phase["id"],
            "label": phase["label"],
            "mode":  state.mode,
        },
        "sections":       sections,
        "reports":        reports,
        "attendee_alert": attendee_message(phase["id"]),
        "ai_actions":     commander["list"],
        "incident_commander": {
            "provider": commander["provider"],
            "error": commander["error"],
            "queue": commander["queue"],
        },
        "system_metrics": {
            "anomaly_flag":   phase["id"] not in {"calm", "resolution"},
            "max_risk":       max(risks.values()),
            "active_reports": len(reports),
            "feed_status":    "online",
        },
        "tagline":      "Every person in the crowd becomes a sensor.",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ════════════════════════════════════════════════════════════════════════════
# Page routes
# ════════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=FileResponse)
def root() -> FileResponse:
    """Landing page — role selection."""
    return FileResponse(str(static_dir / "index.html"))


@app.get("/organizer", response_class=FileResponse)
def organizer_page() -> FileResponse:
    """Unified organizer command center (stadium map + live analysis)."""
    return FileResponse(str(static_dir / "organizer.html"))


@app.get("/attendee", response_class=FileResponse)
def attendee_page() -> FileResponse:
    """Attendee safety view."""
    return FileResponse(str(static_dir / "attendee.html"))


@app.get("/analyzer", response_class=FileResponse)
def analyzer_page() -> FileResponse:
    """Standalone live analysis tool (Crowd Control AI)."""
    return FileResponse(str(templates_dir / "index.html"))


# ════════════════════════════════════════════════════════════════════════════
# Health + utility
# ════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.websocket("/ws/ping")
async def ws_ping(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            await websocket.receive_text()
            await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        return


# ════════════════════════════════════════════════════════════════════════════
# Demo state API
# ════════════════════════════════════════════════════════════════════════════

@app.get("/api/state")
def get_state() -> Dict[str, Any]:
    return build_payload()


@app.get("/api/incident-commander")
def get_incident_commander() -> Dict[str, Any]:
    payload = build_payload()
    return {
        "phase": payload["phase"],
        "incident_commander": payload["incident_commander"],
        "ai_actions": payload["ai_actions"],
    }


@app.post("/api/emergency-pa")
def generate_emergency_pa(input_data: EmergencyPARequest) -> Dict[str, Any]:
    idx = state.get_phase_index()
    phase = PHASES[idx]
    risks = phase["base_risk"]

    base_reports = [
        {
            "seat": r["seat"],
            "message": r["message"],
            "severity": r["severity"],
            "source": "system",
            "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
        }
        for r in phase["reports"]
    ]
    reports = state.user_reports + base_reports

    with realtime_lock:
        snapshot = dict(latest_realtime)

    gate = _evaluate_emergency_pa_gate(risks=risks, reports=reports, realtime_snapshot=snapshot)
    if input_data.force:
        gate["enabled"] = True
        gate["forced"] = True
    else:
        gate["forced"] = False

    if not gate["enabled"]:
        return {
            "enabled": False,
            "status": "Emergency PA not activated. Indicators have not all crossed critical evacuation thresholds.",
            "gate": gate,
            "messages": [
                {
                    "language": "en",
                    "text": "All indicators are below the emergency-evacuation threshold. Continue monitoring and normal operations.",
                }
            ],
            "audio": [],
        }

    max_section = gate.get("max_section") or "C"
    messages = _compose_multilingual_pa(
        section=max_section,
        exits="A/B/C",
        languages=input_data.languages,
    )

    audio: List[Dict[str, Any]] = []
    if input_data.include_audio:
        for message in messages:
            tts_result = elevenlabs_client.synthesize(message["text"], message["language"])
            audio.append(
                {
                    "language": message["language"],
                    "ok": tts_result.ok,
                    "mime_type": tts_result.mime_type,
                    "audio_b64": tts_result.audio_b64 if tts_result.ok else "",
                    "error": tts_result.error,
                }
            )

    return {
        "enabled": True,
        "status": "Emergency PA activated. Evacuation-level multilingual announcement generated.",
        "gate": gate,
        "messages": messages,
        "audio": audio,
    }


@app.post("/api/control")
def post_control(input_data: ControlInput) -> Dict[str, Any]:
    state.control(input_data.action.lower().strip())
    return build_payload()


@app.post("/api/attendee-alert")
def post_attendee_alert(input_data: AttendeeAlertInput) -> Dict[str, Any]:
    state.add_user_report(input_data.seat, input_data.message, input_data.severity)
    with alerts_lock:
        alerts.insert(0, {
            "seat":      input_data.seat.upper().strip()[:8] or "UNK",
            "message":   input_data.message.strip()[:120] or "Attendee safety concern",
            "severity":  input_data.severity.lower().strip(),
            "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
        })
        del alerts[32:]
    return build_payload()


# ── Legacy alert endpoints (kept for compatibility) ───────────────────────────
@app.post("/alerts")
def send_alert(alert: AttendeeAlertInput) -> Dict[str, str]:
    state.add_user_report(alert.seat, alert.message, alert.severity)
    with alerts_lock:
        alerts.insert(0, {
            "seat":      alert.seat.upper().strip()[:8] or "UNK",
            "message":   alert.message.strip()[:120] or "Attendee safety concern",
            "severity":  alert.severity.lower().strip(),
            "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
        })
        del alerts[32:]
    return {"status": "success"}


@app.get("/alerts")
def get_alerts() -> Dict[str, Any]:
    with alerts_lock:
        return {"alerts": list(alerts)}


# ════════════════════════════════════════════════════════════════════════════
# CV / OpenCV analysis API
# ════════════════════════════════════════════════════════════════════════════

@app.post("/api/analyze-video")
async def analyze_video(
    video: UploadFile = File(...),
    area_value: float = Form(...),
    area_unit: str = Form(...),
    alert_density_threshold: float | None = Form(default=None),
    zone_x: float = Form(default=0.0),
    zone_y: float = Form(default=0.0),
    zone_width: float = Form(default=1.0),
    zone_height: float = Form(default=1.0),
) -> JSONResponse:
    """Upload a video file and run full crowd analysis. Returns VideoAnalysisResult."""
    suffix    = Path(video.filename or "upload.mp4").suffix or ".mp4"
    temp_dir  = Path(tempfile.mkdtemp(prefix="crowdshield-"))
    temp_path = temp_dir / f"upload{suffix}"
    try:
        with temp_path.open("wb") as buf:
            shutil.copyfileobj(video.file, buf)
        zone   = NormalizedZone(x=zone_x, y=zone_y, width=zone_width, height=zone_height)
        result = get_analyzer().analyze_video(
            video_path=temp_path,
            area_value=area_value,
            area_unit=area_unit,
            zone=zone,
            alert_density_threshold=alert_density_threshold,
        )
        return JSONResponse(result.model_dump())
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ════════════════════════════════════════════════════════════════════════════
# WebSocket — real-time frame analysis
# ════════════════════════════════════════════════════════════════════════════

@app.websocket("/ws/realtime-analysis")
async def realtime_analysis(websocket: WebSocket) -> None:
    """
    Protocol:
      1. Client connects and sends JSON config (area_value, area_unit,
         alert_density_threshold, zone_x/y/width/height).
      2. Server replies {"type": "session_ready"}.
      3. Client streams raw JPEG frames as binary messages.
      4. Server replies {"type": "analysis", "payload": RealtimeAnalysisResult}
         for each frame.
    """
    await websocket.accept()
    try:
        init_payload = await websocket.receive_json()

        session = get_analyzer().create_realtime_session(
            area_value=float(init_payload["area_value"]),
            area_unit=str(init_payload["area_unit"]),
            alert_density_threshold=(
                float(init_payload["alert_density_threshold"])
                if init_payload.get("alert_density_threshold") not in (None, "")
                else None
            ),
            zone=NormalizedZone(
                x=float(init_payload.get("zone_x", 0.0)),
                y=float(init_payload.get("zone_y", 0.0)),
                width=float(init_payload.get("zone_width", 1.0)),
                height=float(init_payload.get("zone_height", 1.0)),
            ),
        )
        await websocket.send_json({"type": "session_ready"})

        frame_clock = RealtimeFrameClock()
        peak_density = 0.0

        while True:
            frame_bytes = await websocket.receive_bytes()
            frame_timestamp = frame_clock.next_timestamp_seconds()

            result = session.process_frame_bytes(
                frame_bytes=frame_bytes,
                timestamp_seconds=frame_timestamp,
            )

            payload = result if isinstance(result, dict) else result.model_dump()

            fm = payload["frame_metric"]
            if fm["density_per_square_meter"] > peak_density:
                peak_density = fm["density_per_square_meter"]

            with realtime_lock:
                latest_realtime.update(
                    {
                        "available": True,
                        "frame_metric": fm,
                        "risk_assessment": payload.get("risk_assessment", {}),
                        "threshold_breach_frames": int(payload.get("threshold_breach_frames", 0)),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                )

            await websocket.send_json({"type": "analysis", "payload": payload})

    except WebSocketDisconnect:
        return
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"[WS ERROR]\n{tb}")
        try:
            await websocket.send_json({"type": "error", "message": str(exc), "detail": tb})
            await websocket.close(code=1011)
        except Exception:
            pass





























