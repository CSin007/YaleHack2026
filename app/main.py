from __future__ import annotations

import asyncio
import shutil
import tempfile
import traceback
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.services.detector import CrowdAnalyzer, NormalizedZone

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Crowd Control AI", version="0.2.0")

static_dir = BASE_DIR / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

_analyzer: CrowdAnalyzer | None = None


def get_analyzer() -> CrowdAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = CrowdAnalyzer()
    return _analyzer


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


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


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
    suffix    = Path(video.filename or "upload.mp4").suffix or ".mp4"
    temp_dir  = Path(tempfile.mkdtemp(prefix="crowd-control-"))
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


@app.websocket("/ws/realtime-analysis")
async def realtime_analysis(websocket: WebSocket) -> None:
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

        frame_index  = 0
        peak_density = 0.0

        while True:
            frame_bytes = await websocket.receive_bytes()

            result = session.process_frame_bytes(
                frame_bytes=frame_bytes,
                timestamp_seconds=frame_index * 0.4,
            )
            frame_index += 1

            payload = result if isinstance(result, dict) else result.model_dump()

            fm = payload["frame_metric"]
            if fm["density_per_square_meter"] > peak_density:
                peak_density = fm["density_per_square_meter"]

            await websocket.send_json({
                "type":    "analysis",
                "payload": payload,
            })

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