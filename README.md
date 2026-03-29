# CrowdShield MVP

FastAPI-based hackathon MVP for real-time crowd safety storytelling with:
- Role-based entry (`/` -> organizer vs attendee)
- Organizer dashboard (stadium sections + seat reports + AI actions + controls)
- Attendee alert UI (calm evacuation guidance, synced to organizer state)
- Demo controls for your live pitch flow (`start`, `escalate`, `resolve`)

## Demo Story Built-In
Event is preloaded as:
- Artist: Taylor Swift
- Venue: Capital One Arena
- Location: Washington, DC

The flow follows these phases:
1. Calm monitoring
2. Detection
3. Individual alert
4. Organizer response
5. Escalation
6. Resolution

## Run Locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open:
- http://127.0.0.1:8000 (role selection)
- http://127.0.0.1:8000/organizer
- http://127.0.0.1:8000/attendee

## Controls During Demo
- `Start Auto Flow`: runs full sequence
- `Next Scene`: advances one phase
- `Back`: returns one phase
- `Escalate`: jumps to high-risk moment
- `Resolve`: jumps to stabilization ending
- `Reset`: back to calm baseline

## API
- `GET /api/state`
- `POST /api/control` with JSON body:

```json
{ "action": "start" }
```

Supported actions: `start`, `pause`, `next`, `back`, `escalate`, `resolve`, `reset`.
- `POST /api/attendee-alert` with JSON body:

```json
{ "seat": "C14", "message": "Crowd pushing near aisle", "severity": "high" }
```

## Quick Customization
- Phase timings, risk scores, and seat reports are in `app/main.py` (`PHASES`).
- Stadium image path is in `app/main.py` (`event.stadium_image`).
- Put your own map image at `app/static/stadium-map.png`.
- If missing, the UI falls back to `app/static/stadium-map-placeholder.svg`.
- UI styling is in `app/static/style.css`.
- Organizer logic is in `app/static/organizer.js`.
- Attendee logic is in `app/static/attendee.js`.
