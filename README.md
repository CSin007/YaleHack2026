diff --git a/c:\Users\renik\Downloads\crowd-control\README.md b/c:\Users\renik\Downloads\crowd-control\README.md
new file mode 100644
--- /dev/null
+++ b/c:\Users\renik\Downloads\crowd-control\README.md
@@ -0,0 +1,159 @@
+# Crowd Control AI Prototype
+
+This project is a starter web app for venue crowd monitoring with two perspectives:
+
+- Organizer view: surveillance-camera-driven monitoring for crowd density, gate pressure, and incident escalation.
+- Attendee view: user-linked reporting and device-assisted safety telemetry tied to section, seat, and venue location.
+
+The current codebase focuses on the computer-vision path: upload a venue video, estimate the number of people in a monitored zone, measure local crowd density, and raise alerts when a potential stampede pattern is detected.
+
+## System Design Architecture
+
+### 1. Client Layer
+
+- Organizer dashboard web app
+  - Live venue map
+  - Camera feed health
+  - Alerts: "Pause entry at Gate C", "Section 112 congestion", "North concourse at capacity"
+  - Incident review timeline
+- Attendee mobile/web app
+  - Account linked to ticket metadata: section, row, seat, entrance gate
+  - Camera access and optional location sharing
+  - One-tap incident reporting: fight, fainting, abuse, crowd crush, blocked exit
+  - Safety notifications and rerouting guidance
+
+### 2. Edge and Ingestion Layer
+
+- Venue CCTV ingestion service
+  - RTSP/WebRTC stream intake
+  - Frame buffering
+  - Camera calibration metadata per zone
+- User telemetry ingestion
+  - Report events
+  - GPS/BLE/UWB location pings where permitted
+  - Mobile video/image upload for incident verification
+- Ticketing and identity connector
+  - User-seat mapping
+  - Entry/exit logs by gate
+
+### 3. AI and Analytics Layer
+
+- Crowd vision service
+  - Person detection
+  - Zone-based counting
+  - Density estimation
+  - Movement direction and motion instability
+- Incident fusion agent
+  - Combines CCTV detections, user reports, venue capacity data, and gate activity
+  - Produces recommended actions for operators
+- Risk prediction service
+  - Predicts crush or stampede risk using:
+    - people per square meter
+    - density acceleration
+    - directional conflict
+    - motion turbulence
+    - repeated distress/fight/fainting reports
+- Heatmap generator
+  - Density map by zone
+  - Hazard map for blocked exits, aggressive movement, pressure at gates
+
+### 4. Decision and Action Layer
+
+- Alert orchestration
+  - Stop entry at specific gates
+  - Dispatch medical/security staff
+  - Push attendee rerouting messages
+  - Trigger public-address announcements
+- Recommendation engine
+  - Move crowd toward lower-density corridors
+  - Open overflow sections or alternate egress
+  - Slow ingress when venue reaches safe operating density
+
+### 5. Data Layer
+
+- Time-series store for sensor and density events
+- Relational store for users, seats, gates, incidents, and venue topology
+- Object store for uploaded video clips and snapshots
+- Feature store for risk model inputs
+
+### 6. Security and Governance
+
+- Role-based access for organizers, responders, and venue admins
+- Consent management for attendee camera/location features
+- Data retention and audit logs
+- Privacy filtering for attendee-submitted media
+
+## Current Prototype
+
+This repo currently includes:
+
+- FastAPI backend
+- Video upload endpoint
+- OpenCV-based people detection
+- Zone density alerting in square feet or square meters
+- Heuristic "potential stampede" risk scoring
+- Small browser UI for testing
+
+## How Stampede Risk Is Estimated
+
+This prototype uses a rules-plus-score approach:
+
+- Dangerous density begins around 4 people per square meter.
+- Severe crush risk rises near 5.5 to 6+ people per square meter.
+- Rapid increases in density matter even before the peak is reached.
+- Erratic optical flow can indicate unstable movement and pressure waves.
+
+The risk score combines:
+
+- average density
+- peak density
+- density growth rate
+- movement turbulence
+- sustained threshold breaches
+
+This is useful for prototyping, but it is not a certified safety system. Real deployment should include calibrated cameras, venue-specific zone geometry, stronger detection models, and human-in-the-loop operating procedures.
+
+## Run Locally
+
+1. Create a virtual environment and activate it.
+2. Install dependencies:
+
+```bash
+pip install -r requirements.txt
+```
+
+3. Start the app:
+
+```bash
+uvicorn app.main:app --reload
+```
+
+4. Open `http://127.0.0.1:8000`
+
+## API
+
+### `POST /api/analyze-video`
+
+Multipart form fields:
+
+- `video`: video file
+- `area_value`: monitored area size
+- `area_unit`: `square_feet` or `square_meters`
+- `alert_density_threshold`: optional density threshold override
+- `zone_x`, `zone_y`, `zone_width`, `zone_height`: optional normalized ROI values from `0.0` to `1.0`
+
+Response includes:
+
+- estimated people counts
+- density statistics
+- threshold alerts
+- potential stampede risk level
+- recommended actions
+
+## Next Steps
+
+- Replace HOG detector with YOLOv8/RT-DETR for higher accuracy
+- Add multi-camera fusion
+- Add seat-section heatmaps from user events
+- Add WebSocket live stream analysis
+- Add incident reporting and organizer action workflows
