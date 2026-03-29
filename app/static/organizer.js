const phaseLabel = document.getElementById("phaseLabel");
const eventLine = document.getElementById("eventLine");
const actionsList = document.getElementById("actionsList");
const reportsList = document.getElementById("reportsList");
const feedCount = document.getElementById("feedCount");
const tagline = document.getElementById("tagline");
const sectionPopup = document.getElementById("sectionPopup");
const popupClose = document.getElementById("popupClose");
const userAlertsBanner = document.getElementById("userAlertsBanner");
const seatMarkers = document.getElementById("seatMarkers");

const mvReports = document.getElementById("mv-reports");
const mvZones = document.getElementById("mv-zones");
const mvUser = document.getElementById("mv-user");
const mvMedical = document.getElementById("mv-medical");

const SEVERITY_COLORS = {
  low: "#facc15",
  medium: "#f97316",
  high: "#ef4444",
  critical: "#b91c1c"
};

// Label center positions for each zone (for seat pin placement)
const ZONE_CENTERS = {
  A: { x: 250, y: 63 },
  B: { x: 377, y: 155 },
  C: { x: 377, y: 225 },
  D: { x: 250, y: 317 },
  E: { x: 123, y: 225 },
  F: { x: 123, y: 155 }
};

const DEFAULT_ZONE_FILL = "#1e3a6e";

let lastUserReportCount = 0;
let currentData = null;
let selectedZone = null;

// Extract section letter (A-F) from seat string like "C14", "14C", "f2"
function seatToZone(seat) {
  const match = (seat || "").toUpperCase().match(/[A-F]/);
  return match ? match[0] : null;
}

async function callControl(action) {
  await fetch("/api/control", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action })
  });
  await refresh();
}

function reportCard(report) {
  const isUser = report.source === "attendee";
  const zone = seatToZone(report.seat) || "?";
  return `
    <li class="report-item${isUser ? " user-report" : ""}">
      <div class="report-head">
        <span>
          Seat <strong>${report.seat}</strong>
          <span class="sec-hint">Zone ${zone}</span>
          ${isUser ? "<span class='source-tag user-tag'>USER</span>" : ""}
        </span>
        <span class="badge ${report.severity}">${report.severity}</span>
      </div>
      <div class="report-msg">${report.message}</div>
      <div class="tiny">${report.timestamp}</div>
    </li>
  `;
}

function severityRank(v) {
  return { low: 1, medium: 2, high: 3, critical: 4 }[v] || 1;
}

// Build zone -> {severity, reports[]} from USER reports only
function buildZoneAlerts(userReports) {
  const zones = {};
  for (const r of userReports) {
    const z = seatToZone(r.seat);
    if (!z) continue;
    if (!zones[z]) zones[z] = { severity: r.severity, reports: [r] };
    else {
      zones[z].reports.push(r);
      if (severityRank(r.severity) > severityRank(zones[z].severity)) {
        zones[z].severity = r.severity;
      }
    }
  }
  return zones;
}

function renderMap(zoneAlerts) {
  // Reset all zones
  for (const letter of ["A","B","C","D","E","F"]) {
    const el = document.getElementById(`zone-${letter}`);
    if (!el) continue;
    const info = zoneAlerts[letter];
    el.setAttribute("fill", info ? SEVERITY_COLORS[info.severity] : DEFAULT_ZONE_FILL);
    el.setAttribute("opacity", info ? "0.9" : "0.85");
  }

  // Render seat pins for user reports
  seatMarkers.innerHTML = "";
  for (const [letter, info] of Object.entries(zoneAlerts)) {
    const center = ZONE_CENTERS[letter];
    if (!center) continue;
    const color = SEVERITY_COLORS[info.severity] || "#ef4444";
    // Show up to 2 seat labels; if more, show count badge
    const seats = info.reports.map(r => r.seat);

    seatMarkers.innerHTML += `
      <g class="seat-pin" data-zone="${letter}">
        <circle cx="${center.x}" cy="${center.y}" r="18" fill="${color}" stroke="white" stroke-width="2.5" filter="url(#pinShadow)"/>
        <text x="${center.x}" y="${center.y - 3}" text-anchor="middle" fill="white" font-size="9" font-weight="800" font-family="Barlow,sans-serif">${letter}</text>
        <text x="${center.x}" y="${center.y + 8}" text-anchor="middle" fill="white" font-size="7.5" font-family="Barlow,sans-serif">${seats.length > 1 ? seats.length + " reports" : seats[0]}</text>
      </g>
    `;
  }

  // Add drop-shadow filter if not already present
  const svg = document.getElementById("stadiumSVG");
  if (!svg.querySelector("defs")) {
    const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
    defs.innerHTML = `<filter id="pinShadow" x="-50%" y="-50%" width="200%" height="200%">
      <feDropShadow dx="0" dy="1" stdDeviation="3" flood-color="rgba(0,0,0,0.4)"/>
    </filter>`;
    svg.insertBefore(defs, svg.firstChild);
  }
}

function showZonePopup(letter, zoneAlerts) {
  selectedZone = letter;
  const info = zoneAlerts[letter];
  document.getElementById("popupHeader").textContent = `Zone ${letter}`;
  document.getElementById("popupRisk").innerHTML = info
    ? `<span class="badge ${info.severity}">${info.severity.toUpperCase()}</span> ${info.reports.length} active report${info.reports.length > 1 ? "s" : ""}`
    : `<span style="color:var(--muted)">No active reports</span>`;

  const reports = info ? info.reports : [];
  document.getElementById("popupReports").innerHTML = reports.length
    ? reports.map(r => `<li class="popup-user-report"><strong>${r.seat}</strong> — ${r.message}<div class="tiny">${r.timestamp}</div></li>`).join("")
    : "<li class='popup-empty'>No user reports for this zone</li>";

  sectionPopup.classList.remove("hidden");
}

async function refresh() {
  const response = await fetch("/api/state");
  const data = await response.json();
  currentData = data;

  phaseLabel.textContent = `${data.phase.label} (${data.phase.mode})`;
  eventLine.textContent = `${data.event.artist} | ${data.event.venue}, ${data.event.location}`;

  actionsList.innerHTML = data.ai_actions.map(a => `<li>${a}</li>`).join("");

  const userReports = data.reports.filter(r => r.source === "attendee");
  const userCount = userReports.length;

  if (userCount > lastUserReportCount && lastUserReportCount !== -1) {
    userAlertsBanner.classList.remove("hidden");
    setTimeout(() => userAlertsBanner.classList.add("hidden"), 4000);
  }
  lastUserReportCount = userCount;

  reportsList.innerHTML = data.reports.length
    ? data.reports.map(reportCard).join("")
    : "<li class='report-item'>No active incident reports</li>";
  feedCount.textContent = data.reports.length;

  const criticalZones = data.sections.filter(s => s.risk_band === "critical").length;
  const medicalAlerts = data.reports.filter(r => {
    const m = (r.message || "").toLowerCase();
    return m.includes("medical") || m.includes("fell") || m.includes("assistance");
  }).length;

  mvReports.textContent = data.system_metrics.active_reports;
  mvZones.textContent = criticalZones;
  mvUser.textContent = userCount;
  mvMedical.textContent = medicalAlerts;

  tagline.textContent = data.tagline || "";

  const zoneAlerts = buildZoneAlerts(userReports);
  renderMap(zoneAlerts);

  // Refresh popup if open
  if (selectedZone) showZonePopup(selectedZone, zoneAlerts);
}

// Zone click handlers on SVG paths
document.querySelectorAll(".zone").forEach(path => {
  path.addEventListener("click", () => {
    if (!currentData) return;
    const userReports = currentData.reports.filter(r => r.source === "attendee");
    showZonePopup(path.dataset.zone, buildZoneAlerts(userReports));
  });
});

popupClose.addEventListener("click", () => {
  sectionPopup.classList.add("hidden");
  selectedZone = null;
});

for (const btn of document.querySelectorAll("button[data-action]")) {
  btn.addEventListener("click", () => callControl(btn.dataset.action));
}

refresh();
setInterval(refresh, 1200);
