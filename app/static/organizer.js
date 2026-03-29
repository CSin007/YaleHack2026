// ── Constants ─────────────────────────────────────────────────────────────────
const CIRC = 339.3; // 2π × 54

const SEVERITY_COLORS = {
  low: "#facc15", medium: "#f97316", high: "#ef4444", critical: "#b91c1c"
};

const ZONE_CENTERS = {
  A: { x: 250, y: 63 },  B: { x: 377, y: 155 },
  C: { x: 377, y: 225 }, D: { x: 250, y: 317 },
  E: { x: 123, y: 225 }, F: { x: 123, y: 155 }
};

const DEFAULT_ZONE_FILL = "#1e3a6e";

// Camera zone names
const ZONE_NAMES  = ['NW','N','NE','SW','S','SE'];
const ZONE_LABELS = { NW:'West wing', N:'North centre', NE:'East wing', SW:'SW corner', S:'South centre', SE:'SE corner' };
const ZONE_THRESH = { low: 0.02, moderate: 0.06, high: 0.14, critical: 0.25 };
const ZONE_DANGER_PCT = 70;
const ZONE_NEUTRAL_COLOR = "#64748b";

function zonePercentFromHeat(heat) {
  return Math.min((Math.max(heat, 0) / ZONE_THRESH.critical) * 100, 100);
}

function criticalRedShade(pct, alpha = 1) {
  const t = Math.min(Math.max((pct - ZONE_DANGER_PCT) / (100 - ZONE_DANGER_PCT), 0), 1);
  const r = Math.round(205 + (50 * t));
  const g = Math.round(36 - (22 * t));
  const b = Math.round(36 - (22 * t));
  return `rgba(${r},${g},${b},${alpha})`;
}

// ── Live stream state ─────────────────────────────────────────────────────────
let ws = null, videoEl = null, sendTimer = null, rafId = null;
let offscreen = null, offCtx = null;
let lastResult = null;
let lastCount = 0, peakDensity = 0, framesSent = 0, sentAt = 0;

let zoneState = {};
ZONE_NAMES.forEach(n => { zoneState[n] = { heat: 0, prev: 0, state: 'low', smoothed: 0 }; });

// ── Organizer state ───────────────────────────────────────────────────────────
let lastUserReportCount = 0;
let currentData = null;
let selectedZone = null;

// ── Tab switching ─────────────────────────────────────────────────────────────
function setActiveTab(tabName) {
  document.querySelectorAll('.tab-btn').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.tab === tabName);
  });
  document.querySelectorAll('.tab-panel').forEach((panel) => {
    panel.classList.toggle('active', panel.id === 'tab-' + tabName);
  });
  document.body.classList.toggle('live-tab-active', tabName === 'live');
}

document.querySelectorAll('.tab-btn').forEach((btn) => {
  btn.addEventListener('click', () => setActiveTab(btn.dataset.tab));
});

setActiveTab(document.querySelector('.tab-btn.active')?.dataset.tab || 'map');

// ── Stadium map helpers ───────────────────────────────────────────────────────

function levelClass(level) {
  if (level === 'critical') return 'critical';
  if (level === 'high') return 'high';
  if (level === 'moderate') return 'moderate';
  return 'low';
}

function updateCriticalAssessor(risk, frameMetric) {
  const lvl = levelClass((risk && risk.level) || 'low');
  const scorePct = Math.round((((risk && risk.score) || 0) * 100));

  const levelEl = document.getElementById('assessorLevel');
  const scoreEl = document.getElementById('assessorScore');
  const densityEl = document.getElementById('assessorDensity');
  const turbEl = document.getElementById('assessorTurbulence');
  const occEl = document.getElementById('assessorOccupancy');
  const trendEl = document.getElementById('assessorTrend');
  const rationaleEl = document.getElementById('assessorRationale');
  const actionEl = document.getElementById('assessorAction');

  if (!levelEl || !scoreEl || !densityEl || !turbEl || !occEl || !trendEl || !rationaleEl || !actionEl) {
    return;
  }

  levelEl.textContent = lvl.toUpperCase();
  levelEl.className = `assessor-level ${lvl}`;
  scoreEl.textContent = `${scorePct}%`;

  const density = ((frameMetric && frameMetric.density_per_square_meter) || 0);
  const turbulence = ((frameMetric && frameMetric.motion_turbulence) || 0);
  const occupancy = ((frameMetric && frameMetric.occupancy_ratio) || 0);

  densityEl.textContent = `${density.toFixed(2)} p/m2`;
  turbEl.textContent = turbulence.toFixed(2);
  occEl.textContent = `${Math.round(occupancy * 100)}%`;

  const rationale = (risk && risk.rationale) || [];
  const actions = (risk && risk.recommended_actions) || [];

  const trend = rationale.find((line) => line.toLowerCase().includes('accelerat'))
    ? 'Escalating'
    : (lvl === 'critical' || lvl === 'high')
      ? 'Elevated'
      : 'Stable';

  trendEl.textContent = trend;
  rationaleEl.textContent = rationale.length
    ? rationale[0]
    : 'Risk model is monitoring live density and turbulence changes.';
  actionEl.textContent = actions.length
    ? actions[0]
    : 'Continue monitoring this zone.';
}

function seatToZone(seat) {
  const match = (seat || "").toUpperCase().match(/[A-F]/);
  return match ? match[0] : null;
}

function severityRank(v) {
  return { low: 1, medium: 2, high: 3, critical: 4 }[v] || 1;
}

function buildZoneAlerts(userReports) {
  const zones = {};
  for (const r of userReports) {
    const z = seatToZone(r.seat);
    if (!z) continue;
    if (!zones[z]) zones[z] = { severity: r.severity, reports: [r] };
    else {
      zones[z].reports.push(r);
      if (severityRank(r.severity) > severityRank(zones[z].severity))
        zones[z].severity = r.severity;
    }
  }
  return zones;
}

function renderMap(zoneAlerts) {
  for (const letter of ["A","B","C","D","E","F"]) {
    const el = document.getElementById(`zone-${letter}`);
    if (!el) continue;
    const info = zoneAlerts[letter];
    el.setAttribute("fill", info ? SEVERITY_COLORS[info.severity] : DEFAULT_ZONE_FILL);
    el.setAttribute("opacity", info ? "0.9" : "0.85");
  }

  const seatMarkers = document.getElementById("seatMarkers");
  seatMarkers.innerHTML = "";
  for (const [letter, info] of Object.entries(zoneAlerts)) {
    const center = ZONE_CENTERS[letter];
    if (!center) continue;
    const color = SEVERITY_COLORS[info.severity] || "#ef4444";
    const seats = info.reports.map(r => r.seat);
    seatMarkers.innerHTML += `
      <g class="seat-pin" data-zone="${letter}">
        <circle cx="${center.x}" cy="${center.y}" r="18" fill="${color}" stroke="white" stroke-width="2.5" filter="url(#pinShadow)"/>
        <text x="${center.x}" y="${center.y - 3}" text-anchor="middle" fill="white" font-size="9" font-weight="800" font-family="Barlow,sans-serif">${letter}</text>
        <text x="${center.x}" y="${center.y + 8}" text-anchor="middle" fill="white" font-size="7.5" font-family="Barlow,sans-serif">${seats.length > 1 ? seats.length + " reports" : seats[0]}</text>
      </g>`;
  }

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
  document.getElementById("sectionPopup").classList.remove("hidden");
}

async function callControl(action) {
  await fetch("/api/control", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action })
  });
  await refreshState();
}

function reportCard(report) {
  const isUser = report.source === "attendee";
  const zone   = seatToZone(report.seat) || "?";
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
    </li>`;
}

// ── State polling ─────────────────────────────────────────────────────────────
function topRiskSections(data, count = 2) {
  // Count reports per section, fall back to risk_score if no reports exist
  const reports = Array.isArray(data?.reports) ? data.reports : [];
  const countMap = {};
  reports.forEach((r) => {
    const seat = (r.seat || "").toUpperCase();
    // Extract section letter: e.g. "B12" -> "B", "Section B" -> "B"
    const match = seat.match(/([A-F])/);
    if (match) countMap[match[1]] = (countMap[match[1]] || 0) + 1;
  });

  const sections = Array.isArray(data?.sections) ? data.sections.slice() : [];
  sections.sort((a, b) => {
    const aC = countMap[a.id] || 0;
    const bC = countMap[b.id] || 0;
    return bC !== aC ? bC - aC : (b.risk_score || 0) - (a.risk_score || 0);
  });
  return sections.slice(0, count).map((s) => ({ ...s, report_count: countMap[s.id] || 0 }));
}


function escHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}


let paAudioUrls = [];

function clearPAAudioUrls() {
  for (const url of paAudioUrls) {
    try { URL.revokeObjectURL(url); } catch (e) { /* noop */ }
  }
  paAudioUrls = [];
}

function setPAStatus(text, kind = "") {
  const el = document.getElementById("paStatus");
  if (!el) return;
  el.textContent = text;
  el.className = `pa-status ${kind}`.trim();
}

function parseLanguages() {
  const input = document.getElementById("paLangs");
  const raw = (input?.value || "en,es,hi").split(",");
  const langs = raw.map((v) => v.trim().toLowerCase()).filter(Boolean);
  return langs.length ? langs : ["en", "es"];
}

function renderPAAudioList(clips, messages) {
  const listEl = document.getElementById("paAudioList");
  if (!listEl) return;

  clearPAAudioUrls();

  if (!Array.isArray(clips) || !clips.length) {
    listEl.innerHTML = "";
    return;
  }

  const items = [];
  clips.forEach((clip, idx) => {
    if (!clip?.ok || !clip?.audio_b64) {
      items.push(`
        <li class="pa-audio-item">
          <span>${escHtml((clip?.language || "?").toUpperCase())}: ${escHtml(clip?.error || "Audio unavailable")}</span>
        </li>
      `);
      return;
    }

    try {
      const binary = atob(clip.audio_b64);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
      const blob = new Blob([bytes], { type: clip.mime_type || "audio/mpeg" });
      const url = URL.createObjectURL(blob);
      paAudioUrls.push(url);

      const text = messages?.[idx]?.text || "Emergency PA message";
      items.push(`
        <li class="pa-audio-item">
          <span>${escHtml((clip.language || "?").toUpperCase())}: ${escHtml(text.slice(0, 70))}${text.length > 70 ? "..." : ""}</span>
          <button class="pa-play-btn" data-audio-url="${escHtml(url)}">Play</button>
        </li>
      `);
    } catch (e) {
      items.push(`
        <li class="pa-audio-item">
          <span>${escHtml((clip?.language || "?").toUpperCase())}: decode failed</span>
        </li>
      `);
    }
  });

  listEl.innerHTML = items.join("");
}

async function triggerEmergencyPA(force = false) {
  const btnCheck = document.getElementById("btnCheckPA");
  const btnForce = document.getElementById("btnForcePA");
  try {
    if (btnCheck) btnCheck.disabled = true;
    if (btnForce) btnForce.disabled = true;
    setPAStatus("Checking trigger conditions and requesting multilingual PA...", "");

    const resp = await fetch("/api/emergency-pa", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        languages: parseLanguages(),
        include_audio: true,
        force,
      }),
    });

    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status}`);
    }

    const data = await resp.json();
    renderPAAudioList(data.audio, data.messages);

    if (!data.enabled) {
      setPAStatus(data.status || "Emergency PA not enabled yet.", "warn");
      return;
    }

    if (paAudioUrls.length) {
      try {
        const audio = new Audio(paAudioUrls[0]);
        await audio.play();
        setPAStatus("Emergency PA generated and first language clip is playing.", "ok");
      } catch (playErr) {
        setPAStatus("PA generated. Click Play on a language clip to hear audio.", "ok");
      }
    } else {
      setPAStatus("PA trigger passed, but audio generation failed for all languages.", "err");
    }
  } catch (err) {
    setPAStatus(`Emergency PA request failed: ${err}`, "err");
  } finally {
    if (btnCheck) btnCheck.disabled = false;
    if (btnForce) btnForce.disabled = false;
  }
}

function renderAiSuggestions(data) {
  const listEl = document.getElementById("actionsList");
  const contextEl = document.getElementById("aiContext");
  if (!listEl) return;

  const commander = data.incident_commander || {};
  const queue = Array.isArray(commander.queue) ? commander.queue : [];
  const top = topRiskSections(data, 2);
  const topLabel = top.length
    ? top.map((s) => `Section ${s.id} (${s.report_count})`).join(" | ")
    : "No elevated sections";

  if (contextEl) {
    const provider = (commander.provider || "fallback").toUpperCase();
    contextEl.innerHTML = `
      <div class="ai-context-grid">
        <span class="ai-chip"><strong>Status</strong>${escHtml(data.phase.label)}</span>
        <span class="ai-chip"><strong>Active Reports</strong>${escHtml(data.system_metrics.active_reports)}</span>
        <span class="ai-chip"><strong>Top Risk</strong>${escHtml(topLabel)}</span>
        <span class="ai-chip source"><strong>Source</strong>${escHtml(provider)}</span>
      </div>
    `;
  }

  if (queue.length) {
    listEl.innerHTML = queue.map((item) => {
      const priority = (item.priority || "medium").toLowerCase();
      const confidence = Math.round((item.confidence || 0) * 100);
      return `
        <li class="ai-action-card">
          <div class="ai-action-head">
            <span class="ai-priority ${priority}">${escHtml(priority.toUpperCase())}</span>
            <span class="ai-confidence">${confidence}% confidence</span>
          </div>
          <div class="ai-action-main">${escHtml(item.action || "Action")}</div>
          <div class="ai-action-target">${escHtml(item.target || "Venue")}</div>
          <div class="ai-action-rationale">${escHtml(item.rationale || "")}</div>
        </li>
      `;
    }).join("");
    return;
  }

  const fallback = Array.isArray(data.ai_actions) ? data.ai_actions : [];
  listEl.innerHTML = fallback.length
    ? fallback.map((line) => `<li class="ai-action-card">${escHtml(line)}</li>`).join("")
    : "<li class='ai-action-card'>No actions yet</li>";
}

async function refreshState() {
  try {
    const resp = await fetch("/api/state");
    const data = await resp.json();
    currentData = data;

    document.getElementById("phasePill").textContent   = data.phase.label;
    document.getElementById("eventArtist").textContent = data.event.artist;
    document.getElementById("eventVenue").textContent  = `${data.event.venue}, ${data.event.location}`;

    const tagEl = document.getElementById("tagline");
    if (tagEl) tagEl.textContent = data.tagline || "";
    renderAiSuggestions(data);

    const userReports = data.reports.filter(r => r.source === "attendee");
    const userCount   = userReports.length;

    const banner = document.getElementById("userAlertsBanner");
    if (userCount > lastUserReportCount && lastUserReportCount !== -1) {
      banner.classList.remove("hidden");
      setTimeout(() => banner.classList.add("hidden"), 4000);
    }
    lastUserReportCount = userCount;

    document.getElementById("reportsList").innerHTML = data.reports.length
      ? data.reports.map(reportCard).join("")
      : "<li class='report-item'>No active incident reports</li>";
    document.getElementById("feedCount").textContent = data.reports.length;

    const criticalZones = data.sections.filter(s => s.risk_band === "critical").length;
    const medicalAlerts = data.reports.filter(r => {
      const m = (r.message || "").toLowerCase();
      return m.includes("medical") || m.includes("fell") || m.includes("assistance");
    }).length;

    document.getElementById("mv-reports").textContent = data.system_metrics.active_reports;
    document.getElementById("mv-zones").textContent   = criticalZones;
    document.getElementById("mv-user").textContent    = userCount;
    document.getElementById("mv-medical").textContent = medicalAlerts;

    const zoneAlerts = buildZoneAlerts(userReports);
    renderMap(zoneAlerts);
    if (selectedZone) showZonePopup(selectedZone, zoneAlerts);

  } catch (e) {
    console.warn("State refresh failed:", e);
  }
}

// Zone click handlers
document.querySelectorAll(".zone").forEach(path => {
  path.addEventListener("click", () => {
    if (!currentData) return;
    const userReports = currentData.reports.filter(r => r.source === "attendee");
    showZonePopup(path.dataset.zone, buildZoneAlerts(userReports));
  });
});

document.getElementById("popupClose").addEventListener("click", () => {
  document.getElementById("sectionPopup").classList.add("hidden");
  selectedZone = null;
});

for (const btn of document.querySelectorAll("button[data-action]")) {
  btn.addEventListener("click", () => callControl(btn.dataset.action));
}

// ── Camera zone bars ──────────────────────────────────────────────────────────
function gridToNamedZones(grid) {
  const zones = {};
  const rowGroups = [[0,1],[2,3]];
  const colGroups = [[0,1],[2,3],[4,5]];
  const names     = [['NW','N','NE'],['SW','S','SE']];
  rowGroups.forEach((rows, ri) => {
    colGroups.forEach((cols, ci) => {
      let sum = 0, cnt = 0;
      rows.forEach(r => cols.forEach(c => {
        if (grid[r] && grid[r][c] !== undefined) { sum += grid[r][c]; cnt++; }
      }));
      zones[names[ri][ci]] = cnt ? sum / cnt : 0;
    });
  });
  return zones;
}

function classifyHeat(h) {
  if (h >= ZONE_THRESH.critical) return 'critical';
  if (h >= ZONE_THRESH.high)     return 'high';
  if (h >= ZONE_THRESH.moderate) return 'moderate';
  return 'low';
}

function zoneColor(pct) {
  if (pct < ZONE_DANGER_PCT) return ZONE_NEUTRAL_COLOR;
  return criticalRedShade(pct, 1);
}

function initZoneBars() {
  const container = document.getElementById('zone-bars');
  Array.from(container.querySelectorAll('.zone-row')).forEach(e => e.remove());
  ZONE_NAMES.forEach(name => {
    const row = document.createElement('div');
    row.className = 'zone-row';
    row.id = `zrow-${name}`;
    row.innerHTML = `
      <div class="zone-label-row">
        <span class="zone-name">${ZONE_LABELS[name]}</span>
        <span class="zone-val" id="zval-${name}" style="color:${ZONE_NEUTRAL_COLOR}">0%</span>
      </div>
      <div class="zone-track"><div class="zone-fill" id="zfill-${name}" style="width:0%;background:${ZONE_NEUTRAL_COLOR}"></div></div>`;
    container.appendChild(row);
  });
}

const lastEventTime = {};
function shouldFire(key, videoTime, cooldown = 4) {
  if ((videoTime - (lastEventTime[key] || -99)) < cooldown) return false;
  lastEventTime[key] = videoTime;
  return true;
}

function updateZones(grid, peopleCount, videoTime) {
  const named = gridToNamedZones(grid);
  ZONE_NAMES.forEach(name => {
    const raw = named[name] || 0;
    const z   = zoneState[name];
    const prevSmoothed = z.smoothed;

    z.smoothed = z.smoothed * 0.6 + raw * 0.4;
    z.prev     = z.heat;
    z.heat     = z.smoothed;

    const newState = classifyHeat(z.heat);
    const pct      = zonePercentFromHeat(z.heat);
    const col      = zoneColor(pct);

    const fill = document.getElementById(`zfill-${name}`);
    const val  = document.getElementById(`zval-${name}`);
    if (fill) { fill.style.width = pct.toFixed(1) + '%'; fill.style.backgroundColor = col; }
    if (val)  { val.textContent = pct.toFixed(0) + '%'; val.style.color = col; }

    const oldState = z.state;
    if (newState !== oldState) {
      z.state = newState;
      fireZoneEvent(name, oldState, newState, z.heat, pct, peopleCount, videoTime); // z.heat passed for potential future use
    } else {
      const delta = z.smoothed - prevSmoothed;
      if (Math.abs(delta) > 0.015 && newState !== 'low')
        fireDeltaEvent(name, delta > 0 ? 'up' : 'down', delta, z.heat, pct, peopleCount, videoTime);
    }
  });
}

function fireZoneEvent(name, oldState, newState, _heat, pct, count, t) {
  if (!shouldFire(`state-${name}`, t, 5)) return;
  const label = ZONE_LABELS[name];
  const transitions = {
    'low→moderate':      ['info',  'ℹ', `${label} starting to fill — ${count} people detected`],
    'moderate→high':     ['warn',  '⬆', `${label} getting busier — crowd pressure rising`],
    'high→critical':     ['up',    '⚠', `${label} reaching critical density — consider redirecting`],
    'critical→high':     ['down',  '↓', `${label} easing slightly — still high, monitor closely`],
    'high→moderate':     ['down',  '↓', `${label} clearing — crowd dispersing`],
    'moderate→low':      ['down',  '✓', `${label} now clear — density back to safe levels`],
    'low→high':          ['warn',  '⬆', `${label} sudden crowd surge detected`],
    'low→critical':      ['up',    '⚠', `${label} rapid critical surge — act immediately`],
    'critical→moderate': ['down',  '↓', `${label} significant crowd relief detected`],
    'critical→low':      ['down',  '✓', `${label} cleared from critical — good`],
    'high→low':          ['down',  '✓', `${label} cleared quickly`],
  };
  const key = `${oldState}→${newState}`;
  const [type, icon, msg] = transitions[key] || ['info', 'ℹ', `${label} changed: ${oldState} → ${newState}`];
  pushEvent(type, icon, msg, pct, t);
}

function fireDeltaEvent(name, dir, delta, _heat, pct, count, t) {
  if (!shouldFire(`delta-${name}-${dir}`, t, 8)) return;
  const label = ZONE_LABELS[name];
  const mag   = Math.abs(delta);
  let type, icon, msg;
  if (dir === 'up') {
    type = mag > 0.04 ? 'warn' : 'info';
    icon = mag > 0.04 ? '⬆' : '↑';
    msg  = mag > 0.04 ? `${label} adding people fast — ${count} total` : `${label} slowly getting busier`;
  } else {
    type = 'down';
    icon = mag > 0.04 ? '⬇' : '↓';
    msg  = mag > 0.04 ? `${label} clearing quickly` : `${label} gradually freeing up`;
  }
  pushEvent(type, icon, msg, pct, t);
}

function pushEvent(type, icon, msg, pct, videoTime) {
  const feed = document.getElementById('event-feed');
  const na   = document.getElementById('no-events');
  if (na) na.remove();

  const el = document.createElement('div');
  el.className = `event-item ev-${type}`;
  el.innerHTML = `
    <div class="ev-icon">${icon}</div>
    <div class="ev-body">
      <div class="ev-msg">${msg}</div>
      <div class="ev-meta">
        <span>${formatTime(videoTime)}</span>
        <span>${pct.toFixed(0)}% density</span>
      </div>
    </div>`;
  feed.prepend(el);
  while (feed.children.length > 60) feed.removeChild(feed.lastChild);
}

// ── WebSocket / video stream ──────────────────────────────────────────────────
function startStream() {
  const fi = document.getElementById('file-input');
  if (!fi.files.length) { alert('Select a video file first.'); return; }
  stopStream();

  peakDensity = 0; lastCount = 0; framesSent = 0; lastResult = null;
  ZONE_NAMES.forEach(n => { zoneState[n] = { heat:0, prev:0, state:'low', smoothed:0 }; });
  Object.keys(lastEventTime).forEach(k => delete lastEventTime[k]);
  document.getElementById('event-feed').innerHTML =
    '<div class="no-events" id="no-events">Events will appear as crowd moves</div>';

  initZoneBars();
  setLive('Connecting…', false);
  document.getElementById('btn-start').style.display = 'none';
  document.getElementById('btn-stop').style.display  = 'block';
  document.getElementById('tabLiveBadge').classList.remove('hidden');

  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const host  = location.hostname + (location.port ? ':' + location.port : '');
  ws = new WebSocket(`${proto}://${host}/ws/realtime-analysis`);
  ws.binaryType = 'arraybuffer';

  ws.onopen = () => {
    ws.send(JSON.stringify({
      area_value:              parseFloat(document.getElementById('area-val').value),
      area_unit:               document.getElementById('area-unit').value,
      alert_density_threshold: parseFloat(document.getElementById('density-thresh').value),
      zone_x: 0, zone_y: 0, zone_width: 1, zone_height: 1,
    }));
  };

  ws.onmessage = (evt) => {
    const msg = JSON.parse(evt.data);
    if (msg.type === 'session_ready') {
      setLive('Live', true);
      loadAndPlayVideo(fi.files[0]);
    } else if (msg.type === 'analysis') {
      document.getElementById('lat').textContent = `${Date.now() - sentAt}ms`;
      lastResult = msg.payload;
      // Push live counts to header stats
      const fm   = msg.payload.frame_metric;
      const risk = msg.payload.risk_assessment;
      document.getElementById('hsPeople').textContent = fm.people_count;
      document.getElementById('hsDensity').textContent = fm.density_per_square_meter.toFixed(2);
      const rEl = document.getElementById('hsRisk');
      rEl.textContent = risk.level.toUpperCase();
      rEl.className = `hs-val risk-val ${risk.level}`;
      updateCriticalAssessor(risk, fm);
    } else if (msg.type === 'error') {
      console.error('Server error:', msg.message);
      alert('Server error: ' + msg.message);
      stopStream();
    }
  };

  ws.onerror = () => {
    fetch('/health').then(r => r.json())
      .then(() => alert('WebSocket failed. Server is up — check console.'))
      .catch(() => alert('Server unreachable. Run: uvicorn app.main:app --reload'));
    resetBtns(); setLive('Error', false);
  };

  ws.onclose = (e) => {
    if (document.getElementById('btn-stop').style.display !== 'none')
      setLive(`Closed (${e.code})`, false);
  };
}

function loadAndPlayVideo(file) {
  videoEl = document.createElement('video');
  videoEl.src = URL.createObjectURL(file);
  videoEl.muted = true;
  videoEl.playsInline = true;
  videoEl.loop = false;
  videoEl.currentTime = 0;

  videoEl.addEventListener('loadedmetadata', () => {
    const canvas  = document.getElementById('main-canvas');
    canvas.width  = videoEl.videoWidth;
    canvas.height = videoEl.videoHeight;

    offscreen = document.createElement('canvas');
    offscreen.width  = Math.min(videoEl.videoWidth, 640);
    offscreen.height = Math.min(videoEl.videoHeight, 480);
    offCtx = offscreen.getContext('2d');

    document.getElementById('placeholder').style.display = 'none';
    canvas.style.display = 'block';

    const ms = parseInt(document.getElementById('interval-ms').value) || 400;
    if (sendTimer) clearInterval(sendTimer);
    sendTimer = setInterval(() => {
      if (!ws || ws.readyState !== WebSocket.OPEN || !videoEl || videoEl.paused || videoEl.ended) return;
      offCtx.drawImage(videoEl, 0, 0, offscreen.width, offscreen.height);
      offscreen.toBlob(blob => {
        if (!blob || !ws || ws.readyState !== WebSocket.OPEN) return;
        blob.arrayBuffer().then(buf => { sentAt = Date.now(); ws.send(buf); framesSent++; });
      }, 'image/jpeg', 0.75);
    }, ms);

    renderLoop();
    videoEl.play();
  }, { once: true });

  videoEl.addEventListener('ended', () => {
    setLive('Done', false);
    stopStream();
  }, { once: true });
}

// ── Render loop ───────────────────────────────────────────────────────────────
function renderLoop() {
  if (!videoEl || videoEl.ended) return;

  const canvas = document.getElementById('main-canvas');
  const ctx    = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;

  ctx.drawImage(videoEl, 0, 0, W, H);

  if (lastResult) {
    const fm   = lastResult.frame_metric;
    const risk = lastResult.risk_assessment;
    const hb64 = lastResult.heatmap_b64;
    const grid = lastResult.density_grid;

    const count   = fm.people_count;
    const density = fm.density_per_square_meter;
    if (density > peakDensity) peakDensity = density;

    // Heatmap overlay
    if (hb64) {
      if (!window._hmImg) window._hmImg = new Image();
      const img = window._hmImg;
      const src = 'data:image/jpeg;base64,' + hb64;
      if (img.src !== src) img.src = src;
      if (img.complete && img.naturalWidth > 0) {
        ctx.globalAlpha = 0.45;
        ctx.drawImage(img, 0, 0, W, H);
        ctx.globalAlpha = 1.0;
      }
    }

    // // Density grid cell overlay
    // if (grid && grid.length) {
    //   const rows = grid.length, cols = grid[0].length;
    //   const cw = W / cols, ch = H / rows;
    //   grid.forEach((row, r) => {
    //     row.forEach((val, c) => {
    //       const pct = zonePercentFromHeat(val);
    //       if (pct < ZONE_DANGER_PCT) return;
    //       const t = Math.min(Math.max((pct - ZONE_DANGER_PCT) / (100 - ZONE_DANGER_PCT), 0), 1);
    //       const alpha = 0.12 + (0.22 * t);
    //       ctx.fillStyle = criticalRedShade(pct, alpha);
    //       ctx.fillRect(c * cw, r * ch, cw, ch);
    //     });
    //   });
    // }

    // People count badge
    const col = fm.alert_triggered ? '#ef4444' : density >= 2.5 ? '#f59e0b' : '#22c55e';
    ctx.fillStyle = 'rgba(0,0,0,0.72)';
    roundRect(ctx, 12, 12, 215, 52, 8); ctx.fill();
    ctx.fillStyle = col; ctx.font = 'bold 30px system-ui';
    ctx.fillText(count, 20, 50);
    ctx.fillStyle = '#94a3b8'; ctx.font = '12px system-ui';
    ctx.fillText('people in frame', 64, 34);
    ctx.fillStyle = col; ctx.font = 'bold 11px system-ui';
    ctx.fillText(`${density.toFixed(2)} p/m²`, 64, 54);

    // Delta indicator
    const delta = count - lastCount;
    if (delta !== 0) {
      ctx.fillStyle = delta > 0 ? '#ef4444' : '#22c55e';
      ctx.font = 'bold 13px system-ui';
      ctx.fillText((delta > 0 ? '▲ +' : '▼ ') + delta, 20, 72);
    }

    // Zone labels on video
    if (grid && grid.length) {
      const named = gridToNamedZones(grid);
      const positions = {
        NW:[0.1,0.15], N:[0.5,0.08], NE:[0.88,0.15],
        SW:[0.1,0.82], S:[0.5,0.90], SE:[0.88,0.82],
      };
      ctx.font = 'bold 16px system-ui'; ctx.textAlign = 'center';
      ZONE_NAMES.forEach(name => {
        const h = named[name] || 0;
        const [fx, fy] = positions[name];
        const pct = zonePercentFromHeat(h);
        const isDanger = pct >= ZONE_DANGER_PCT;
        const zcol = zoneColor(pct);
        const boxW = 128;
        const boxH = 34;
        ctx.fillStyle = isDanger ? 'rgba(22,6,6,0.74)' : 'rgba(15,23,42,0.62)';
        roundRect(ctx, fx * W - (boxW / 2), fy * H - (boxH / 2), boxW, boxH, 7); ctx.fill();
        ctx.strokeStyle = isDanger ? zcol : 'rgba(148,163,184,0.45)';
        ctx.lineWidth = isDanger ? 2 : 1;
        ctx.stroke();
        ctx.fillStyle = isDanger ? zcol : '#cbd5e1';
        ctx.fillText(`${name} ${pct.toFixed(0)}%`, fx * W, fy * H + 6);
      });
      ctx.textAlign = 'left';
    }

    // Alert banner
    if (fm.alert_triggered) {
      const flash = Math.sin(Date.now() / 180) > 0;
      ctx.fillStyle = flash ? 'rgba(239,68,68,0.88)' : 'rgba(180,20,20,0.72)';
      ctx.fillRect(0, H - 40, W, 40);
      ctx.fillStyle = '#fff'; ctx.font = 'bold 13px system-ui'; ctx.textAlign = 'center';
      ctx.fillText('⚠  DENSITY THRESHOLD BREACHED — TAKE IMMEDIATE ACTION', W/2, H - 14);
      ctx.textAlign = 'left';
    }

    // Timestamp
    ctx.fillStyle = 'rgba(0,0,0,0.55)';
    roundRect(ctx, W-110, 12, 98, 22, 5); ctx.fill();
    ctx.fillStyle = '#64748b'; ctx.font = '11px system-ui';
    ctx.fillText(`${formatTime(videoEl.currentTime)}  f${framesSent}`, W-104, 27);

    setStat('s-people',  count,                      10,  20);
    setStat('s-density', density.toFixed(2),          2.5, 4.0);
    setStat('s-peak',    peakDensity.toFixed(2),      2.5, 4.0);
    setStat('s-turb',    fm.motion_turbulence.toFixed(2), 0.5, 1.0);
    updateRing(risk.score, risk.level);
    updateBottomBar(count, density);
    if (grid) updateZones(grid, count, videoEl.currentTime);

    lastCount = count;
  }

  document.getElementById('b-time').textContent   = formatTime(videoEl?.currentTime || 0);
  document.getElementById('b-frames').textContent = framesSent;
  rafId = requestAnimationFrame(renderLoop);
}

function stopStream() {
  if (sendTimer) { clearInterval(sendTimer); sendTimer = null; }
  if (rafId)     { cancelAnimationFrame(rafId); rafId = null; }
  if (ws)        { ws.close(); ws = null; }
  if (videoEl)   {
    videoEl.pause();
    if (videoEl.src) URL.revokeObjectURL(videoEl.src);
    videoEl.removeAttribute('src');
    videoEl.load();
    videoEl = null;
  }
  document.getElementById('tabLiveBadge').classList.add('hidden');
  resetBtns();
  setLive('Idle', false);
  updateCriticalAssessor({ score: 0, level: 'low', rationale: [], recommended_actions: [] }, { density_per_square_meter: 0, motion_turbulence: 0, occupancy_ratio: 0 });
}

function resetBtns() {
  document.getElementById('btn-start').style.display = 'block';
  document.getElementById('btn-stop').style.display  = 'none';
}

// ── UI helpers ────────────────────────────────────────────────────────────────
function updateRing(score, level) {
  const cols = { low:'#22c55e', moderate:'#3b82f6', high:'#f59e0b', critical:'#ef4444' };
  const col  = cols[level] || '#64748b';
  document.getElementById('ring-track').setAttribute('stroke-dashoffset', (CIRC*(1-score)).toFixed(1));
  document.getElementById('ring-track').setAttribute('stroke', col);
  document.getElementById('ring-pct').textContent = Math.round(score*100)+'%';
  document.getElementById('ring-pct').setAttribute('fill', col);
  document.getElementById('ring-level').textContent = level.toUpperCase();
  document.getElementById('ring-level').setAttribute('fill', col);
  document.getElementById('ring-sub').textContent =
    score>=.8?'CRITICAL — act now':score>=.52?'High — monitor':score>=.28?'Moderate — watch':'Within safe limits';
}

function setStat(id, val, w, d) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = val;
  const n = parseFloat(val);
  el.className = 'sv ' + (n >= d ? 'danger' : n >= w ? 'warn' : 'ok');
}

function updateBottomBar(count, density) {
  const delta = count - lastCount;
  document.getElementById('b-count').childNodes[0].nodeValue = count;
  const dEl = document.getElementById('b-delta');
  dEl.textContent = delta===0?'':(delta>0?' ▲+'+delta:' ▼'+delta);
  dEl.className   = 'delta '+(delta>0?'up':delta<0?'dn':'eq');
  const pct = Math.min(density/4.0*100, 100);
  const col = density>=4.0?'#ef4444':density>=2.5?'#f59e0b':'#22c55e';
  document.getElementById('prog-fill').style.width           = pct+'%';
  document.getElementById('prog-fill').style.backgroundColor = col;
  document.getElementById('prog-label').textContent          = pct.toFixed(0)+'%';
}

function formatTime(s) {
  const m = Math.floor(s/60), sec = Math.floor(s%60);
  return `${m}:${sec.toString().padStart(2,'0')}`;
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x+r,y); ctx.lineTo(x+w-r,y); ctx.arcTo(x+w,y,x+w,y+r,r);
  ctx.lineTo(x+w,y+h-r); ctx.arcTo(x+w,y+h,x+w-r,y+h,r);
  ctx.lineTo(x+r,y+h); ctx.arcTo(x,y+h,x,y+h-r,r);
  ctx.lineTo(x,y+r); ctx.arcTo(x,y,x+r,y,r); ctx.closePath();
}

function setLive(txt, on) {
  document.getElementById('live-txt').textContent = txt;
  document.getElementById('live-dot').className   = 'dot' + (on ? ' on' : '');
}

// ── Bootstrap ─────────────────────────────────────────────────────────────────
const paListEl = document.getElementById("paAudioList");
if (paListEl) {
  paListEl.addEventListener("click", (event) => {
    const btn = event.target.closest("button[data-audio-url]");
    if (!btn) return;
    const url = btn.getAttribute("data-audio-url");
    if (!url) return;
    const audio = new Audio(url);
    audio.play().catch(() => setPAStatus("Browser blocked autoplay. Click Play again.", "warn"));
  });
}

document.getElementById("btnCheckPA")?.addEventListener("click", () => triggerEmergencyPA(false));
document.getElementById("btnForcePA")?.addEventListener("click", () => triggerEmergencyPA(true));
window.addEventListener("beforeunload", clearPAAudioUrls);

initZoneBars();
refreshState();
setInterval(refreshState, 1200);
















