const phaseLabel = document.getElementById("phaseLabel");
const eventLine = document.getElementById("eventLine");
const attendeeTitle = document.getElementById("attendeeTitle");
const attendeeBody = document.getElementById("attendeeBody");
const attendeeDirection = document.getElementById("attendeeDirection");
const phoneCard = document.getElementById("phoneCard");
const alertForm = document.getElementById("alertForm");
const sendStatus = document.getElementById("sendStatus");
const sendBtn = document.getElementById("sendBtn");
const routeStatus = document.getElementById("routeStatus");
const routeLine = document.getElementById("routeLine");
const seatInput = document.getElementById("seatInput");
const severityInput = document.getElementById("severityInput");
const messageInput = document.getElementById("messageInput");
const noteInput = document.getElementById("noteInput");
const statusBanner = document.getElementById("statusBanner");
const bannerText = document.getElementById("bannerText");
const myReports = document.getElementById("myReports");
const myReportsList = document.getElementById("myReportsList");
const phoneTime = document.getElementById("phoneTime");
const crowdSummary = document.getElementById("crowdSummary");
const exitStatus = document.getElementById("exitStatus");
const btnNeedAssist = document.getElementById("btnNeedAssist");
const btnImSafe = document.getElementById("btnImSafe");

let countdownValue = 180;
let mySubmittedReports = [];

function updatePhoneTime() {
  phoneTime.textContent = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function showStatus(text, type) {
  sendStatus.textContent = text;
  sendStatus.className = `send-status ${type}`;
  sendStatus.classList.remove("hidden");
  setTimeout(() => sendStatus.classList.add("hidden"), 5000);
}

function renderMyReports() {
  if (!mySubmittedReports.length) {
    myReports.classList.add("hidden");
    return;
  }

  myReports.classList.remove("hidden");
  myReportsList.innerHTML = mySubmittedReports
    .map((report) => `
      <li>
        <strong>${report.seat}</strong> - ${report.message}
        ${report.note ? `<em class="report-note">"${report.note}"</em>` : ""}
        <span class="tiny">${report.time}</span>
      </li>
    `)
    .join("");
}

async function sendReport(payload) {
  const response = await fetch("/api/attendee-alert", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });

  return response.ok;
}

async function sendAlert(event) {
  event.preventDefault();

  const seat = seatInput.value.trim();
  const message = messageInput.value.trim();
  const note = noteInput.value.trim();
  const severity = severityInput.value;

  if (!seat || !message) {
    showStatus("Enter seat and issue details before sending.", "error");
    return;
  }

  sendBtn.disabled = true;
  sendBtn.textContent = "Sending...";

  const fullMessage = note ? `${message} | Note: ${note}` : message;
  const ok = await sendReport({ seat, severity, message: fullMessage });

  sendBtn.disabled = false;
  sendBtn.textContent = "Send Alert Now";

  if (ok) {
    const time = new Date().toLocaleTimeString();
    showStatus(`Alert sent at ${time}. Organizer has been notified.`, "ok");
    mySubmittedReports.unshift({ seat, severity, message, note, time });
    mySubmittedReports = mySubmittedReports.slice(0, 6);
    noteInput.value = "";
    renderMyReports();
    await refresh();
  } else {
    showStatus("Failed to send alert. Please try again.", "error");
  }
}

function startCountdown() {
  const countdownTimer = document.getElementById("countdownTimer");
  setInterval(() => {
    if (countdownValue > 0) {
      countdownValue -= 1;
    }

    const m = Math.floor(countdownValue / 60);
    const s = countdownValue % 60;
    countdownTimer.textContent = `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
  }, 1000);
}

function bindIssueChips() {
  for (const chip of document.querySelectorAll(".issue-chip[data-issue]")) {
    chip.addEventListener("click", () => {
      document.querySelectorAll(".issue-chip[data-issue]").forEach((node) => node.classList.remove("active"));
      chip.classList.add("active");
      messageInput.value = chip.dataset.issue;
      severityInput.value = chip.dataset.severity || "medium";
    });
  }
}

function bindSafetyButtons() {
  btnNeedAssist.addEventListener("click", () => {
    messageInput.value = "Immediate assistance requested in my row";
    severityInput.value = "critical";
    noteInput.focus();
    showStatus("Assistance template loaded. Add detail and send.", "ok");
  });

  btnImSafe.addEventListener("click", () => {
    showStatus("Safety check-in recorded. Continue following guidance.", "ok");
  });
}

function updateCrowdSummary(data) {
  const sections = data.sections || [];
  const critical = sections.filter((s) => s.risk_band === "critical").length;
  const high = sections.filter((s) => s.risk_band === "high").length;
  const medium = sections.filter((s) => s.risk_band === "medium").length;

  crowdSummary.textContent = `Live crowd pressure: ${critical} critical, ${high} high, ${medium} medium zones.`;

  if (critical > 0) {
    exitStatus.textContent = "Exit guidance priority: use nearest open side exits and avoid high-pressure sections.";
  } else if (high > 0) {
    exitStatus.textContent = "Exit guidance: moderate pressure nearby, continue moving calmly toward open corridors.";
  } else {
    exitStatus.textContent = "Exit status: all primary routes remain open.";
  }
}

async function refresh() {
  const response = await fetch("/api/state");
  const data = await response.json();

  const event = data.event;
  phaseLabel.textContent = data.phase.label;
  eventLine.textContent = `${event.artist} | ${event.venue}, ${event.location}`;

  const alert = data.attendee_alert;
  attendeeTitle.textContent = alert.title;
  attendeeBody.textContent = alert.body;
  attendeeDirection.textContent = alert.direction;

  phoneCard.classList.remove("alert", "recovery");
  statusBanner.className = "status-banner";

  if (alert.tone === "alert") {
    phoneCard.classList.add("alert");
    statusBanner.classList.add("danger");
    bannerText.textContent = "High-risk zone detected. Follow directed exits now.";
    routeStatus.textContent = "Primary route: Exit B. Avoid Section 107.";
    routeLine.classList.add("urgent");
    countdownValue = Math.min(countdownValue, 120);
  } else if (alert.tone === "recovery") {
    phoneCard.classList.add("recovery");
    statusBanner.classList.add("recovery");
    bannerText.textContent = "Conditions stabilizing. Continue moving to open space.";
    routeStatus.textContent = "Flow improving. Continue to open concourse space.";
    routeLine.classList.remove("urgent");
  } else {
    statusBanner.classList.add("normal");
    bannerText.textContent = "Live safety monitoring active.";
    routeStatus.textContent = "Safe route available to nearest open exit.";
    routeLine.classList.remove("urgent");
  }

  updateCrowdSummary(data);
}

alertForm.addEventListener("submit", sendAlert);
bindIssueChips();
bindSafetyButtons();
startCountdown();
updatePhoneTime();
setInterval(updatePhoneTime, 30000);
refresh();
setInterval(refresh, 1200);
