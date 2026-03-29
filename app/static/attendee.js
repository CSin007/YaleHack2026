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
const videoPlaceholder = document.getElementById("videoPlaceholder");
const videoBadge = document.getElementById("videoBadge");
const myReports = document.getElementById("myReports");
const myReportsList = document.getElementById("myReportsList");
const phoneTime = document.getElementById("phoneTime");

let countdownValue = 180;
let mySubmittedReports = [];

function updatePhoneTime() {
  phoneTime.textContent = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}
updatePhoneTime();
setInterval(updatePhoneTime, 30000);

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

  const response = await fetch("/api/attendee-alert", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ seat, severity, message: fullMessage })
  });

  sendBtn.disabled = false;
  sendBtn.textContent = "Send Alert Now";

  if (response.ok) {
    const time = new Date().toLocaleTimeString();
    showStatus(`Alert sent at ${time}. Organizer has been notified.`, "ok");
    mySubmittedReports.unshift({ seat, severity, message, note, time });
    mySubmittedReports = mySubmittedReports.slice(0, 5);
    renderMyReports();
    noteInput.value = "";
    await refresh();
  } else {
    showStatus("Failed to send alert. Please try again.", "error");
  }
}

function showStatus(text, type) {
  sendStatus.textContent = text;
  sendStatus.className = `send-status ${type}`;
  sendStatus.classList.remove("hidden");
  setTimeout(() => sendStatus.classList.add("hidden"), 5000);
}

function renderMyReports() {
  if (!mySubmittedReports.length) return;
  myReports.classList.remove("hidden");
  myReportsList.innerHTML = mySubmittedReports.map(r => `
    <li>
      <strong>${r.seat}</strong> — ${r.message}
      ${r.note ? `<em class="report-note">"${r.note}"</em>` : ""}
      <span class="tiny">${r.time}</span>
    </li>
  `).join("");
}

function startCountdown() {
  const countdownTimer = document.getElementById("countdownTimer");
  setInterval(() => {
    if (countdownValue > 0) {
      countdownValue -= 1;
      const m = Math.floor(countdownValue / 60);
      const s = countdownValue % 60;
      countdownTimer.textContent = `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
    } else {
      countdownTimer.textContent = "Notify staff";
    }
  }, 1000);
}

function bindIssueChips() {
  for (const chip of document.querySelectorAll(".issue-chip")) {
    chip.addEventListener("click", () => {
      document.querySelectorAll(".issue-chip").forEach(c => c.classList.remove("active"));
      chip.classList.add("active");
      messageInput.value = chip.dataset.issue;
      severityInput.value = chip.dataset.severity || "medium";
    });
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
  videoPlaceholder.className = "video-placeholder";

  if (alert.tone === "alert") {
    phoneCard.classList.add("alert");
    statusBanner.classList.add("danger");
    bannerText.textContent = "⚠ High-risk zone detected — follow exit guidance";
    routeStatus.textContent = "Primary route: Exit B, avoid Section 107.";
    routeLine.classList.add("urgent");
    videoPlaceholder.classList.add("active-alert");
    videoBadge.textContent = "ALERT ACTIVE";
    videoBadge.className = "video-badge alert-badge";
    countdownValue = Math.min(countdownValue, 120);
  } else if (alert.tone === "recovery") {
    phoneCard.classList.add("recovery");
    statusBanner.classList.add("recovery");
    bannerText.textContent = "✓ Situation stabilizing — continue to open space";
    routeStatus.textContent = "Flow improving. Continue to open concourse space.";
    routeLine.classList.remove("urgent");
    videoBadge.textContent = "Recovery Mode";
    videoBadge.className = "video-badge recovery-badge";
  } else {
    statusBanner.classList.add("normal");
    bannerText.textContent = "Live safety monitoring active";
    routeStatus.textContent = "Safe route available to nearest open exit.";
    routeLine.classList.remove("urgent");
    videoBadge.textContent = "Standby";
    videoBadge.className = "video-badge";
  }
}

alertForm.addEventListener("submit", sendAlert);
bindIssueChips();
startCountdown();
refresh();
setInterval(refresh, 1200);
