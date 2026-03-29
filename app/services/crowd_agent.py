from __future__ import annotations

import json
import anthropic

TOOLS = [
    {
        "name": "send_gate_instruction",
        "description": "Send an instruction to a specific gate — open, close, throttle entry, or redirect flow.",
        "input_schema": {
            "type": "object",
            "properties": {
                "gate":        {"type": "string", "description": "Gate identifier e.g. 'Gate A', 'North Entrance'"},
                "action":      {"type": "string", "enum": ["close", "throttle", "open", "redirect"], "description": "Action to take"},
                "message":     {"type": "string", "description": "Human-readable instruction for the gate marshal"},
                "urgency":     {"type": "string", "enum": ["low", "medium", "high", "critical"]},
            },
            "required": ["gate", "action", "message", "urgency"],
        },
    },
    {
        "name": "dispatch_staff",
        "description": "Dispatch security or medical staff to a zone.",
        "input_schema": {
            "type": "object",
            "properties": {
                "zone":        {"type": "string", "description": "Zone name e.g. 'West wing', 'North centre'"},
                "staff_type":  {"type": "string", "enum": ["security", "medical", "steward", "all"]},
                "count":       {"type": "integer", "description": "Number of staff to dispatch"},
                "instruction": {"type": "string", "description": "What they should do when they arrive"},
                "urgency":     {"type": "string", "enum": ["low", "medium", "high", "critical"]},
            },
            "required": ["zone", "staff_type", "count", "instruction", "urgency"],
        },
    },
    {
        "name": "broadcast_pa_announcement",
        "description": "Trigger a public address announcement to redirect or calm the crowd.",
        "input_schema": {
            "type": "object",
            "properties": {
                "message":     {"type": "string", "description": "The announcement text to broadcast"},
                "zones":       {"type": "array", "items": {"type": "string"}, "description": "Zones to target, or ['all'] for venue-wide"},
                "urgency":     {"type": "string", "enum": ["low", "medium", "high", "critical"]},
            },
            "required": ["message", "zones", "urgency"],
        },
    },
    {
        "name": "open_overflow_area",
        "description": "Open an overflow or alternate area to absorb crowd from a high-density zone.",
        "input_schema": {
            "type": "object",
            "properties": {
                "area":        {"type": "string", "description": "Name of the overflow area to open"},
                "from_zone":   {"type": "string", "description": "The congested zone this relieves"},
                "instruction": {"type": "string", "description": "Steward instruction for managing the overflow"},
            },
            "required": ["area", "from_zone", "instruction"],
        },
    },
    {
        "name": "alert_emergency_services",
        "description": "Alert emergency services — only use when risk is critical and crowd crush is imminent.",
        "input_schema": {
            "type": "object",
            "properties": {
                "situation":   {"type": "string", "description": "Description of the emergency"},
                "location":    {"type": "string", "description": "Precise location in the venue"},
                "density":     {"type": "number",  "description": "Current peak density in people per square metre"},
            },
            "required": ["situation", "location", "density"],
        },
    },
]

SYSTEM_PROMPT = """
You are a crowd safety AI agent for a live venue. You receive real-time crowd analytics
and must decide what crowd control actions to take using your tools.

Guidelines:
- Always use the minimum intervention necessary — don't escalate unless the data justifies it.
- Prefer PA announcements and staff dispatch before closing gates.
- Only call alert_emergency_services when density exceeds 4.5 p/m² or risk score >= 0.85.
- Be specific: name the zones, give concrete staff counts, write clear PA scripts.
- If multiple zones are problematic, prioritise the highest density zone first.
- Keep PA announcements calm and non-alarming — never use the word "danger" or "stampede".
- When a zone is clearing, proactively open gates or stand down staff.

You may call multiple tools in one response if the situation warrants it.
"""


class CrowdControlAgent:
    def __init__(self, api_key: str, interval_seconds: float = 20.0) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model  = "claude-sonnet-4-20250514"
        self._interval = interval_seconds


    def assess_and_act(self, situation: dict) -> list[dict]:
        """
        situation: {
            people_count, density, peak_density, risk_score, risk_level,
            turbulence, alert_triggered, zone_densities: {name: pct},
            rationale: [str], video_time: str
        }
        Returns list of tool call results (actions taken).
        """
        user_msg = self._build_prompt(situation)

        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=[{"role": "user", "content": user_msg}],
        )

        actions = []
        for block in response.content:
            if block.type == "tool_use":
                actions.append({
                    "tool":   block.name,
                    "input":  block.input,
                })

        # if Claude chose to just respond in text (low risk), include that too
        for block in response.content:
            if block.type == "text" and block.text.strip():
                actions.append({
                    "tool":  "agent_message",
                    "input": {"message": block.text.strip()},
                })

        return actions

    @staticmethod
    def _build_prompt(s: dict) -> str:
        zones = "\n".join(
            f"  - {name}: {pct:.0f}% density"
            for name, pct in s.get("zone_densities", {}).items()
        )
        rationale = "\n".join(f"  - {r}" for r in s.get("rationale", []))

        return f"""
Current crowd situation at {s.get('video_time', 'unknown time')}:

METRICS
  People in frame : {s.get('people_count', 0)}
  Current density : {s.get('density', 0):.2f} p/m²
  Peak density    : {s.get('peak_density', 0):.2f} p/m²
  Risk score      : {s.get('risk_score', 0):.0%}  ({s.get('risk_level', 'low').upper()})
  Turbulence      : {s.get('turbulence', 0):.2f}
  Threshold alert : {'YES — BREACHED' if s.get('alert_triggered') else 'No'}

ZONE BREAKDOWN
{zones if zones else '  No zone data'}

RISK ANALYSIS
{rationale if rationale else '  Within safe limits'}

Based on this data, decide what crowd control actions to take right now.
Only act if the data justifies it. If all zones are safe, say so briefly.
""".strip()