from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List
from urllib import error, request


PRIORITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


@dataclass
class CommanderResult:
    provider: str
    actions: List[Dict[str, Any]]
    error: str | None = None


class LavaIncidentCommander:
    """
    Incident Commander Agent powered by Lava gateway (OpenAI-compatible endpoint).
    Includes in-memory caching so high-frequency polling won't call the model each tick.
    """

    def __init__(
        self,
        api_key: str | None,
        base_url: str = "https://api.lava.so/v1",
        model: str = "openai/gpt-4o-mini",
        cache_ttl_seconds: float = 12.0,
        timeout_seconds: float = 8.0,
    ) -> None:
        self._api_key = (api_key or "").strip()
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._cache_ttl_seconds = cache_ttl_seconds
        self._timeout_seconds = timeout_seconds
        self._cached_at = 0.0
        self._cached_fingerprint = ""
        self._cached_result: CommanderResult | None = None

    @property
    def enabled(self) -> bool:
        return bool(self._api_key)

    def suggest_actions(
        self,
        phase_id: str,
        risks: Dict[str, int],
        reports: List[Dict[str, str]],
        max_actions: int = 4,
    ) -> CommanderResult:
        fingerprint = self._fingerprint(phase_id, risks, reports)
        now = time.time()
        if (
            self._cached_result is not None
            and self._cached_fingerprint == fingerprint
            and (now - self._cached_at) < self._cache_ttl_seconds
        ):
            return self._cached_result

        if not self.enabled:
            result = CommanderResult(provider="fallback", actions=self._heuristic_actions(phase_id, risks, reports))
            self._update_cache(fingerprint, result)
            return result

        try:
            result = CommanderResult(provider="lava", actions=self._call_lava(phase_id, risks, reports, max_actions))
        except Exception as exc:
            result = CommanderResult(
                provider="fallback",
                actions=self._heuristic_actions(phase_id, risks, reports),
                error=str(exc),
            )
        self._update_cache(fingerprint, result)
        return result

    def _call_lava(
        self,
        phase_id: str,
        risks: Dict[str, int],
        reports: List[Dict[str, str]],
        max_actions: int,
    ) -> List[Dict[str, Any]]:
        url = f"{self._base_url}/chat/completions"
        body = {
            "model": self._model,
            "temperature": 0.2,
            "max_tokens": 500,
            "messages": [
                {"role": "system", "content": self._system_prompt()},
                {
                    "role": "user",
                    "content": self._user_prompt(phase_id=phase_id, risks=risks, reports=reports, max_actions=max_actions),
                },
            ],
        }

        req = request.Request(
            url=url,
            method="POST",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )

        try:
            with request.urlopen(req, timeout=self._timeout_seconds) as resp:
                raw = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Lava HTTP {exc.code}: {detail[:200]}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Lava network error: {exc.reason}") from exc

        data = json.loads(raw)
        content = self._extract_content(data)
        payload = self._extract_json_payload(content)
        actions_raw = payload.get("actions", [])
        if not isinstance(actions_raw, list):
            raise RuntimeError("Lava response missing 'actions' array")
        return self._normalize_actions(actions_raw, max_actions=max_actions)

    @staticmethod
    def _extract_content(data: Dict[str, Any]) -> str:
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("Lava response had no choices")
        msg = (choices[0] or {}).get("message") or {}
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [part.get("text", "") for part in content if isinstance(part, dict)]
            return "\n".join(parts)
        return str(content)

    @staticmethod
    def _extract_json_payload(text: str) -> Dict[str, Any]:
        text = (text or "").strip()
        if not text:
            raise RuntimeError("Lava response content was empty")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end <= start:
                raise RuntimeError("Lava response did not contain JSON")
            return json.loads(text[start : end + 1])

    @staticmethod
    def _normalize_actions(actions: List[Dict[str, Any]], max_actions: int) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for action in actions:
            if not isinstance(action, dict):
                continue
            priority = str(action.get("priority", "medium")).lower().strip()
            if priority not in PRIORITY_ORDER:
                priority = "medium"

            try:
                confidence = float(action.get("confidence", 0.6))
            except (TypeError, ValueError):
                confidence = 0.6
            confidence = max(0.0, min(1.0, confidence))

            verb = str(action.get("action", "")).strip() or "Assess zone condition"
            target = str(action.get("target", "")).strip() or "Venue floor"
            rationale = str(action.get("rationale", "")).strip() or "Model-selected best next step from current signals."

            normalized.append(
                {
                    "action": verb[:120],
                    "target": target[:80],
                    "priority": priority,
                    "confidence": round(confidence, 2),
                    "rationale": rationale[:220],
                }
            )

        normalized.sort(key=lambda item: (PRIORITY_ORDER[item["priority"]], -item["confidence"]))
        return normalized[: max(1, max_actions)]

    def _heuristic_actions(
        self,
        phase_id: str,
        risks: Dict[str, int],
        reports: List[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        if not risks:
            return []

        hottest = max(risks, key=risks.get)
        peak = risks[hottest]
        report_count = len(reports)

        actions: List[Dict[str, Any]] = [
            {
                "action": "Dispatch security team",
                "target": f"Section {hottest}",
                "priority": "critical" if peak >= 80 else "high",
                "confidence": 0.92 if peak >= 80 else 0.82,
                "rationale": f"Section {hottest} has peak risk score {peak}, highest in venue.",
            },
            {
                "action": "Broadcast calm reroute advisory",
                "target": f"Aisles near Section {hottest}",
                "priority": "high",
                "confidence": 0.78,
                "rationale": "Reduces counter-flow pressure and lowers panic in dense aisles.",
            },
        ]

        if peak >= 70:
            actions.append(
                {
                    "action": "Throttle ingress and open alternate egress",
                    "target": "Gate 3 -> Exit A corridor",
                    "priority": "critical",
                    "confidence": 0.86,
                    "rationale": "Immediate flow redistribution needed when localized density is severe.",
                }
            )

        if phase_id in {"organizer", "escalation"} or report_count >= 3:
            actions.append(
                {
                    "action": "Deploy medical response standby",
                    "target": f"Concourse adjacent to Section {hottest}",
                    "priority": "high" if peak < 85 else "critical",
                    "confidence": 0.74 if peak < 85 else 0.84,
                    "rationale": f"{report_count} attendee reports suggest elevated injury probability.",
                }
            )

        actions.sort(key=lambda item: (PRIORITY_ORDER[item["priority"]], -item["confidence"]))
        return actions[:4]

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are an Incident Commander AI for crowd safety operations. "
            "Output only strict JSON with an 'actions' array. "
            "Each action item must include: action, target, priority, confidence, rationale. "
            "priority must be one of critical/high/medium/low. "
            "confidence must be a number from 0 to 1. "
            "Prefer concrete operational commands with section/gate specificity."
        )

    @staticmethod
    def _user_prompt(
        phase_id: str,
        risks: Dict[str, int],
        reports: List[Dict[str, str]],
        max_actions: int,
    ) -> str:
        top_sections = sorted(risks.items(), key=lambda item: item[1], reverse=True)[:3]
        report_slice = reports[:6]
        return (
            "Build a prioritized incident action queue from this live venue state.\n"
            f"Phase: {phase_id}\n"
            f"Top risk sections: {top_sections}\n"
            f"Recent reports: {report_slice}\n"
            f"Return at most {max_actions} actions.\n"
            "Return JSON shape:\n"
            '{ "actions": [ { "action": "...", "target": "...", "priority": "high", "confidence": 0.84, "rationale": "..." } ] }'
        )

    @staticmethod
    def format_action_line(action: Dict[str, Any]) -> str:
        confidence_pct = int(round(float(action.get("confidence", 0.0)) * 100))
        priority = str(action.get("priority", "medium")).upper()
        verb = str(action.get("action", "Action"))
        target = str(action.get("target", "Venue"))
        rationale = str(action.get("rationale", ""))
        return f"[{priority} {confidence_pct}%] {verb} -> {target}. {rationale}"

    @staticmethod
    def _fingerprint(phase_id: str, risks: Dict[str, int], reports: List[Dict[str, str]]) -> str:
        key_payload = {
            "phase": phase_id,
            "risks": risks,
            "reports": reports[:8],
        }
        return json.dumps(key_payload, sort_keys=True, separators=(",", ":"))

    def _update_cache(self, fingerprint: str, result: CommanderResult) -> None:
        self._cached_fingerprint = fingerprint
        self._cached_result = result
        self._cached_at = time.time()
