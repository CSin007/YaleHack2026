from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Dict
from urllib import error, request

from dotenv import load_dotenv
import os

load_dotenv()

key = os.getenv("ELEVENLABS_API_KEY")
print("ELEVENLABS present:", repr(key))
print("ELEVENLABS bool:", bool((key or "").strip()))

@dataclass
class ElevenLabsResult:
    ok: bool
    audio_b64: str = ""
    mime_type: str = "audio/mpeg"
    error: str | None = None


class ElevenLabsTTSClient:
    def __init__(
        self,
        api_key: str | None,
        voice_map: Dict[str, str],
        model_id: str = "eleven_multilingual_v2",
        timeout_seconds: float = 12.0,
    ) -> None:
        self._api_key = (api_key or "").strip()
        self._voice_map = {k.lower(): v for k, v in (voice_map or {}).items() if v}
        self._model_id = model_id
        self._timeout_seconds = timeout_seconds

    @property
    def enabled(self) -> bool:
        return bool(self._api_key)

    def synthesize(self, text: str, language: str) -> ElevenLabsResult:
        if not self.enabled:
            return ElevenLabsResult(ok=False, error="ELEVENLABS_API_KEY not configured.")

        lang = (language or "en").lower()
        voice_id = self._voice_map.get(lang) or self._voice_map.get("default")
        if not voice_id:
            return ElevenLabsResult(ok=False, error=f"No ElevenLabs voice configured for '{lang}'.")

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        payload = {
            "text": text,
            "model_id": self._model_id,
            "voice_settings": {
                "stability": 0.40,
                "similarity_boost": 0.75,
            },
        }
        req = request.Request(
            url=url,
            method="POST",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "xi-api-key": self._api_key,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg",
            },
        )

        try:
            with request.urlopen(req, timeout=self._timeout_seconds) as resp:
                audio_bytes = resp.read()
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            return ElevenLabsResult(ok=False, error=f"ElevenLabs HTTP {exc.code}: {detail[:220]}")
        except error.URLError as exc:
            return ElevenLabsResult(ok=False, error=f"ElevenLabs network error: {exc.reason}")
        except Exception as exc:
            return ElevenLabsResult(ok=False, error=f"ElevenLabs error: {exc}")

        return ElevenLabsResult(ok=True, audio_b64=base64.b64encode(audio_bytes).decode("utf-8"))
