from __future__ import annotations

from typing import Any

import requests

from src.chatbot.config import OllamaConfig


class OllamaClient:
    def __init__(self, config: OllamaConfig) -> None:
        self._config = config

    def chat_completion(self, messages: list[dict[str, str]], model: str) -> str:
        response = requests.post(
            self._config.endpoint,
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "keep_alive": self._config.keep_alive,
            },
            timeout=self._config.timeout_seconds,
        )

        if not response.ok:
            details = _safe_error_text(response)
            raise RuntimeError(f"Ollama call failed ({response.status_code}): {details}")

        payload = response.json()
        message = payload.get("message", {})
        content = message.get("content", "")
        if isinstance(content, str):
            return content.strip() or "(Empty response)"
        return "(Unsupported response format)"


def _safe_error_text(response: requests.Response) -> str:
    try:
        return str(response.json())
    except ValueError:
        return response.text.strip() or "No details"
