from __future__ import annotations

from typing import Any

import requests

from src.chatbot.config import ModelFarmConfig
from src.chatbot.constant import DEFAULT_REASONING_EFFORT


class ModelFarmClient:
    def __init__(self, config: ModelFarmConfig) -> None:
        self._config = config

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        model: str | None = None,
    ) -> str:
        if not self._config.api_key:
            raise ValueError("Missing MODEL_FARM_API_KEY in .env")

        response = requests.post(
            self._config.endpoint,
            params={"api-version": self._config.api_version},
            headers={
                "Authorization": f"Bearer {self._config.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model or self._config.model,
                "messages": messages,
                "reasoning_effort": reasoning_effort,
            },
            timeout=self._config.timeout_seconds,
        )

        if not response.ok:
            details = _safe_error_text(response)
            raise RuntimeError(f"API call failed ({response.status_code}): {details}")

        payload = response.json()
        return _extract_text(payload)


def _extract_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices", [])
    if not choices:
        return "No response choices were returned by the API."

    message = choices[0].get("message", {})
    content = message.get("content", "")

    if isinstance(content, str):
        return content.strip() or "(Empty response)"

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text_value = item.get("text")
                if isinstance(text_value, str):
                    parts.append(text_value)
        merged = "\n".join(parts).strip()
        return merged or "(Empty response)"

    return "(Unsupported response format)"


def _safe_error_text(response: requests.Response) -> str:
    try:
        return str(response.json())
    except ValueError:
        return response.text.strip() or "No details"
