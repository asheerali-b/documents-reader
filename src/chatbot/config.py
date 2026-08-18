import os
from dataclasses import dataclass

from dotenv import load_dotenv

from src.chatbot.constant import DEFAULT_MODEL_NAME


@dataclass(frozen=True)
class ModelFarmConfig:
    api_key: str
    endpoint: str
    api_version: str
    model: str
    timeout_seconds: int = 60


@dataclass(frozen=True)
class OllamaConfig:
    endpoint: str
    keep_alive: str
    timeout_seconds: int = 120


def load_config() -> ModelFarmConfig:
    load_dotenv()

    api_key = os.getenv("MODEL_FARM_API_KEY", "").strip()
    endpoint = os.getenv(
        "MODEL_FARM_ENDPOINT",
        "https://aoai-farm.bosch-temp.com/api/openai/deployments/gpt-5-nano-2025-08-07/chat/completions",
    ).strip()
    api_version = os.getenv("MODEL_FARM_API_VERSION", "2024-05-01-preview").strip()
    model = os.getenv("MODEL_FARM_MODEL", DEFAULT_MODEL_NAME).strip()

    return ModelFarmConfig(
        api_key=api_key,
        endpoint=endpoint,
        api_version=api_version,
        model=model,
    )


def load_ollama_config() -> OllamaConfig:
    load_dotenv()

    endpoint = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434/api/chat").strip()
    keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "30m").strip()
    timeout_seconds = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120").strip())

    return OllamaConfig(
        endpoint=endpoint,
        keep_alive=keep_alive,
        timeout_seconds=timeout_seconds,
    )
