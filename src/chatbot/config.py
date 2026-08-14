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
