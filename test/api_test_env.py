import os

import requests
from dotenv import load_dotenv

load_dotenv()

api_key = str(os.getenv("MODEL_FARM_API_KEY"))

response = requests.post(
    "https://aoai-farm.bosch-temp.com/api/openai/deployments/gpt-5-nano-2025-08-07/chat/completions?api-version=2024-05-01-preview",
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    },
    json={
        # "model": "gpt-5-nano-2025-08-07",
        "model": "gemini-2.5-flash",
        # "model": "gemini-3.5-flash",
        "messages": [{"role": "user", "content": "Just say Hello! and tell me who won the fifa world cup 2022 and 2026 and who was the best player"}],
        "reasoning_effort": "low",  # Set to "low" to reduce thinking time, or "disabled" to disable completely
    }
)

print(response.text)