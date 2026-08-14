from src.chatbot.clients.model_farm_client import ModelFarmClient
from src.chatbot.constant import DEFAULT_REASONING_EFFORT


class ChatService:
    def __init__(self, client: ModelFarmClient) -> None:
        self._client = client

    def simple_chat(self, user_prompt: str) -> str:
        messages = [{"role": "user", "content": user_prompt}]
        return self._client.chat_completion(messages=messages, reasoning_effort=DEFAULT_REASONING_EFFORT)

    def chat_with_knowledge(self, user_prompt: str, knowledge_text: str) -> str:
        system_prompt = (
            "You are a helpful assistant. Use only the knowledge provided below. "
            "If the answer is not present, say you do not know based on the provided knowledge.\n\n"
            f"Knowledge:\n{knowledge_text}"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self._client.chat_completion(messages=messages, reasoning_effort=DEFAULT_REASONING_EFFORT)
