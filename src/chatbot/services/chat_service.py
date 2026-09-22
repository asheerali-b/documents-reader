from src.chatbot.clients.model_farm_client import ModelFarmClient
from src.chatbot.clients.ollama_client import OllamaClient
from src.chatbot.constant import (
    DEFAULT_REASONING_EFFORT,
    MODEL_FARM_PROVIDER,
    OLLAMA_PROVIDER,
)


class ChatService:
    def __init__(self, model_farm_client: ModelFarmClient, ollama_client: OllamaClient) -> None:
        self._model_farm_client = model_farm_client
        self._ollama_client = ollama_client

    def simple_chat(self, conversation: list[dict[str, str]], provider: str, model: str) -> str:
        return self._chat(messages=conversation, provider=provider, model=model)

    def chat_with_knowledge(
        self,
        user_prompt: str,
        knowledge_text: str,
        provider: str,
        model: str,
    ) -> str:
        system_prompt = (
            "You are a helpful assistant answering questions from retrieved documents. "
            "Use only the knowledge provided below. Each block starts with a 'Source:' "
            "field containing the exact document filename. Treat that filename as the "
            "document name for all facts in the same block. If the user asks which "
            "document contains a fact, identify the exact Source filename and explain "
            "the matching fact. Do not say the document name is unavailable when a "
            "matching Source field is present. If the answer is not present, say you "
            "do not know based on the provided knowledge.\n\n"
            f"Knowledge:\n{knowledge_text}"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self._chat(messages=messages, provider=provider, model=model)

    def _chat(self, messages: list[dict[str, str]], provider: str, model: str) -> str:
        selected_provider = provider.strip()
        if selected_provider == OLLAMA_PROVIDER:
            return self._ollama_client.chat_completion(messages=messages, model=model)

        if selected_provider == MODEL_FARM_PROVIDER:
            return self._model_farm_client.chat_completion(
                messages=messages,
                reasoning_effort=DEFAULT_REASONING_EFFORT,
                model=model,
            )

        raise ValueError(f"Unsupported provider: {provider}")
