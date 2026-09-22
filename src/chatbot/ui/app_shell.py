import streamlit as st

from src.chatbot.clients.model_farm_client import ModelFarmClient
from src.chatbot.clients.ollama_client import OllamaClient
from src.chatbot.config import load_config, load_ollama_config
from src.chatbot.constant import (
    MODEL_FARM_MODEL_OPTIONS,
    MODEL_FARM_PROVIDER,
    OLLAMA_MODEL_OPTIONS,
    OLLAMA_PROVIDER,
)
from src.chatbot.knowledge.knowledge_service import KnowledgeService
from src.chatbot.services.chat_service import ChatService
from src.chatbot.ui.knowledge_chat_tab import render_knowledge_chat_tab


@st.cache_resource
def _knowledge_service() -> KnowledgeService:
    return KnowledgeService()


def run_app() -> None:
    st.set_page_config(page_title="OTTO AI Assistant", page_icon=":speech_balloon:", layout="wide")
    st.title("OTTO AI Assistant")

    model_farm_config = load_config()
    ollama_config = load_ollama_config()

    model_farm_client = ModelFarmClient(config=model_farm_config)
    ollama_client = OllamaClient(config=ollama_config)
    chat_service = ChatService(model_farm_client=model_farm_client, ollama_client=ollama_client)
    knowledge_service = _knowledge_service()

    with st.sidebar:
        st.subheader("Model Selection")
        provider = st.selectbox(
            "Provider",
            options=[MODEL_FARM_PROVIDER, OLLAMA_PROVIDER],
            key="selected_llm_provider",
        )

        if provider == MODEL_FARM_PROVIDER:
            model_options = MODEL_FARM_MODEL_OPTIONS.copy()
            if model_farm_config.model not in model_options:
                model_options.insert(0, model_farm_config.model)
            model = st.selectbox(
                "Model",
                options=model_options,
                key="selected_model_farm_model",
            )
        else:
            model = st.selectbox(
                "Model",
                options=OLLAMA_MODEL_OPTIONS,
                key="selected_ollama_model",
            )

        st.caption(f"Using: {provider} / {model}")

    render_knowledge_chat_tab(
        chat_service,
        knowledge_service,
        provider=provider,
        model=model,
    )
