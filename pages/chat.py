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
from src.chatbot.services.chat_service import ChatService
from src.chatbot.ui.simple_chat_tab import render_simple_chat_tab


def _create_chat_service() -> ChatService:
    return ChatService(
        model_farm_client=ModelFarmClient(config=load_config()),
        ollama_client=OllamaClient(config=load_ollama_config()),
    )


def _render_model_selection() -> tuple[str, str]:
    model_farm_config = load_config()

    with st.sidebar:
        st.subheader("Model Selection")
        provider = st.selectbox(
            "Provider",
            options=[MODEL_FARM_PROVIDER, OLLAMA_PROVIDER],
            key="chat_page_provider",
        )

        if provider == MODEL_FARM_PROVIDER:
            model_options = MODEL_FARM_MODEL_OPTIONS.copy()
            if model_farm_config.model not in model_options:
                model_options.insert(0, model_farm_config.model)
            model = st.selectbox("Model", options=model_options, key="chat_page_model_farm_model")
        else:
            model = st.selectbox(
                "Model",
                options=OLLAMA_MODEL_OPTIONS,
                key="chat_page_ollama_model",
            )

        st.caption(f"Using: {provider} / {model}")

    return provider, model


st.set_page_config(page_title="OTTO AI Assistant", page_icon=":speech_balloon:", layout="wide")
st.title("OTTO AI Assistant")

selected_provider, selected_model = _render_model_selection()
render_simple_chat_tab(
    _create_chat_service(),
    provider=selected_provider,
    model=selected_model,
)