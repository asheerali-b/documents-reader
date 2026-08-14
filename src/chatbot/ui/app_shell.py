import streamlit as st

from src.chatbot.clients.model_farm_client import ModelFarmClient
from src.chatbot.config import load_config
from src.chatbot.services.chat_service import ChatService
from src.chatbot.ui.knowledge_chat_tab import render_knowledge_chat_tab
from src.chatbot.ui.simple_chat_tab import render_simple_chat_tab


def run_app() -> None:
    st.set_page_config(page_title="Streamlit Chatbot", page_icon=":speech_balloon:", layout="wide")
    st.title("Chatbot Application")

    config = load_config()
    client = ModelFarmClient(config=config)
    chat_service = ChatService(client=client)

    tab_simple, tab_knowledge = st.tabs(["Simple Chat", "Chat With Knowledge"])

    with tab_simple:
        render_simple_chat_tab(chat_service)

    with tab_knowledge:
        render_knowledge_chat_tab(chat_service)
