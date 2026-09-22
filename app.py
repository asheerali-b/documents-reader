import streamlit as st

from src.chatbot.ui.app_shell import run_app

main_page = st.Page(
    run_app,
    title="OTTO AI Assistant",
    default=True,
)
chat_page = st.Page(
    "pages/chat.py",
    title="Chat",
    url_path="chat",
    visibility="hidden",
)
documents_page = st.Page(
    "pages/documents.py",
    title="Documents Status",
    url_path="documents",
)

selected_page = st.navigation([main_page, chat_page, documents_page])
selected_page.run()
