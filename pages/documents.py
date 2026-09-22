import streamlit as st

from src.chatbot.knowledge.knowledge_service import KnowledgeService
from src.chatbot.ui.documents_status_tab import render_documents_status_tab


@st.cache_resource
def _knowledge_service() -> KnowledgeService:
    return KnowledgeService()


st.set_page_config(page_title="Documents Status", page_icon=":file_folder:", layout="wide")
st.title("Documents Status")
render_documents_status_tab(_knowledge_service())