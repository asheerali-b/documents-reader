import streamlit as st

from src.chatbot.services.chat_service import ChatService


def render_knowledge_chat_tab(chat_service: ChatService) -> None:
    st.subheader("Chat With Knowledge")
    st.caption("Provide your own knowledge text, then ask grounded questions.")

    knowledge_text = st.text_area(
        "Knowledge Base",
        height=220,
        placeholder="Paste documents, notes, or reference text here...",
    )

    question = st.text_input("Question", placeholder="Ask based on the knowledge above")

    if st.button("Ask Knowledge Bot", type="primary"):
        if not knowledge_text.strip():
            st.warning("Please provide knowledge text first.")
            return
        if not question.strip():
            st.warning("Please enter a question.")
            return

        with st.spinner("Generating answer..."):
            try:
                answer = chat_service.chat_with_knowledge(question, knowledge_text)
                st.success("Answer")
                st.write(answer)
            except Exception as ex:
                st.error(f"Error: {ex}")
