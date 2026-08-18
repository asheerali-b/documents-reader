import streamlit as st

from src.chatbot.constant import DOCUMENTS_DIR
from src.chatbot.knowledge.knowledge_service import KnowledgeService
from src.chatbot.services.chat_service import ChatService


def render_knowledge_chat_tab(
    chat_service: ChatService,
    knowledge_service: KnowledgeService,
    provider: str,
    model: str,
) -> None:
    st.subheader("Chat With Knowledge")
    st.caption("Ingest .txt files from the documents folder, then ask grounded questions.")

    st.info(f"Put your .txt files inside the '{DOCUMENTS_DIR}' folder and click Ingest Data.")

    if "knowledge_chat_history" not in st.session_state:
        st.session_state.knowledge_chat_history = []

    if "knowledge_stats" not in st.session_state:
        st.session_state.knowledge_stats = knowledge_service.get_index_stats()

    left_col, right_col = st.columns([1, 1])

    with left_col:
        if st.button("Ingest Data", type="primary", key="ingest_data_button"):
            with st.spinner("Building embeddings and storing in ChromaDB..."):
                try:
                    stats = knowledge_service.build_index()
                    st.session_state.knowledge_stats = stats
                    st.success(
                        "Ingestion complete. "
                        f"Total docs: {stats['documents_total']} | "
                        f"Embedded docs: {stats['documents_embedded']} | "
                        f"Ingested now: {stats['documents_ingested_now']} | "
                        f"Skipped unchanged: {stats['documents_skipped']} | "
                        f"Chunks: {stats['chunks']}"
                    )
                except Exception as ex:
                    st.error(f"Ingestion failed: {ex}")

    with right_col:
        status = "Ready" if knowledge_service.has_index() else "Not indexed"
        st.metric("Knowledge Index Status", status)

    stats = knowledge_service.get_index_stats()
    st.session_state.knowledge_stats = stats
    st.caption(
        f"Total docs: {stats['documents_total']} | Embedded docs: {stats['documents_embedded']} | "
        f"Not embedded: {stats['documents_not_embedded']} | Outdated: {stats['documents_outdated']} | "
        f"Total chunks: {stats['chunks']}"
    )

    for message in st.session_state.knowledge_chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask using ingested knowledge", key="knowledge_chat_input")
    if not question:
        return

    if not knowledge_service.has_index():
        st.warning("No index found. Click Ingest Data first.")
        return

    st.session_state.knowledge_chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching and generating answer..."):
            try:
                hits = knowledge_service.search(question)
                if not hits:
                    answer = "I could not find relevant information in the ingested documents."
                    sources = []
                else:
                    context = knowledge_service.context_from_hits(hits)
                    answer = chat_service.chat_with_knowledge(
                        question,
                        context,
                        provider=provider,
                        model=model,
                    )
                    sources = hits
                knowledge_service.log_query(question=question, hits=hits, answer=answer)
            except Exception as ex:
                answer = f"Error: {ex}"
                sources = []

        st.markdown(answer)

        if sources:
            with st.expander("Retrieved Context"):
                for hit in sources:
                    st.markdown(
                        f"Source: {hit.chunk.source} | Lines: {hit.chunk.start_line}-{hit.chunk.end_line} | "
                        f"Dense: {hit.dense_score:.3f} | Sparse: {hit.sparse_score:.3f} | Hybrid: {hit.score:.3f}"
                    )
                    st.write(hit.chunk.text)

    st.session_state.knowledge_chat_history.append({"role": "assistant", "content": answer})
