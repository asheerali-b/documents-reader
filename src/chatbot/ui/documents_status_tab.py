import streamlit as st

from src.chatbot.knowledge.knowledge_service import KnowledgeService


def render_documents_status_tab(knowledge_service: KnowledgeService) -> None:
    st.subheader("Documents Status")
    st.caption("Shows files in the documents folder and whether they are embedded in the index.")

    refresh_col, _ = st.columns([1, 4])
    with refresh_col:
        if st.button("Refresh", key="documents_status_refresh"):
            st.rerun()

    statuses = knowledge_service.get_document_statuses()

    if not statuses:
        st.info("No .txt files found in documents folder.")
        return

    rows = []
    for item in statuses:
        rows.append(
            {
                "Document": item.source,
                "Embedded": "Yes" if item.embedded else "No",
                "Up To Date": "Yes" if item.up_to_date else "No",
                "Chunk Count": item.chunk_count,
            }
        )

    st.dataframe(rows, use_container_width=True)

    embedded_sources = [item.source for item in statuses if item.embedded]
    if not embedded_sources:
        st.info("No embedded documents available to delete.")
        return

    st.markdown("Delete Embedded Document")
    selected_source = st.selectbox(
        "Select document",
        options=embedded_sources,
        key="delete_embedded_source",
    )
    if st.button("Delete Selected Embedding", type="secondary", key="delete_selected_embedding"):
        deleted = knowledge_service.delete_document_embeddings(selected_source)
        if deleted:
            st.success(f"Deleted embeddings for: {selected_source}")
        else:
            st.warning(f"No embeddings found for: {selected_source}")
        st.rerun()
