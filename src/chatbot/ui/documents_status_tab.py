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

    # Download Document
    st.markdown("---")
    st.markdown("### Download Document")
    all_sources = [item.source for item in statuses]
    
    if all_sources:
        download_col1, download_col2 = st.columns([3, 1])
        with download_col1:
            selected_download = st.selectbox(
                "Select document to download",
                options=all_sources,
                key="download_document_source",
            )
        with download_col2:
            st.write("")  # Add spacing
            if st.button("📥 Download", key="download_button"):
                file_content = knowledge_service.get_document_file_content(selected_download)
                if file_content:
                    st.download_button(
                        label="Save File",
                        data=file_content,
                        file_name=selected_download,
                        mime="text/plain",
                        key="download_file_button",
                    )
                else:
                    st.error(f"Failed to read: {selected_download}")
    else:
        st.info("No documents available to download.")

    # Delete Embedded Document
    st.markdown("---")
    st.markdown("### Delete Document Embeddings")
    embedded_sources = [item.source for item in statuses if item.embedded]
    if not embedded_sources:
        st.info("No embedded documents available to delete.")
    else:
        del_emb_col1, del_emb_col2 = st.columns([3, 1])
        with del_emb_col1:
            selected_source = st.selectbox(
                "Select document",
                options=embedded_sources,
                key="delete_embedded_source",
            )
        with del_emb_col2:
            st.write("")  # Add spacing
            if st.button("🗑️ Delete Embeddings", type="secondary", key="delete_selected_embedding"):
                deleted = knowledge_service.delete_document_embeddings(selected_source)
                if deleted:
                    st.success(f"Deleted embeddings for: {selected_source}")
                else:
                    st.warning(f"No embeddings found for: {selected_source}")
                st.rerun()

    # Delete Document File
    st.markdown("---")
    st.markdown("### Delete Document File")
    if all_sources:
        del_file_col1, del_file_col2 = st.columns([3, 1])
        with del_file_col1:
            selected_delete = st.selectbox(
                "Select document file to delete",
                options=all_sources,
                key="delete_document_file_source",
            )
        with del_file_col2:
            st.write("")  # Add spacing
            if st.button("🗑️ Delete File", type="primary", key="delete_document_file"):
                deleted = knowledge_service.delete_document_file(selected_delete)
                if deleted:
                    st.success(f"Deleted document: {selected_delete}")
                    st.rerun()
                else:
                    st.error(f"Failed to delete: {selected_delete}")
    else:
        st.info("No documents available to delete.")
