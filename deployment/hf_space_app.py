import streamlit as st
from backend.rag_pipeline import RagPipeline

st.set_page_config(page_title="RAG Document Assistant", page_icon="🔎")

st.title("🔎 AI Document Search Assistant")

rag = RagPipeline()

files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if st.button("Index Documents"):
    if files:
        rag.build_index(files)
        st.success("Index created!")
    else:
        st.warning("Upload at least one PDF.")

question = st.text_input("Enter your question")

if st.button("Ask"):
    if not rag.is_ready:
        st.error("Index not ready.")
    else:
        answer, sources = rag.answer(question)
        st.write("### Answer")
        st.success(answer)

        st.write("### Relevant Passages")
        for s in sources:
            st.info(f"**{s['title']}**\n\n{s['snippet']}")
