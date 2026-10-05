#################################################################################
#
#  Project Name  : Intelligent Document Q&A System (RAG & LLMs) - Streamlit UI
#  Description   : Web interface for the RAG pipeline. Lets the user upload a
#                   text document, builds the embedding index once per
#                   document, then answers questions in real time using the
#                   retrieved context and a local Llama3 model via Ollama.
#  Date          : 04-Oct-2026
#  Author        : Shubham Shitole
#
#################################################################################

import streamlit as st
from sentence_transformers import SentenceTransformer

from rag_pipeline import (
    chunk_text,
    build_embeddings,
    build_faiss_index,
    answer_question,
    EMBEDDING_MODEL_NAME
)


#################################################################################
#
# Function Name : load_embedding_model
# Description :   Loads the Sentence Transformer model once and caches it
#                 across reruns, since reloading it on every interaction
#                 would be slow
# Return Value :  SentenceTransformer model
# Date :          04-Oct-2026
# Author :        Shubham Shitole
#
#################################################################################

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


#################################################################################
#
# Function Name : build_index_for_document
# Input :         text, model
# Description :   Chunks the uploaded document and builds a FAISS index for
#                 it. Cached per document so re-asking questions on the same
#                 document does not rebuild the index each time.
# Return Value :  chunks list, FAISS index
# Date :          04-Oct-2026
# Author :        Shubham Shitole
#
#################################################################################

@st.cache_resource(show_spinner="Building document index...")
def build_index_for_document(text, _model):
    chunks = chunk_text(text)
    embeddings = build_embeddings(chunks, _model)
    index = build_faiss_index(embeddings)

    return chunks, index


#################################################################################
#
# Function Name : main
# Description :   Streamlit app entry point: file upload, index building,
#                 question input, and displaying the generated answer along
#                 with the source chunks it was grounded in
# Date :          04-Oct-2026
# Author :        Shubham Shitole
#
#################################################################################

def main():
    st.set_page_config(page_title="Document Q&A (RAG)", page_icon="📄")
    st.title("📄 Intelligent Document Q&A System")
    st.caption("Retrieval-Augmented Generation using FAISS + Sentence Transformers + Llama3 (Ollama)")

    uploaded_file = st.file_uploader("Upload a text document (.txt)", type=["txt"])

    if uploaded_file is None:
        st.info("Upload a .txt document to get started.")
        return

    document_text = uploaded_file.read().decode("utf-8")
    st.success(f"Document loaded: {len(document_text)} characters")

    embedding_model = load_embedding_model()
    chunks, index = build_index_for_document(document_text, embedding_model)
    st.caption(f"Indexed {len(chunks)} chunks.")

    question = st.text_input("Ask a question about the document:")

    if question:
        with st.spinner("Searching document and generating answer..."):
            answer, retrieved_chunks = answer_question(question, embedding_model, index, chunks)

        st.subheader("Answer")
        st.write(answer)

        with st.expander("Show retrieved context (what the model used to answer)"):
            for i, chunk in enumerate(retrieved_chunks, start=1):
                st.markdown(f"**Chunk {i}:**")
                st.text(chunk)


if __name__ == "__main__":
    main()
