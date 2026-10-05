# Intelligent Document Q&A System (RAG & LLMs)

A Retrieval-Augmented Generation (RAG) system that answers questions about
any uploaded document using only the document's own content, grounding a
local LLM's answers in retrieved context instead of its general training
knowledge.

## Problem Statement

LLMs cannot answer questions about private or recent documents they were
never trained on, and tend to hallucinate when asked anyway. This project
solves that by retrieving the most relevant passages from a document
before generation, so the model answers from the document itself rather
than guessing.

## Approach

1. **Chunking** - The document is split into overlapping ~500-character
   chunks (50-character overlap) so no sentence is lost at a chunk
   boundary.
2. **Embeddings** - Each chunk is converted into a 384-dimension vector
   using the `all-MiniLM-L6-v2` Sentence Transformer, which captures
   semantic meaning rather than just keywords.
3. **Vector Search (FAISS)** - All chunk embeddings are indexed with
   FAISS for fast similarity search. A user's question is embedded the
   same way and matched against the index to retrieve the top-3 most
   relevant chunks.
4. **Grounded Generation** - The retrieved chunks are inserted into a
   prompt sent to a locally running Llama3 model (via Ollama), instructed
   to answer only from the provided context and say so explicitly if the
   answer isn't present.
5. **Streamlit UI** - A simple web interface lets a user upload a `.txt`
   document and ask questions in real time, with the retrieved source
   chunks shown alongside each answer for transparency.

## Results

Tested on a sample company HR policy document (7 sections: leave, WFH,
probation, reimbursement, code of conduct, resignation, appraisal) with 5
test questions spanning different sections. All 5 answers were correct
and exactly grounded in the document's stated figures (e.g. "24 days of
paid annual leave", "60 days notice period", "2 days per week" WFH),
confirming the retrieval step consistently surfaces the right section
before generation, and the LLM does not hallucinate when context is
provided.

## How to Run

```bash
pip install -r requirements.txt
ollama pull llama3
python rag_pipeline.py          # command-line test on the sample document
streamlit run app.py            # web UI - upload any .txt document
```

Requires [Ollama](https://ollama.com) installed and running locally, with
the `llama3` model pulled.

## Tech Stack

- Python, Sentence Transformers (`all-MiniLM-L6-v2`)
- FAISS (vector similarity search)
- Ollama + Llama3 (local LLM, no API cost, no data leaves the machine)
- Streamlit (web UI)
