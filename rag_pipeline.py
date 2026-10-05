#################################################################################
#
#  Project Name  : Intelligent Document Q&A System (RAG & LLMs)
#  Description   : Core Retrieval-Augmented Generation (RAG) pipeline. Loads
#                   a text document, splits it into overlapping chunks,
#                   converts chunks into embeddings using Sentence
#                   Transformers, indexes them with FAISS for fast similarity
#                   search, retrieves the most relevant chunks for a user
#                   question, and sends them as context to a local Llama3
#                   model (via Ollama) to generate a grounded answer.
#  Date          : 04-Oct-2026
#  Author        : Shubham Shitole
#
#################################################################################

import os
import requests
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

Border = "-" * 50

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3


#################################################################################
#
# Function Name : load_document
# Input :         file_path
# Description :   Reads a plain text document from disk
# Return Value :  Full document text as a single string
# Date :          04-Oct-2026
# Author :        Shubham Shitole
#
#################################################################################

def load_document(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    print("Document loaded : ", file_path)
    print("Total characters : ", len(text))

    return text


#################################################################################
#
# Function Name : chunk_text
# Input :         text, chunk_size, chunk_overlap
# Description :   Splits a long document into smaller overlapping chunks.
#                 Overlap ensures a sentence split across a chunk boundary
#                 still has its full context in at least one chunk.
# Return Value :  List of text chunks
# Date :          04-Oct-2026
# Author :        Shubham Shitole
#
#################################################################################

def chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += (chunk_size - chunk_overlap)

    print("Chunks created : ", len(chunks))
    print("Average chunk length : ", int(np.mean([len(c) for c in chunks])))

    return chunks


#################################################################################
#
# Function Name : build_embeddings
# Input :         chunks, model
# Description :   Converts each text chunk into a dense numeric vector
#                 (embedding) that captures its semantic meaning, using a
#                 pretrained Sentence Transformer model
# Return Value :  numpy array of shape (num_chunks, embedding_dimension)
# Date :          04-Oct-2026
# Author :        Shubham Shitole
#
#################################################################################

def build_embeddings(chunks, model):
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)

    print("Embeddings shape : ", embeddings.shape)

    return embeddings.astype("float32")


#################################################################################
#
# Function Name : build_faiss_index
# Input :         embeddings
# Description :   Builds a FAISS index for fast similarity search over the
#                 chunk embeddings, using L2 (Euclidean) distance
# Return Value :  FAISS index object
# Date :          04-Oct-2026
# Author :        Shubham Shitole
#
#################################################################################

def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    print("FAISS index built with ", index.ntotal, " vectors of dimension ", dimension)

    return index


#################################################################################
#
# Function Name : retrieve_relevant_chunks
# Input :         question, model, index, chunks, top_k
# Description :   Embeds the user's question with the same Sentence
#                 Transformer model, then searches the FAISS index for the
#                 top_k chunks whose embeddings are closest to the question's
#                 embedding
# Return Value :  List of the top_k most relevant chunk strings
# Date :          04-Oct-2026
# Author :        Shubham Shitole
#
#################################################################################

def retrieve_relevant_chunks(question, model, index, chunks, top_k=TOP_K):
    question_embedding = model.encode([question], convert_to_numpy=True).astype("float32")

    distances, indices = index.search(question_embedding, top_k)

    retrieved_chunks = [chunks[i] for i in indices[0]]

    print("Retrieved ", len(retrieved_chunks), " chunks for question : ", question)
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
        print(f"   Rank {rank} | Chunk #{idx} | Distance {dist:.4f}")

    return retrieved_chunks


#################################################################################
#
# Function Name : build_prompt
# Input :         question, retrieved_chunks
# Description :   Constructs the final prompt sent to the LLM, instructing it
#                 to answer ONLY using the retrieved context, and to say so
#                 explicitly if the answer is not present in the context
# Return Value :  Prompt string
# Date :          04-Oct-2026
# Author :        Shubham Shitole
#
#################################################################################

def build_prompt(question, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)

    prompt = f"""You are a helpful assistant that answers questions using ONLY the
context provided below. If the answer is not present in the context,
say "I could not find this information in the document" instead of
guessing.

Context:
{context}

Question: {question}

Answer:"""

    return prompt


#################################################################################
#
# Function Name : generate_answer
# Input :         prompt, model_name
# Description :   Sends the prompt to a locally running Ollama server and
#                 returns the generated answer. Requires Ollama to be
#                 running in the background (`ollama serve`, usually
#                 automatic after install) and the model to be pulled
#                 (`ollama pull llama3`).
# Return Value :  Generated answer string
# Date :          04-Oct-2026
# Author :        Shubham Shitole
#
#################################################################################

def generate_answer(prompt, model_name=OLLAMA_MODEL):
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        answer = response.json()["response"].strip()
        return answer

    except requests.exceptions.ConnectionError:
        return ("ERROR: Could not connect to Ollama. Make sure Ollama is "
                "installed and running in the background.")

    except requests.exceptions.HTTPError:
        return (f"ERROR: Ollama responded with an error for model "
                f"'{model_name}' (likely not pulled yet). Run "
                f"'ollama pull {model_name}' and try again.")


#################################################################################
#
# Function Name : answer_question
# Input :         question, model, index, chunks
# Description :   End-to-end convenience function: retrieves relevant
#                 chunks for a question, builds the prompt, and generates
#                 the final answer
# Return Value :  answer string, retrieved_chunks list
# Date :          04-Oct-2026
# Author :        Shubham Shitole
#
#################################################################################

def answer_question(question, model, index, chunks):
    retrieved_chunks = retrieve_relevant_chunks(question, model, index, chunks)
    prompt = build_prompt(question, retrieved_chunks)
    answer = generate_answer(prompt)

    return answer, retrieved_chunks


#################################################################################
#
# Function Name : main
# Description :   Builds the RAG pipeline end-to-end on the sample document:
#                 load, chunk, embed, index, then answers a set of test
#                 questions from the command line
# Date :          04-Oct-2026
# Author :        Shubham Shitole
#
#################################################################################

def main():
    print(Border)
    print("Step 1 : Load Document")
    print(Border)

    text = load_document("sample_company_policy.txt")

    print(Border)
    print("Step 2 : Chunk Document")
    print(Border)

    chunks = chunk_text(text)

    print(Border)
    print("Step 3 : Load Embedding Model")
    print(Border)

    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Embedding model loaded : ", EMBEDDING_MODEL_NAME)

    print(Border)
    print("Step 4 : Build Embeddings")
    print(Border)

    embeddings = build_embeddings(chunks, embedding_model)

    print(Border)
    print("Step 5 : Build FAISS Index")
    print(Border)

    index = build_faiss_index(embeddings)

    print(Border)
    print("Step 6 : Answer Test Questions")
    print(Border)

    test_questions = [
        "How many days of annual leave do employees get?",
        "What is the notice period for confirmed employees?",
        "Can I work from home, and how many days?",
        "What is the probation period for new employees?",
        "How much internet reimbursement can I claim?"
    ]

    for question in test_questions:
        print("\n" + Border)
        print("Q: ", question)
        answer, _ = answer_question(question, embedding_model, index, chunks)
        print("A: ", answer)


if __name__ == "__main__":
    main()