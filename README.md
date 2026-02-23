

# Rag‑Document‑Assistant‑V2
### Rag‑Document‑Assistant‑V2 (FAISS & Embeddings)
**Rag‑Document‑Assistant‑V2** is an enhanced version of the RAG assistant that uses dense vector representations and FAISS for fast search over large document collections. It draws inspiration from the “RAG Search Assistant” project:contentReference[oaicite:8]{index=8} that combines a FLAN‑T5 generative model, Sentence Transformers and FAISS, and is optimized for CPU use.

## Key components

* **Document loading and segmentation** – Text files (PDF, TXT) are parsed, cleaned and split into chunks using a `RecursiveCharacterTextSplitter` with a length of 350 characters and an overlap of 50:contentReference[oaicite:9]{index=9}.
* **Embedding generation** – Instead of TF‑IDF, a Sentence Transformers model (e.g., `all‑MiniLM‑L6‑v2`) computes dense vector representations of the segments:contentReference[oaicite:10]{index=10}.
* **FAISS vector index** – The vectors are stored in a FAISS index so that search is fast and scalable; FAISS supports efficient nearest‑neighbor search even for large datasets:contentReference[oaicite:11]{index=11}.
* **Retrieval‑Augmented‑Generation** – For a given query the closest segments from FAISS are retrieved, combined into a prompt and fed to a generative model (e.g., `flan‑t5‑large`) to generate an answer:contentReference[oaicite:12]{index=12}.
* **Evaluation and interface** – The project provides metrics such as BLEU/ROUGE and embedding similarity for performance evaluation as well as an interactive Gradio UI for testing.

## Project structure
Rag‑Document‑Assistant-V2/ <br/>
├── app/ – Streamlit application (UI, document upload, question input) <br/>
├── backend/ – Functions for loading documents, vectorization and search <br/>
├── assets/ – Static assets (styles, icons) <br/>
├── deployment/ – Hosting configuration (Docker, Heroku, etc.) <br/>
├── tests/ – Unit tests for search logic <br/>
├── .streamlit/ – Streamlit environment configuration <br/>
├── requirements.txt – Python dependencies <br/> 
└── keep_alive.py – Script for keeping the service alive on free hosts <br/>

## Installation and usage

1. **Clone and install dependencies**  
   ```bash
   git clone https://github.com/JulijanaMilosavljevic/Rag-Document-Assistant-V2.git
   cd Rag-Document-Assistant-V2
   pip install -r requirements.txt
   ```
2. **Build the FAISS index** – In /*notebooks/*/ there is a /*build_index.ipynb*/ notebook that loads documents, generates embeddings and saves the FAISS index.
3. **Launch the interface** – Run the Gradio/Streamlit UI:
   ```bash
   python src/app.py
   ```
4. **Use** – In the interface upload your own documents or use the provided examples, enter a question and the system will search the FAISS index and generate an answer.

## Why V2?

Version V2 delivers much higher accuracy and scalability thanks to dense vectors and FAISS. Unlike the TF‑IDF approach, vectors from a Sentence Transformers model capture semantic similarity, enabling more precise retrieval of relevant context.
