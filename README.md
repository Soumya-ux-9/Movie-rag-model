## Movie RAG Question Answering System (Offline / No API)

This project implements a **Retrieval-Augmented Generation (RAG)** based Question Answering system
that answers questions about movies using an **open-source LLM (FLAN-T5 Small)** without relying on any paid APIs.

---

## Project Overview

Large Language Models may hallucinate when answering factual questions.
To address this, this project uses a RAG pipeline, where relevant movie plot data
is retrieved from a vector database and provided as context to the LLM for accurate answer generation.

This implementation works **completely offline** once the model is downloaded.

---

## Tech Stack

- Python  
- LangChain  
- ChromaDB (Vector Database)  
- Sentence Transformers (Embeddings)  
- **FLAN-T5 Small (HuggingFace open-source LLM)**  

---

## 🔹 Architecture / Working

1. Movie plot data is loaded and split into smaller text chunks.
2. Each chunk is converted into embeddings using a sentence transformer model.
3. Embeddings are stored in **ChromaDB** for fast similarity search.
4. When a user asks a question:
   - The query is embedded
   - Most relevant movie chunks are retrieved
   - Retrieved context is passed to **FLAN-T5 Small**
5. The LLM generates a grounded answer based on retrieved content.

---

## 🔹 Folder Structure
movie_rag/
-app.py
- movies.txt
-chroma_db/

## On terminal 
─  cd movie_rag:
─ python app.py

