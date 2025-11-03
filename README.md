# Offline RAG-Powered Chatbot for PDF Knowledge Extraction

## 📂 Project Files (in order)
1. **code_3.py** – Main Python script for running the RAG chatbot.  
2. **lsb.pdf** – Sample PDF document used for testing question-answering.  
3. **.gitignore** – Configuration file to exclude large model and installer files.  

---

## 🚀 Project Overview
This project implements an **offline Retrieval-Augmented Generation (RAG) chatbot** that can answer questions based on the contents of PDF documents.  
The chatbot works completely **offline**, ensuring **data privacy, security, and independence** from external APIs.  

**Workflow:**
1. Extract text from PDFs using `pdfplumber`.  
2. Split text into chunks and generate embeddings using `SentenceTransformers`.  
3. Store embeddings in **DuckDB** for efficient vector search.  
4. Retrieve the most relevant chunks for a given query.  
5. Use **LangChain** to pass the retrieved context into a locally run **LLM (LLaMA 3)**.  
6. Generate accurate, context-aware responses.  

---

## 🛠️ Technologies & Libraries Used
- **Python 3.10+**
- [LangChain](https://www.langchain.com/) – for building RAG pipelines  
- [pdfplumber](https://github.com/jsvine/pdfplumber) – for extracting text from PDFs  
- [DuckDB](https://duckdb.org/) – lightweight embedded database  
- [SentenceTransformers](https://www.sbert.net/) – for semantic vector embeddings  
- [Torch](https://pytorch.org/) – backend for deep learning  
- [Ollama](https://ollama.ai/) (or other LLaMA runtime) – for running the LLaMA model locally  

---

## 📦 Requirements
Install dependencies manually with `pip`, or just copy-paste the below block into your terminal:

```bash
pip install langchain
pip install pdfplumber
pip install duckdb
pip install sentence-transformers
pip install torch
