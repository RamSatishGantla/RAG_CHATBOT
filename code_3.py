import pdfplumber
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from llama_cpp import Llama
import chromadb

# ---------------------------
# Initialize ChromaDB client (new API)
# ---------------------------
client = chromadb.Client()  # no Settings needed

# ---------------------------
# Initialize LLaMA model
# ---------------------------
llm = Llama(model_path="C:/Users/prudh/Mine/projects/rag_model/open-llama-3b-v2-wizard-evol-instuct-v2-196k.Q4_0.gguf")

# ---------------------------
# Step 1: Extract text from PDF
# ---------------------------
def extract_text_from_pdf(pdf_path: str) -> List[Tuple[int, str]]:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text()
            if page_text:
                pages.append((i, page_text))
    return pages

# ---------------------------
# Step 2: Chunking with overlap
# ---------------------------
def chunk_text_with_pages(
    pages: List[Tuple[int, str]],
    chunk_words: int = 220,
    overlap_words: int = 40
) -> List[Dict]:
    docs = []
    uid = 0
    for page_num, text in pages:
        words = text.split()
        step = max(1, chunk_words - overlap_words)
        for start in range(0, len(words), step):
            chunk = " ".join(words[start:start + chunk_words]).strip()
            if chunk:
                docs.append({
                    "id": f"doc-{uid}",
                    "text": chunk,
                    "page": page_num
                })
                uid += 1
    return docs

# ---------------------------
# Step 3: Build embeddings and store in ChromaDB
# ---------------------------
def build_chroma_collection(docs: List[Dict], collection_name="pdf_docs"):
    # Delete old collection if exists
    if collection_name in [c.name for c in client.list_collections()]:
        client.delete_collection(name=collection_name)

    collection = client.create_collection(name=collection_name, metadata={"description": "PDF chunks"})
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    for doc in docs:
        emb = model.encode([doc["text"]])[0].astype(np.float32)
        collection.add(
            documents=[doc["text"]],
            metadatas=[{"page": doc["page"], "id": doc["id"]}],
            ids=[doc["id"]],
            embeddings=[emb]
        )
    return collection

# ---------------------------
# Step 4: Query top chunk(s) using LLaMA safely
# ---------------------------
def query_with_llama(question: str, collection, model_name="all-MiniLM-L6-v2", top_k=1, max_tokens=256) -> str:
    # Get embeddings for the query
    model = SentenceTransformer(model_name)
    q_emb = model.encode([question])[0].astype(np.float32)
    
    results = collection.query(query_embeddings=[q_emb], n_results=top_k)
    answers = []

    for doc_text in results['documents'][0]:
        prompt = f"Answer the question based on the following context:\n{doc_text}\n\nQuestion: {question}\nAnswer concisely:"
        response = llm(prompt=prompt, max_tokens=max_tokens)
        text = response["choices"][0]["text"].strip()
        answers.append(text)

    return " ".join(answers)

# ---------------------------
# Step 5: Build Chroma store from PDF
# ---------------------------
def build_store_from_pdf(pdf_path: str):
    pages = extract_text_from_pdf(pdf_path)
    if not pages:
        raise ValueError("No extractable text found in PDF.")
    docs = chunk_text_with_pages(pages, chunk_words=220, overlap_words=40)
    collection = build_chroma_collection(docs)
    return collection

# ---------------------------
# Step 6: CLI Demo
# ---------------------------
if __name__ == "__main__":
    pdf_path = r"C:\Users\prudh\Mine\projects\rag_model\lsb.pdf"

    if not Path(pdf_path).exists():
        print(f"File not found: {pdf_path}")
        exit(1)

    print("🔧 Building ChromaDB index (pdfplumber + embeddings). This may take a moment...")
    collection = build_store_from_pdf(pdf_path)
    print("✅ Ready. Type your question (or 'exit').\n")

    while True:
        q = input("Q: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        # Only top 1 chunk for safety
        answer = query_with_llama(q, collection, top_k=1, max_tokens=256)
        print(f"\n💡 Answer:\n{answer}\n")
        print("-" * 60)
