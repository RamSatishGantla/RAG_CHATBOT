import streamlit as st
import pdfplumber
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA

# -------------------
# Streamlit UI
# -------------------
st.set_page_config(page_title="Offline RAG Chatbot", layout="wide")
st.title("📄💬 Offline RAG-Powered Chatbot")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF to chat with", type=["pdf"])

if uploaded_file:
    with pdfplumber.open(uploaded_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_texts(chunks, embeddings)

    # Load local LLaMA model (adjust path to your .gguf file)
    llm = LlamaCpp(
        model_path="open-llama-3b-v2-wizard-evol-instuct-v2-196k.Q4_0.gguf",
        n_ctx=2048,
        temperature=0.7,
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(),
        chain_type="stuff",
    )

    # Chat UI
    st.subheader("Ask your PDF anything 👇")
    query = st.text_input("Your question:")

    if query:
        with st.spinner("Thinking..."):
            response = qa_chain.run(query)
        st.success(response)
