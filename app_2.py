import streamlit as st
import pdfplumber
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import tempfile
import os
import time
from typing import List, Dict, Any
import json

# -------------------
# Page Configuration
# -------------------
st.set_page_config(
    page_title="Offline RAG Chatbot",
    page_icon="📄💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------
# Custom CSS
# -------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 700;
    }
    .stSidebar {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    .sidebar-title {
        font-size: 1.5rem;
        color: #1f4e79;
        margin-bottom: 1.5rem;
        font-weight: 600;
    }
    .sidebar-section {
        margin-bottom: 1.5rem;
    }
    .sidebar-subtitle {
        font-size: 1.1rem;
        color: #2c6b9e;
        margin-bottom: 0.8rem;
        font-weight: 600;
    }
    .stButton button {
        width: 100%;
        background: linear-gradient(to right, #2c6b9e, #1f4e79);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.8rem;
        border-radius: 8px;
        transition: all 0.3s;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .chat-container {
        background-color: #f8f9fa;
        border-radius: 12px;
        padding: 1.5rem;
        height: 600px;
        overflow-y: auto;
    }
    .user-message {
        background: linear-gradient(135deg, #2c6b9e 0%, #1f4e79 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .assistant-message {
        background-color: white;
        color: #333;
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        max-width: 80%;
        margin-right: auto;
        border: 1px solid #e9ecef;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
    }
    .message-timestamp {
        font-size: 0.7rem;
        color: #6c757d;
        text-align: right;
        margin-top: 0.5rem;
    }
    .processing-indicator {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 1rem;
        color: #2c6b9e;
        font-style: italic;
    }
    .feature-card {
        background-color: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        height: 100%;
        transition: transform 0.3s;
    }
    .feature-card:hover {
        transform: translateY(-5px);
    }
    .feature-icon {
        font-size: 2.5rem;
        color: #2c6b9e;
        margin-bottom: 1rem;
    }
    .feature-title {
        font-size: 1.2rem;
        color: #1f4e79;
        margin-bottom: 0.8rem;
        font-weight: 600;
    }
    .instructions {
        background-color: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    .footer {
        text-align: center;
        color: #6c757d;
        margin-top: 2rem;
        padding: 1rem;
        font-size: 0.9rem;
    }
    .file-upload-area {
        border: 2px dashed #2c6b9e;
        border-radius: 8px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 1rem;
        background-color: rgba(44, 107, 158, 0.05);
        transition: all 0.3s;
    }
    .file-upload-area:hover {
        background-color: rgba(44, 107, 158, 0.1);
    }
    .param-info {
        background-color: #e9f7fe;
        border-left: 4px solid #2c6b9e;
        padding: 0.8rem;
        border-radius: 4px;
        margin-top: 0.5rem;
        font-size: 0.85rem;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border-left: 4px solid #28a745;
    }
    .info-box {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border-left: 4px solid #17a2b8;
    }
</style>
""", unsafe_allow_html=True)

# -------------------
# Initialize session state
# -------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "processing" not in st.session_state:
    st.session_state.processing = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------
# Sidebar
# -------------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">⚙️ Settings & Configuration</div>', unsafe_allow_html=True)
    
    # Model parameters
    st.markdown('<div class="sidebar-subtitle">Model Configuration</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1, 
                               help="Controls randomness: Lower = more deterministic, Higher = more creative")
    with col2:
        max_tokens = st.slider("Max Tokens", 128, 4096, 1024, 128, 
                              help="Maximum length of the generated response")
    
    st.markdown("""
    <div class="param-info">
        <b>Temperature</b>: Lower values (0.2-0.5) give more focused answers, higher values (0.7-1.0) make output more creative.<br>
        <b>Max Tokens</b>: Controls response length. Higher values allow longer answers but use more memory.
    </div>
    """, unsafe_allow_html=True)
    
    # File upload
    st.markdown('<div class="sidebar-subtitle">Document Upload</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="file-upload-area">
        <p style="font-size: 3rem;">📄</p>
        <p><b>Drag & drop a PDF file here or click to browse</b></p>
        <p style="font-size: 0.9rem; color: #6c757d;">Supported: PDF documents</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type=["pdf"], label_visibility="collapsed")
    
    # Model path input
    st.markdown('<div class="sidebar-subtitle">Model Path</div>', unsafe_allow_html=True)
    model_path = st.text_input(
        "Path to your LLaMA model (.gguf file)", 
        value="open-llama-3b-v2-wizard-evol-instuct-v2-196k.Q4_0.gguf",
        label_visibility="collapsed"
    )
    
    # Process button
    process_btn = st.button("Process Document", use_container_width=True)
    
    # Clear chat button
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

# -------------------
# Main Content
# -------------------
st.markdown('<h1 class="main-header">📄💬 Offline RAG-Powered Chatbot</h1>', unsafe_allow_html=True)

# Document processing
if uploaded_file and process_btn:
    st.session_state.processing = True
    with st.spinner("Processing your document..."):
        try:
            # Extract text from PDF
            with pdfplumber.open(uploaded_file) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
            
            if not text.strip():
                st.error("Could not extract text from the PDF. Please try another file.")
            else:
                # Show processing progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Splitting text into chunks...")
                # Split text
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, 
                    chunk_overlap=200
                )
                chunks = text_splitter.split_text(text)
                progress_bar.progress(30)
                
                status_text.text("Creating embeddings...")
                # Create embeddings
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                progress_bar.progress(60)
                
                status_text.text("Building vector database...")
                # Create vector store
                with tempfile.TemporaryDirectory() as temp_dir:
                    vectordb = Chroma.from_texts(
                        chunks, 
                        embeddings, 
                        persist_directory=temp_dir
                    )
                    st.session_state.vector_db = vectordb
                progress_bar.progress(80)
                
                status_text.text("Initializing language model...")
                # Initialize LLM
                callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
                
                llm = LlamaCpp(
                    model_path=model_path,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n_ctx=2048,
                    top_p=1,
                    callback_manager=callback_manager,
                    verbose=False,
                )
                
                # Create QA chain
                st.session_state.qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=st.session_state.vector_db.as_retriever(),
                    chain_type="stuff",
                    return_source_documents=False
                )
                
                progress_bar.progress(100)
                status_text.text("")
                
                st.session_state.pdf_processed = True
                st.session_state.processing = False
                st.success("PDF processed successfully! You can now ask questions.")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.session_state.processing = False

# Display chat messages
if st.session_state.messages:
    st.markdown("### Conversation")
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="user-message">
                    <div><b>You</b></div>
                    <div>{message["content"]}</div>
                    <div class="message-timestamp">{message["timestamp"]}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="assistant-message">
                    <div><b>Assistant</b></div>
                    <div>{message["content"]}</div>
                    <div class="message-timestamp">{message["timestamp"]}</div>
                </div>
                """, unsafe_allow_html=True)
else:
    # Welcome message and instructions
    st.markdown("""
    <div class="info-box">
        <b>👈 Get Started</b> - Upload a PDF document and configure the settings in the sidebar to begin chatting with your document.
    </div>
    """, unsafe_allow_html=True)
    
    # Features section
    st.markdown("### Why Use Our RAG Chatbot?")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🔒</div>
            <div class="feature-title">Privacy First</div>
            <p>Your documents never leave your machine. All processing happens locally.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Document Insights</div>
            <p>Ask questions about your PDF and get accurate answers based on the content.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <div class="feature-title">Fast Responses</div>
            <p>Get quick answers without compromising on accuracy or privacy.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # How to use section
    st.markdown("### How to Use")
    st.markdown("""
    <div class="instructions">
        <ol>
            <li><b>Upload a PDF document</b> using the file uploader in the sidebar</li>
            <li><b>Configure model settings</b> (temperature and max tokens) if needed</li>
            <li><b>Click 'Process Document'</b> to analyze your PDF</li>
            <li><b>Start asking questions</b> about your document in the chat interface</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# Chat input
if st.session_state.pdf_processed and not st.session_state.processing:
    st.markdown("---")
    st.markdown("### Ask a Question")
    
    query = st.chat_input("Type your question here...")
    
    if query:
        # Add user message to chat history
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.messages.append({"role": "user", "content": query, "timestamp": timestamp})
        
        # Get assistant response
        with st.spinner("Thinking..."):
            try:
                # Get response from QA chain
                result = st.session_state.qa_chain({"query": query})
                response = result["result"]
                
                # Add assistant response to chat history
                timestamp = datetime.now().strftime("%H:%M:%S")
                st.session_state.messages.append({"role": "assistant", "content": response, "timestamp": timestamp})
                
                # Add to chat history for context
                st.session_state.chat_history.append((query, response))
                
                st.rerun()
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                timestamp = datetime.now().strftime("%H:%M:%S")
                st.session_state.messages.append({"role": "assistant", "content": error_msg, "timestamp": timestamp})
                st.rerun()

elif not uploaded_file:
    # Show welcome state
    pass
else:
    if st.session_state.processing:
        st.info("Processing your document. Please wait...")
    else:
        st.warning("Please click 'Process Document' to analyze your PDF and enable the chat.")

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>Built with Streamlit, LangChain, and LLaMA.cpp • All processing happens offline on your machine</p>
</div>
""", unsafe_allow_html=True)