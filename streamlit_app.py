"""
Enhanced RAG Application with Google Gemini API
A powerful document Q&A system with web search capabilities
"""
import streamlit as st
import os
from typing import List, Dict
import time

# Import custom utilities
from utils.llm_client import GeminiClient
from utils.document_processor import DocumentProcessor
from utils.vector_store import VectorStore
from utils.web_search import WebSearcher

# Configure page
st.set_page_config(
    page_title="🧠 Enhanced RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton > button {
        width: 100%;
        margin-top: 1rem;
    }
    .success-message {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def get_api_key() -> str:
    """Get Google API key from secrets or environment"""
    try:
        # Try Streamlit secrets first (for deployment)
        return st.secrets["GOOGLE_API_KEY"]
    except:
        # Fallback to environment variable
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error("🔑 Google API key not found! Please add GOOGLE_API_KEY to your Streamlit secrets or environment variables.")
            st.info("Get your free API key at: https://makersuite.google.com/app/apikey")
            st.stop()
        return api_key

def initialize_session_state():
    """Initialize session state variables"""
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = VectorStore()
    if "documents_processed" not in st.session_state:
        st.session_state.documents_processed = False
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

def main():
    """Main application function"""
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🧠 Enhanced RAG Assistant</h1>', unsafe_allow_html=True)
    st.markdown("### 📚 Upload documents, ask questions, and get AI-powered answers with web search enhancement!")
    
    # Get API key
    api_key = get_api_key()
    
    # Initialize services
    web_searcher = WebSearcher(st.secrets.get("SERP_API_KEY", None) if "SERP_API_KEY" in st.secrets else None)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        available_models = GeminiClient.get_available_models()
        selected_model = st.selectbox(
            "🤖 Select Model",
            options=list(available_models.keys()),
            format_func=lambda x: f"{x} - {available_models[x]}",
            index=0
        )
        
        # Temperature control
        temperature = st.slider(
            "🌡️ Response Creativity (Temperature)",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Lower values = more focused, Higher values = more creative"
        )
        
        # Perspective and audience settings
        st.subheader("🎭 Response Style")
        perspective = st.selectbox(
            "👤 Perspective",
            ["scientist", "engineer", "teacher", "policy expert", "AI researcher", "environmentalist", "physician", "consultant", "analyst"],
            index=0
        )
        
        audience = st.selectbox(
            "🎯 Target Audience", 
            ["students", "teachers", "laypeople", "policy makers", "researchers", "community members", "professionals", "executives"],
            index=0
        )
        
        # Search settings
        st.subheader("🔍 Search Settings")
        enable_web_search = st.checkbox("Enable Web Search", value=True, help="Enhance answers with real-time web information")
        num_doc_results = st.slider("Document Results", 3, 10, 5)
        num_web_results = st.slider("Web Results", 1, 5, 2)
        
        # Vector store status
        st.subheader("📊 Status")
        doc_count = st.session_state.vector_store.get_collection_count()
        st.metric("Documents Stored", doc_count)
        
        if st.button("🗑️ Clear All Documents"):
            st.session_state.vector_store.clear_collection()
            st.session_state.documents_processed = False
            st.session_state.chat_history = []
            st.rerun()
    
    # Initialize LLM client
    llm_client = GeminiClient(api_key, selected_model)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📄 Document Upload")
        
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload up to 10 PDF files (max 200MB each)"
        )
        
        if uploaded_files and st.button("🚀 Process Documents"):
            if len(uploaded_files) > 10:
                st.warning("⚠️ Please upload maximum 10 files")
            else:
                with st.spinner("Processing documents..."):
                    # Process PDFs
                    documents = DocumentProcessor.process_multiple_pdfs(uploaded_files)
                    
                    if documents:
                        # Create chunks for each document
                        all_chunks = []
                        for filename, text in documents:
                            chunks = DocumentProcessor.chunk_text(text)
                            all_chunks.append(chunks)
                            
                            # Show document stats
                            stats = DocumentProcessor.get_document_stats(text)
                            with st.expander(f"📊 {filename} Stats"):
                                col_a, col_b, col_c = st.columns(3)
                                col_a.metric("Words", stats['words'])
                                col_b.metric("Characters", stats['characters'])
                                col_c.metric("Est. Tokens", stats['estimated_tokens'])
                        
                        # Add to vector store
                        st.session_state.vector_store.add_documents(documents, all_chunks)
                        st.session_state.documents_processed = True
                        
                        st.success(f"✅ Successfully processed {len(documents)} documents!")
    
    with col2:
        st.subheader("💬 Ask Questions")
        
        # Query input
        query = st.text_area(
            "What would you like to know?",
            height=100,
            placeholder="Ask anything about your uploaded documents..."
        )
        
        if query and st.button("🔍 Get Answer"):
            if not st.session_state.documents_processed and st.session_state.vector_store.get_collection_count() == 0:
                st.warning("⚠️ Please upload and process documents first!")
            else:
                with st.spinner("🧠 Generating answer..."):
                    
                    # Search documents
                    doc_results = st.session_state.vector_store.search(query, num_doc_results)
                    
                    # Search web if enabled
                    web_results = []
                    if enable_web_search:
                        web_results = web_searcher.search_web(query, num_web_results)
                    
                    # Prepare context
                    context_parts = []
                    
                    # Add document context
                    if doc_results:
                        context_parts.append("=== DOCUMENT CONTEXT ===")
                        for i, result in enumerate(doc_results, 1):
                            source = result['metadata'].get('source', 'Unknown')
                            context_parts.append(f"Source {i} ({source}):")
                            context_parts.append(result['content'])
                            context_parts.append("")
                    
                    # Add web context
                    if web_results:
                        context_parts.append("=== WEB SEARCH CONTEXT ===")
                        for i, result in enumerate(web_results, 1):
                            context_parts.append(f"Web Result {i}: {result['title']}")
                            context_parts.append(result['snippet'])
                            context_parts.append("")
                    
                    context = "\n".join(context_parts)
                    
                    # Create prompt
                    prompt = f"""You are responding as a {perspective}, and your response must be tailored for an audience of {audience}.

### Instructions:
- Use tone, vocabulary, and examples suitable for {audience}
- Explain in the way a {perspective} would — with clarity, depth, and reasoning appropriate to the role
- Avoid jargon unless necessary, and define any technical terms
- Use the following context as background, not as quoted text
- If the context doesn't contain relevant information, say so clearly
- Integrate web search results naturally when available

### Context:
{context}

### Question:
{query}

### Answer (as a {perspective} to {audience}):"""
                    
                    # Generate response
                    response = llm_client.generate_response(prompt, temperature)
                    
                    # Display results
                    st.markdown("### 🎯 Answer")
                    st.markdown(response)
                    
                    # Show sources
                    if doc_results or web_results:
                        with st.expander("📚 Sources Used"):
                            if doc_results:
                                st.markdown("**Document Sources:**")
                                for result in doc_results:
                                    source = result['metadata'].get('source', 'Unknown')
                                    st.markdown(f"- {source}")
                            
                            if web_results:
                                st.markdown("**Web Sources:**")
                                for result in web_results:
                                    if result['url']:
                                        st.markdown(f"- [{result['title']}]({result['url']})")
                                    else:
                                        st.markdown(f"- {result['title']}")
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "query": query,
                        "response": response,
                        "timestamp": time.time()
                    })
    
    # Chat History
    if st.session_state.chat_history:
        st.subheader("💭 Recent Conversations")
        for i, chat in enumerate(reversed(st.session_state.chat_history[-5:]), 1):
            with st.expander(f"Q{i}: {chat['query'][:100]}..."):
                st.markdown(f"**Question:** {chat['query']}")
                st.markdown(f"**Answer:** {chat['response']}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🚀 Built for the Hackathon | Powered by Google Gemini & Streamlit Community Cloud</p>
        <p>💡 Upload PDFs, ask questions, get intelligent answers with web search enhancement!</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()