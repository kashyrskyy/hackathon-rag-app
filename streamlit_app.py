"""
Enhanced RAG Application with Google Gemini API
A powerful document Q&A system with web search capabilities
"""

# Fix for ChromaDB SQLite compatibility on Streamlit Cloud
import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

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
            # Show warning but don't stop the app
            st.warning("🔑 **Google API key not configured!** Add it to Streamlit secrets to enable AI responses.")
            st.info("Get your free API key at: https://makersuite.google.com/app/apikey")
            return None
        return api_key

def initialize_session_state():
    """Initialize session state variables"""
    # Core components
    if "vector_store" not in st.session_state:
        try:
            st.session_state.vector_store = VectorStore()
        except Exception as e:
            st.error(f"Failed to initialize vector store: {str(e)}")
            st.info("📄 App will run in web-search only mode")
            st.session_state.vector_store = None
    
    # Processing status
    if "documents_processed" not in st.session_state:
        st.session_state.documents_processed = False
    if "processing_status" not in st.session_state:
        st.session_state.processing_status = ""
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""
    if "last_response" not in st.session_state:
        st.session_state.last_response = ""
    
    # Chat and history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "query_count" not in st.session_state:
        st.session_state.query_count = 0
    
    # Document stats
    if "document_stats" not in st.session_state:
        st.session_state.document_stats = []

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
    
    # Initialize LLM client if API key is available
    llm_client = None
    if api_key:
        try:
            llm_client = GeminiClient(api_key)
        except Exception as e:
            st.error(f"Failed to initialize AI client: {str(e)}")
    else:
        st.sidebar.warning("🔑 Add Google API key to enable AI responses")
    
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
        doc_count = st.session_state.vector_store.get_collection_count() if st.session_state.vector_store else 0
        col_a, col_b = st.columns(2)
        col_a.metric("Documents", doc_count)
        col_b.metric("Queries", st.session_state.query_count)
        
        # Status indicators
        if st.session_state.documents_processed:
            st.success("✅ Documents Ready")
        else:
            st.info("📄 Upload documents to begin")
            
        if st.session_state.last_response:
            st.success("💬 Ready for questions")
        
        # Quick actions
        st.subheader("⚡ Quick Actions")
        
        if st.button("🗑️ Clear All Data"):
            # Clear all session state
            if st.session_state.vector_store:
                st.session_state.vector_store.clear_collection()
            st.session_state.documents_processed = False
            st.session_state.processing_status = ""
            st.session_state.last_query = ""
            st.session_state.last_response = ""
            st.session_state.chat_history = []
            st.session_state.query_count = 0
            st.session_state.document_stats = []
            st.success("🧹 All data cleared!")
            st.rerun()
    
    # Initialize LLM client
    llm_client = GeminiClient(api_key, selected_model)
    
    # Main content area - improved layout inspired by bio-app
    col1, col2 = st.columns([1, 2])  # Controls on left, results on right
    
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
                        if st.session_state.vector_store:
                            st.session_state.vector_store.add_documents(documents, all_chunks)
                        else:
                            st.warning("⚠️ Vector store not available - documents processed but search disabled")
                        st.session_state.documents_processed = True
                        st.session_state.processing_status = f"✅ Processed {len(documents)} documents with {sum(len(chunks) for chunks in all_chunks)} total chunks"
                        st.session_state.document_stats = [(filename, DocumentProcessor.get_document_stats(text)) for filename, text in documents]
                        
                        st.success(st.session_state.processing_status)
    
    with col2:
        # Show processing status if available
        if st.session_state.processing_status:
            st.info(st.session_state.processing_status)
        
        # Show document stats
        if st.session_state.document_stats:
            with st.expander("📊 Document Statistics", expanded=False):
                for filename, stats in st.session_state.document_stats:
                    st.markdown(f"**{filename}:**")
                    col_a, col_b, col_c, col_d = st.columns(4)
                    col_a.metric("Words", stats['words'])
                    col_b.metric("Characters", stats['characters'])
                    col_c.metric("Lines", stats['lines'])
                    col_d.metric("Est. Tokens", stats['estimated_tokens'])
        
        st.subheader("💬 Ask Questions")
        
        # Query input
        query = st.text_area(
            "What would you like to know?",
            height=80,
            placeholder="Ask anything about your uploaded documents...",
            value=st.session_state.last_query
        )
        
        if query and st.button("🔍 Get Answer"):
            doc_count = st.session_state.vector_store.get_collection_count() if st.session_state.vector_store else 0
            if not st.session_state.documents_processed and doc_count == 0 and not enable_web_search:
                st.warning("⚠️ Please upload documents or enable web search!")
            else:
                with st.spinner("🧠 Generating answer..."):
                    
                    # Search documents
                    doc_results = st.session_state.vector_store.search(query, num_doc_results) if st.session_state.vector_store else []
                    
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
                    if not llm_client:
                        response = "🔑 **API Key Required**: Please add your Google API key to Streamlit secrets:\n\n1. Click hamburger menu (☰) → Settings → Secrets\n2. Add: `GOOGLE_API_KEY = \"your_key_here\"`\n3. Save and restart\n\nGet your free API key at: https://makersuite.google.com/app/apikey"
                    else:
                        try:
                            response = llm_client.generate_response(prompt, temperature)
                            
                            if not response or response.strip() == "":
                                response = "⚠️ No response generated. Please check that your Google API key is properly configured in Streamlit secrets."
                                
                        except Exception as e:
                            error_msg = str(e)
                            if "api_key" in error_msg.lower() or "unauthorized" in error_msg.lower():
                                response = "🔑 **API Key Error**: Your Google API key may be invalid or expired.\n\nPlease check your API key in Streamlit secrets:\n1. Click hamburger menu (☰) → Settings → Secrets\n2. Verify: `GOOGLE_API_KEY = \"your_key_here\"`\n3. Get a new key at: https://makersuite.google.com/app/apikey"
                            else:
                                response = f"❌ **Error generating response**: {error_msg}\n\nPlease check your API key configuration."
                    
                    # Update session state
                    st.session_state.last_query = query
                    st.session_state.last_response = response
                    st.session_state.query_count += 1
                    
                    # Display results
                    st.markdown("### 🎯 Answer")
                    st.text_area("Generated Response", response, height=400)
                    
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
        
        # Show last response if available (even when not actively querying)
        if st.session_state.last_response and not query:
            st.markdown("### 💭 Last Response")
            st.text_area("Previous Answer", st.session_state.last_response, height=300)
            
            # Show query count
            if st.session_state.query_count > 0:
                st.caption(f"Total queries processed: {st.session_state.query_count}")
    
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