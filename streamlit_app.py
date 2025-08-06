"""
Enhanced RAG Assistant - Standalone Version
A comprehensive RAG (Retrieval Augmented Generation) application with web search enhancement
"""

# SQLite fix for ChromaDB on Streamlit Cloud
import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

# Core imports
import streamlit as st
import os
import PyPDF2
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import chromadb
import requests
from bs4 import BeautifulSoup
import time
import uuid
import numpy as np
from typing import List, Dict, Optional, Tuple
import re

# Configure page
st.set_page_config(
    page_title="🧠 Enhanced RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 2px solid #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ===== UTILITY CLASSES =====

class GeminiClient:
    """Client for Google Gemini API"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)
        
    def generate_response(
        self, 
        prompt: str, 
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> str:
        try:
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config,
                stream=stream
            )
            
            if stream:
                return response
            else:
                return response.text
                
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return f"Error: Unable to generate response. {str(e)}"
    
    @staticmethod
    def get_available_models() -> Dict[str, str]:
        return {
            "gemini-1.5-flash": "Fast and efficient for most tasks",
            "gemini-1.5-pro": "Most capable model for complex tasks",
            "gemini-2.0-flash": "Latest generation model with improved performance"
        }

class DocumentProcessor:
    """Handles PDF processing and text chunking"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_file) -> str:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start += (chunk_size - overlap)
        return chunks
    
    @staticmethod
    def get_document_stats(text: str) -> Dict[str, int]:
        return {
            'characters': len(text),
            'words': len(text.split()),
            'lines': len(text.splitlines()),
            'estimated_tokens': len(text) // 4
        }

class WebSearcher:
    """Handles web search functionality"""
    
    def __init__(self, serp_api_key: Optional[str] = None):
        self.serp_api_key = serp_api_key
        try:
            from duckduckgo_search import DDGS
            self.ddgs_available = True
        except ImportError:
            self.ddgs_available = False
            st.warning("⚠️ duckduckgo-search not installed. Web search will use fallback method.")
    
    def search_web(self, query: str, num_results: int = 3) -> List[Dict[str, str]]:
        if self.serp_api_key:
            return self._search_with_serpapi(query, num_results)
        elif self.ddgs_available:
            return self._search_with_ddgs_library(query, num_results)
        else:
            return self._search_with_duckduckgo_api(query, num_results)
    
    def _search_with_ddgs_library(self, query: str, num_results: int) -> List[Dict[str, str]]:
        try:
            from duckduckgo_search import DDGS
            results = []
            time.sleep(1)  # Rate limiting
            with DDGS() as ddgs:
                search_results = ddgs.text(query, max_results=num_results)
                for result in search_results:
                    results.append({
                        "title": result.get("title", ""),
                        "snippet": result.get("body", "") or result.get("content", ""),
                        "url": result.get("href", "")
                    })
            return results if results else self._create_fallback_result(query)
        except Exception as e:
            error_msg = str(e)
            if "ratelimit" in error_msg.lower() or "202" in error_msg:
                st.warning("🚫 Web search temporarily rate limited. Using knowledge-based response...")
                return self._create_fallback_result(query)
            else:
                st.warning(f"DDGS search failed: {error_msg[:100]}...")
                return self._search_with_duckduckgo_api(query, num_results)
    
    def _search_with_duckduckgo_api(self, query: str, num_results: int) -> List[Dict[str, str]]:
        try:
            url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1&skip_disambig=1"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            results = []
            for item in data.get("RelatedTopics", [])[:num_results]:
                if "Text" in item and "FirstURL" in item:
                    results.append({
                        "title": item.get("Text", "")[:100],
                        "snippet": item.get("Text", ""),
                        "url": item.get("FirstURL", "")
                    })
            
            return results if results else self._create_fallback_result(query)
        except Exception as e:
            st.warning(f"Web search failed: {str(e)}")
            return self._create_fallback_result(query)
    
    def _create_fallback_result(self, query: str) -> List[Dict[str, str]]:
        return [{
            "title": f"Knowledge-Based Response for: {query}",
            "snippet": f"Web search is temporarily unavailable due to rate limiting. The AI will provide a response based on its training knowledge and any uploaded documents.",
            "url": ""
        }]

class VectorStore:
    """Manages vector storage and retrieval with fallback to in-memory storage"""
    
    def __init__(self, collection_name: str = "rag_documents"):
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.embedding_model = self._load_embedding_model()
        
        # Fallback in-memory storage
        self.fallback_documents = []
        self.use_fallback = False
        
        self._initialize_client()
    
    @st.cache_resource
    def _load_embedding_model(_self):
        try:
            return SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            st.error(f"Error loading embedding model: {str(e)}")
            return None
    
    def _initialize_client(self):
        try:
            # Try different ChromaDB configurations
            try:
                import chromadb.config
                settings = chromadb.config.Settings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=None,
                    anonymized_telemetry=False
                )
                self.client = chromadb.Client(settings)
                self.collection = self.client.get_or_create_collection(name=self.collection_name)
                st.success("✅ Vector store initialized (in-memory mode)")
                return
            except Exception:
                pass
            
            # Try default client
            try:
                self.client = chromadb.Client()
                self.collection = self.client.get_or_create_collection(name=self.collection_name)
                st.success("✅ Vector store initialized successfully")
                return
            except Exception:
                pass
            
            raise Exception("All ChromaDB initialization methods failed")
            
        except Exception:
            st.warning("⚠️ ChromaDB failed - using in-memory vector storage")
            st.info("📄 Document search will work in memory (data won't persist between sessions)")
            self.client = None
            self.collection = None
            self.use_fallback = True
    
    def add_documents(self, documents: List[Tuple[str, str]], chunks_per_doc: List[List[str]]):
        if not self.embedding_model:
            st.error("Embedding model not loaded - document search disabled")
            return
        
        all_chunks = []
        all_metadatas = []
        all_ids = []
        
        for (filename, full_text), chunks in zip(documents, chunks_per_doc):
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadatas.append({
                    "source": filename,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                })
                all_ids.append(f"{filename}_{i}_{uuid.uuid4().hex[:8]}")
        
        if all_chunks:
            try:
                with st.spinner("🧠 Generating embeddings..."):
                    embeddings = self.embedding_model.encode(all_chunks).tolist()
                
                if self.use_fallback:
                    # Store in fallback storage
                    for i, chunk in enumerate(all_chunks):
                        self.fallback_documents.append({
                            'id': all_ids[i],
                            'content': chunk,
                            'metadata': all_metadatas[i],
                            'embedding': embeddings[i]
                        })
                    st.success(f"✅ Added {len(all_chunks)} chunks to in-memory vector store")
                else:
                    # Try ChromaDB
                    try:
                        self.collection.add(
                            documents=all_chunks,
                            metadatas=all_metadatas,
                            ids=all_ids,
                            embeddings=embeddings
                        )
                        st.success(f"✅ Added {len(all_chunks)} chunks to vector store")
                    except Exception as e:
                        error_msg = str(e)
                        if "no such table" in error_msg.lower():
                            # Switch to fallback
                            st.warning("⚠️ ChromaDB table error - switching to in-memory storage")
                            self.use_fallback = True
                            self.client = None
                            self.collection = None
                            
                            # Store in fallback
                            for i, chunk in enumerate(all_chunks):
                                self.fallback_documents.append({
                                    'id': all_ids[i],
                                    'content': chunk,
                                    'metadata': all_metadatas[i],
                                    'embedding': embeddings[i]
                                })
                            st.success(f"✅ Added {len(all_chunks)} chunks to in-memory vector store")
                        else:
                            st.error(f"Error adding documents: {error_msg}")
            except Exception as e:
                st.error(f"Error processing documents: {str(e)}")
    
    def search(self, query: str, n_results: int = 5) -> List[dict]:
        if not self.embedding_model:
            st.warning("Embedding model not available")
            return []
        
        try:
            query_embedding = self.embedding_model.encode([query]).tolist()[0]
            
            if self.use_fallback:
                if not self.fallback_documents:
                    return []
                
                # Calculate cosine similarities
                similarities = []
                for doc in self.fallback_documents:
                    dot_product = np.dot(query_embedding, doc['embedding'])
                    norm_a = np.linalg.norm(query_embedding)
                    norm_b = np.linalg.norm(doc['embedding'])
                    similarity = dot_product / (norm_a * norm_b)
                    similarities.append((similarity, doc))
                
                # Sort and return top results
                similarities.sort(key=lambda x: x[0], reverse=True)
                top_results = similarities[:n_results]
                
                return [{
                    "content": doc['content'],
                    "metadata": doc['metadata'],
                    "similarity": float(similarity)
                } for similarity, doc in top_results]
            else:
                # Use ChromaDB
                if not self.collection:
                    return []
                
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    include=["documents", "metadatas", "distances"]
                )
                
                formatted_results = []
                if results["documents"] and results["documents"][0]:
                    for i, doc in enumerate(results["documents"][0]):
                        formatted_results.append({
                            "content": doc,
                            "metadata": results["metadatas"][0][i],
                            "distance": results["distances"][0][i]
                        })
                return formatted_results
        except Exception as e:
            st.error(f"Error searching: {str(e)}")
            return []
    
    def clear_collection(self):
        if self.use_fallback:
            self.fallback_documents.clear()
            st.success("🗑️ In-memory vector store cleared")
            return
        
        if not self.client or not self.collection:
            st.warning("Vector store not available")
            return
        
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
            st.success("🗑️ Vector store cleared")
        except Exception as e:
            st.error(f"Error clearing vector store: {str(e)}")
    
    def get_collection_count(self) -> int:
        if self.use_fallback:
            return len(self.fallback_documents)
        
        if not self.collection:
            return 0
        try:
            return self.collection.count()
        except:
            return 0

# ===== HELPER FUNCTIONS =====

def get_api_key() -> str:
    """Get Google API key from secrets or environment"""
    try:
        return st.secrets["GOOGLE_API_KEY"]
    except:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.warning("🔑 **Google API key not configured!** Add it to Streamlit secrets to enable AI responses.")
            st.info("Get your free API key at: https://makersuite.google.com/app/apikey")
            return None
        return api_key

def initialize_session_state():
    """Initialize session state variables"""
    if "vector_store" not in st.session_state:
        try:
            st.session_state.vector_store = VectorStore()
        except Exception as e:
            st.error(f"Failed to initialize vector store: {str(e)}")
            st.info("📄 App will run in web-search only mode")
            st.session_state.vector_store = None
    
    if "documents_processed" not in st.session_state:
        st.session_state.documents_processed = False
    if "processing_status" not in st.session_state:
        st.session_state.processing_status = ""
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""
    if "last_response" not in st.session_state:
        st.session_state.last_response = ""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "query_count" not in st.session_state:
        st.session_state.query_count = 0
    if "document_stats" not in st.session_state:
        st.session_state.document_stats = []

# ===== MAIN APPLICATION =====

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
    
    # Main layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("⚙️ Configuration")
        
        # Model selection
        available_models = GeminiClient.get_available_models()
        selected_model = st.selectbox(
            "🤖 AI Model",
            options=list(available_models.keys()),
            format_func=lambda x: f"{x} - {available_models[x]}",
            help="Choose the AI model for generating responses"
        )
        
        # Perspective and audience
        st.subheader("🎭 Response Style")
        perspective = st.selectbox("Perspective", [
            "scientist", "engineer", "teacher", "policy expert", 
            "AI researcher", "environmentalist", "physician"
        ])
        audience = st.selectbox("Audience", [
            "students", "teacher", "laypeople", "policy makers", 
            "researchers", "community members", "scientist"
        ])
        
        # Temperature control
        temperature = st.slider(
            "🌡️ Creativity Level",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Higher values make responses more creative but less focused"
        )
        
        # Search settings
        st.subheader("🔍 Search Settings")
        enable_web_search = st.checkbox("Enable Web Search", value=True)
        num_doc_results = st.slider("Document Results", 3, 10, 5)
        num_web_results = st.slider("Web Results", 1, 5, 2)
        
        # Status
        st.subheader("📊 Status")
        doc_count = st.session_state.vector_store.get_collection_count() if st.session_state.vector_store else 0
        col_a, col_b = st.columns(2)
        col_a.metric("Documents", doc_count)
        col_b.metric("Queries", st.session_state.query_count)
        
        if st.session_state.documents_processed:
            st.success("✅ Documents Ready")
        else:
            st.info("📄 Upload documents to begin")
        
        if st.session_state.last_response:
            st.success("💬 Ready for questions")
        
        # Clear data button
        st.subheader("⚡ Quick Actions")
        if st.button("🗑️ Clear All Data"):
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
    
    with col2:
        # Document upload
        st.subheader("📄 Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Upload up to 10 PDF files for analysis"
        )
        
        if uploaded_files:
            if st.button("📥 Process Documents"):
                with st.spinner("Processing documents..."):
                    documents = []
                    all_chunks = []
                    
                    for uploaded_file in uploaded_files:
                        # Extract text
                        text = DocumentProcessor.extract_text_from_pdf(uploaded_file)
                        documents.append((uploaded_file.name, text))
                        
                        # Create chunks
                        chunks = DocumentProcessor.chunk_text(text)
                        all_chunks.append(chunks)
                        
                        # Show processing status
                        st.success(f"✅ Processed: {uploaded_file.name}")
                        
                        # Show document stats
                        stats = DocumentProcessor.get_document_stats(text)
                        with st.expander(f"📊 {uploaded_file.name} Stats"):
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
        
        # Show processing status
        if st.session_state.processing_status:
            st.info(st.session_state.processing_status)
        
        # Show document statistics
        if st.session_state.document_stats:
            with st.expander("📊 Document Statistics"):
                for filename, stats in st.session_state.document_stats:
                    st.markdown(f"**{filename}:**")
                    col_a, col_b, col_c, col_d = st.columns(4)
                    col_a.metric("Words", stats['words'])
                    col_b.metric("Chars", stats['characters'])
                    col_c.metric("Lines", stats['lines'])
                    col_d.metric("Tokens", stats['estimated_tokens'])
        
        # Query interface
        st.subheader("💬 Ask Questions")
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
        
        # Show last response if available
        if st.session_state.last_response and not query:
            st.markdown("### 💭 Last Response")
            st.text_area("Previous Answer", st.session_state.last_response, height=300)
            if st.session_state.query_count > 0:
                st.caption(f"Total queries processed: {st.session_state.query_count}")
        
        # Chat history
        if st.session_state.chat_history:
            with st.expander("💬 Chat History"):
                for i, chat in enumerate(reversed(st.session_state.chat_history[-5:]), 1):
                    st.markdown(f"**Q{len(st.session_state.chat_history) - i + 1}:** {chat['query'][:100]}...")
                    st.markdown(f"**A:** {chat['response'][:200]}...")
                    st.markdown("---")

if __name__ == "__main__":
    main()