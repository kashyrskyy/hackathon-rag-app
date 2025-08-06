"""
Vector Store Management
Handles embeddings, vector storage, and similarity search using ChromaDB
"""
import chromadb
from sentence_transformers import SentenceTransformer
import streamlit as st
from typing import List, Tuple, Optional
import uuid

class VectorStore:
    """Manages vector storage and retrieval using ChromaDB"""
    
    def __init__(self, collection_name: str = "rag_documents"):
        """
        Initialize vector store
        
        Args:
            collection_name: Name of the ChromaDB collection
        """
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.embedding_model = self._load_embedding_model()
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client with error handling"""
        try:
            # Try to create client with ephemeral mode for Streamlit Cloud
            self.client = chromadb.Client()
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
            st.success("✅ Vector store initialized successfully")
        except Exception as e:
            st.warning(f"⚠️ Vector store initialization failed: {str(e)}")
            st.info("📄 Vector search will be disabled. You can still use the app without document search.")
            self.client = None
            self.collection = None
    
    @st.cache_resource
    def _load_embedding_model(_self):
        """Load sentence transformer model for embeddings"""
        try:
            return SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            st.error(f"Error loading embedding model: {str(e)}")
            return None
    
    def add_documents(self, documents: List[Tuple[str, str]], chunks_per_doc: List[List[str]]):
        """
        Add documents and their chunks to the vector store
        
        Args:
            documents: List of (filename, full_text) tuples
            chunks_per_doc: List of chunk lists for each document
        """
        if not self.embedding_model or not self.client or not self.collection:
            st.error("Vector store not properly initialized - document search disabled")
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
                # Generate embeddings
                with st.spinner("🧠 Generating embeddings..."):
                    embeddings = self.embedding_model.encode(all_chunks).tolist()
                
                # Add to ChromaDB
                self.collection.add(
                    documents=all_chunks,
                    metadatas=all_metadatas,
                    ids=all_ids,
                    embeddings=embeddings
                )
                
                st.success(f"✅ Added {len(all_chunks)} chunks to vector store")
                
            except Exception as e:
                st.error(f"Error adding documents to vector store: {str(e)}")
    
    def search(self, query: str, n_results: int = 5) -> List[dict]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of search results with metadata
        """
        if not self.embedding_model or not self.client or not self.collection:
            st.warning("Vector store not available - skipping document search")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            if results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    formatted_results.append({
                        "content": doc,
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "distance": results["distances"][0][i] if results["distances"] else 0
                    })
            
            return formatted_results
            
        except Exception as e:
            st.error(f"Error searching vector store: {str(e)}")
            return []
    
    def clear_collection(self):
        """Clear all documents from the collection"""
        if not self.client or not self.collection:
            st.warning("Vector store not available")
            return
            
        try:
            # Delete and recreate collection
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
            st.success("🗑️ Vector store cleared")
        except Exception as e:
            st.error(f"Error clearing vector store: {str(e)}")
    
    def get_collection_count(self) -> int:
        """Get number of documents in collection"""
        if not self.collection:
            return 0
        try:
            return self.collection.count()
        except:
            return 0