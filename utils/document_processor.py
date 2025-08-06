"""
Document Processing Utilities
Handles PDF processing, text extraction, and chunking
"""
import PyPDF2
import streamlit as st
from typing import List, Tuple
from io import BytesIO

class DocumentProcessor:
    """Handles document processing operations"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_file) -> str:
        """
        Extract text from uploaded PDF file
        
        Args:
            pdf_file: Streamlit uploaded file object
            
        Returns:
            Extracted text content
        """
        try:
            # Read PDF content
            pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file.read()))
            text = ""
            
            # Extract text from each page
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            
            return text.strip()
            
        except Exception as e:
            st.error(f"Error processing PDF {pdf_file.name}: {str(e)}")
            return ""
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Input text to chunk
            chunk_size: Size of each chunk in words
            overlap: Number of overlapping words between chunks
            
        Returns:
            List of text chunks
        """
        if not text.strip():
            return []
        
        words = text.split()
        chunks = []
        
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            
            # Move start position considering overlap
            start += (chunk_size - overlap)
            
            # Break if we've processed all words
            if end >= len(words):
                break
        
        return chunks
    
    @staticmethod
    def process_multiple_pdfs(pdf_files) -> List[Tuple[str, str]]:
        """
        Process multiple PDF files
        
        Args:
            pdf_files: List of uploaded PDF files
            
        Returns:
            List of tuples (filename, extracted_text)
        """
        processed_docs = []
        
        for pdf_file in pdf_files:
            if pdf_file is not None:
                text = DocumentProcessor.extract_text_from_pdf(pdf_file)
                if text:
                    processed_docs.append((pdf_file.name, text))
                    st.success(f"✅ Processed: {pdf_file.name}")
                else:
                    st.warning(f"⚠️ No text extracted from: {pdf_file.name}")
        
        return processed_docs
    
    @staticmethod
    def get_document_stats(text: str) -> dict:
        """
        Get statistics about the document
        
        Args:
            text: Document text
            
        Returns:
            Dictionary with document statistics
        """
        words = len(text.split())
        characters = len(text)
        lines = len(text.split('\n'))
        
        return {
            'words': words,
            'characters': characters,
            'lines': lines,
            'estimated_tokens': characters // 4  # Rough approximation
        }