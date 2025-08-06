"""
LLM Client for Google Gemini API
Handles all interactions with the Gemini API for text generation
"""
import google.generativeai as genai
import streamlit as st
from typing import Optional, Dict, Any
import time

class GeminiClient:
    """Client for Google Gemini API"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        """
        Initialize Gemini client
        
        Args:
            api_key: Google API key for Gemini
            model_name: Name of the Gemini model to use
        """
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
        """
        Generate response using Gemini API
        
        Args:
            prompt: The input prompt
            temperature: Controls randomness (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            
        Returns:
            Generated text response
        """
        try:
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            
            # Generate response
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
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text (approximate)
        
        Args:
            text: Input text
            
        Returns:
            Approximate token count
        """
        try:
            result = self.model.count_tokens(text)
            return result.total_tokens
        except:
            # Fallback: rough approximation (1 token ≈ 4 characters)
            return len(text) // 4
    
    @staticmethod
    def get_available_models() -> Dict[str, str]:
        """
        Get available Gemini models
        
        Returns:
            Dictionary of model names and descriptions
        """
        return {
            "gemini-1.5-flash": "Fast and efficient for most tasks",
            "gemini-1.5-pro": "Most capable model for complex tasks",
            "gemini-2.0-flash": "Latest generation model with improved performance"
        }