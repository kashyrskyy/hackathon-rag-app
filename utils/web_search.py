"""
Web Search Functionality
Provides web search capabilities to enhance RAG with real-time information
"""
import requests
from typing import List, Dict, Optional
import streamlit as st
from bs4 import BeautifulSoup
import time

try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    st.warning("⚠️ duckduckgo-search not installed. Web search will use fallback method.")

class WebSearcher:
    """Handles web search operations"""
    
    def __init__(self, serp_api_key: Optional[str] = None):
        """
        Initialize web searcher
        
        Args:
            serp_api_key: Optional SerpAPI key for enhanced search
        """
        self.serp_api_key = serp_api_key
        self.last_search_time = 0
        self.min_search_interval = 3  # Minimum seconds between searches
    
    def search_web(self, query: str, num_results: int = 3) -> List[Dict[str, str]]:
        """
        Search the web for information
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            List of search results with title, snippet, and URL
        """
        # Rate limiting check
        current_time = time.time()
        time_since_last = current_time - self.last_search_time
        
        if time_since_last < self.min_search_interval:
            remaining_wait = self.min_search_interval - time_since_last
            st.info(f"⏱️ Web search rate limited. Waiting {remaining_wait:.1f} seconds...")
            time.sleep(remaining_wait)
        
        self.last_search_time = time.time()
        
        if self.serp_api_key:
            return self._search_with_serpapi(query, num_results)
        elif DDGS_AVAILABLE:
            return self._search_with_ddgs_library(query, num_results)
        else:
            return self._search_with_duckduckgo_api(query, num_results)
    
    def _search_with_serpapi(self, query: str, num_results: int) -> List[Dict[str, str]]:
        """Search using SerpAPI (requires API key)"""
        try:
            url = "https://serpapi.com/search"
            params = {
                "q": query,
                "api_key": self.serp_api_key,
                "engine": "google",
                "num": num_results
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for result in data.get("organic_results", [])[:num_results]:
                results.append({
                    "title": result.get("title", ""),
                    "snippet": result.get("snippet", ""),
                    "url": result.get("link", "")
                })
            
            return results
            
        except Exception as e:
            st.warning(f"SerpAPI search failed: {str(e)}")
            if DDGS_AVAILABLE:
                return self._search_with_ddgs_library(query, num_results)
            else:
                return self._search_with_duckduckgo_api(query, num_results)
    
    def _search_with_ddgs_library(self, query: str, num_results: int) -> List[Dict[str, str]]:
        """Enhanced search using duckduckgo-search library with rate limiting handling"""
        max_retries = 2
        base_delay = 2
        
        for attempt in range(max_retries):
            try:
                results = []
                # Progressive delay to avoid rate limiting
                if attempt > 0:
                    delay = base_delay * (attempt + 1)
                    time.sleep(delay)
                else:
                    time.sleep(1)
                
                # Use different approaches based on attempt
                if attempt == 0:
                    # First attempt: regular text search
                    with DDGS() as ddgs:
                        search_results = ddgs.text(query, max_results=num_results, safesearch='moderate')
                        for result in search_results:
                            results.append({
                                "title": result.get("title", ""),
                                "snippet": result.get("body", "") or result.get("content", ""),
                                "url": result.get("href", "")
                            })
                else:
                    # Second attempt: try with different parameters
                    with DDGS() as ddgs:
                        search_results = ddgs.text(query, max_results=min(num_results, 3), region='us-en')
                        for result in search_results:
                            results.append({
                                "title": result.get("title", ""),
                                "snippet": result.get("body", "") or result.get("content", ""),
                                "url": result.get("href", "")
                            })
                
                if results:
                    return results
                    
            except Exception as e:
                error_msg = str(e)
                if "ratelimit" in error_msg.lower() or "202" in error_msg or "429" in error_msg:
                    if attempt == max_retries - 1:
                        # Final attempt failed - use fallback
                        return self._search_with_duckduckgo_api(query, num_results)
                    continue  # Try again with longer delay
                else:
                    # Non-rate-limit error - try fallback immediately
                    break
        
        # All attempts failed - try fallback API
        return self._search_with_duckduckgo_api(query, num_results)
    
    def _search_with_duckduckgo_api(self, query: str, num_results: int) -> List[Dict[str, str]]:
        """Fallback search using DuckDuckGo (no API key required)"""
        try:
            # Use DuckDuckGo's instant answer API
            url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1"
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            # Try to get abstract
            if data.get("Abstract"):
                results.append({
                    "title": data.get("Heading", query),
                    "snippet": data.get("Abstract"),
                    "url": data.get("AbstractURL", "")
                })
            
            # Try to get related topics
            for topic in data.get("RelatedTopics", [])[:num_results-len(results)]:
                if isinstance(topic, dict) and topic.get("Text"):
                    results.append({
                        "title": topic.get("Text", "")[:100] + "...",
                        "snippet": topic.get("Text", ""),
                        "url": topic.get("FirstURL", "")
                    })
            
            return results if results else self._create_fallback_result(query)
            
        except Exception as e:
            st.warning(f"Web search failed: {str(e)}")
            return self._create_fallback_result(query)
    
    def _create_fallback_result(self, query: str) -> List[Dict[str, str]]:
        """Create a fallback result when search fails"""
        return [{
            "title": f"AI Knowledge Response: {query}",
            "snippet": f"Web search is temporarily unavailable. The AI will respond using its training knowledge and any documents you've uploaded. For real-time information, please try again in a few moments.",
            "url": ""
        }]
    
    def extract_text_from_url(self, url: str, max_chars: int = 1000) -> str:
        """
        Extract text content from a URL
        
        Args:
            url: URL to extract text from
            max_chars: Maximum characters to extract
            
        Returns:
            Extracted text content
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text[:max_chars]
            
        except Exception as e:
            return f"Could not extract content from {url}: {str(e)}"