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
    # Don't show warning here - it shows on every page load
    # st.warning("⚠️ duckduckgo-search not installed. Web search will use fallback method.")

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
        # Rate limiting check (only for DuckDuckGo, not SerpAPI)
        if not self.serp_api_key:
            current_time = time.time()
            time_since_last = current_time - self.last_search_time
            
            if time_since_last < self.min_search_interval:
                remaining_wait = self.min_search_interval - time_since_last
                st.info(f"⏱️ DuckDuckGo rate limited. Waiting {remaining_wait:.1f} seconds...")
                time.sleep(remaining_wait)
            
            self.last_search_time = time.time()
        
        try:
            # Always try SerpAPI first if available
            if self.serp_api_key:
                results = self._search_with_serpapi(query, num_results)
                if results:  # SerpAPI succeeded
                    return results
                # If SerpAPI fails, continue to fallbacks below
            
            # Fallback to DuckDuckGo only if SerpAPI failed or not available
            if DDGS_AVAILABLE:
                return self._search_with_ddgs_library(query, num_results)
            else:
                return self._search_with_duckduckgo_api(query, num_results)
                
        except Exception as e:
            st.warning(f"Web search failed: {str(e)[:100]}...")
            return self._create_fallback_result(query)
    
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
            error_msg = str(e)
            if "401" in error_msg or "unauthorized" in error_msg.lower():
                st.error("🔑 SerpAPI key is invalid or expired. Check your API key in Streamlit secrets.")
            elif "403" in error_msg or "quota" in error_msg.lower():
                st.warning("⚠️ SerpAPI quota exceeded for today.")
            else:
                st.warning(f"SerpAPI search failed: {error_msg[:100]}...")
            
            # Return empty list to let main search function handle fallback
            return []
    
    def _search_with_ddgs_library(self, query: str, num_results: int) -> List[Dict[str, str]]:
        """Enhanced search using duckduckgo-search library with rate limiting handling"""
        max_retries = 3
        base_delay = 3
        
        for attempt in range(max_retries):
            try:
                results = []
                # Progressive delay to avoid rate limiting
                delay = base_delay * (attempt + 1) if attempt > 0 else 2
                time.sleep(delay)
                
                # Use different search strategies
                search_params = [
                    {"safesearch": "moderate", "timelimit": None},
                    {"safesearch": "off", "region": "us-en"},
                    {"max_results": min(num_results, 3), "timelimit": "m"}  # Recent results only
                ][attempt % 3]
                
                with DDGS() as ddgs:
                    search_results = ddgs.text(query, max_results=num_results, **search_params)
                    for result in search_results:
                        title = result.get("title", "")
                        snippet = result.get("body", "") or result.get("content", "")
                        url = result.get("href", "")
                        
                        # Skip if this looks like a fallback result
                        if "AI Knowledge Response" not in title and snippet and len(snippet) > 50:
                            results.append({
                                "title": title,
                                "snippet": snippet,
                                "url": url
                            })
                
                if results:
                    return results
                    
            except Exception as e:
                error_msg = str(e).lower()
                if any(term in error_msg for term in ["ratelimit", "202", "429", "blocked", "forbidden"]):
                    if attempt == max_retries - 1:
                        # Final attempt failed - try alternative search
                        return self._search_with_alternative_method(query, num_results)
                    continue  # Try again with longer delay
                else:
                    # Non-rate-limit error - try fallback immediately
                    break
        
        # All attempts failed - try alternative search
        return self._search_with_alternative_method(query, num_results)
    
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
    
    def _search_with_alternative_method(self, query: str, num_results: int) -> List[Dict[str, str]]:
        """Alternative search method when DuckDuckGo fails"""
        # Try DuckDuckGo API first
        ddg_results = self._search_with_duckduckgo_api(query, num_results)
        if ddg_results and not any("AI Knowledge Response" in result.get("title", "") for result in ddg_results):
            return ddg_results
        
        # Try a simple web scraping approach as last resort
        return self._search_with_simple_scraping(query, num_results)
    
    def _search_with_simple_scraping(self, query: str, num_results: int) -> List[Dict[str, str]]:
        """Simple web scraping when all other methods fail"""
        try:
            # Use a different search approach - search for academic papers if the query seems academic
            if any(term in query.lower() for term in ['paper', 'research', 'study', 'findings', 'compare', 'domain']):
                return self._create_academic_context_result(query)
            
            # For general queries, try a basic web request
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            # Try to get some basic web content
            search_url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
            response = requests.get(search_url, headers=headers, timeout=5)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                results = []
                
                # Look for result links
                for link in soup.find_all('a', class_='result__a')[:num_results]:
                    title = link.get_text(strip=True)
                    url = link.get('href', '')
                    
                    if title and len(title) > 10:
                        results.append({
                            "title": title,
                            "snippet": f"Search result for: {query}",
                            "url": url
                        })
                
                return results if results else []
            
        except Exception:
            pass
        
        # If all else fails, return empty list
        return []
    
    def _create_academic_context_result(self, query: str) -> List[Dict[str, str]]:
        """Create contextual academic search result"""
        # Detect the type of academic query for more specific guidance
        query_lower = query.lower()
        
        if any(term in query_lower for term in ['compare', 'comparison', 'other papers', 'similar domain']):
            snippet = """To compare research findings across papers in similar domains:
1. Search Google Scholar, PubMed, or field-specific databases
2. Look for systematic reviews and meta-analyses in the same area
3. Compare methodologies, sample sizes, and statistical approaches
4. Examine how findings align or contradict with established literature
5. Consider publication dates and technological advances between studies
6. Check citation patterns to identify influential papers in the field"""
        elif any(term in query_lower for term in ['findings', 'results', 'conclusions']):
            snippet = """For analyzing research findings and their significance:
1. Examine the statistical significance and effect sizes reported
2. Consider the study design and potential limitations
3. Look for replication studies and validation of results
4. Check how findings contribute to existing theoretical frameworks
5. Assess the practical implications and real-world applications
6. Review peer commentary and citations of the work"""
        else:
            snippet = """For comprehensive academic research analysis:
1. Use databases like Google Scholar, PubMed, Web of Science, or Scopus
2. Search with relevant keywords and MeSH terms if applicable
3. Filter by publication date, study type, and impact factor
4. Read abstracts and full papers from reputable journals
5. Look for systematic reviews and meta-analyses
6. Check citation networks to find related influential work"""
        
        return [{
            "title": f"Academic Research Guidance: {query[:80]}...",
            "snippet": snippet,
            "url": "https://scholar.google.com/"
        }]
    
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