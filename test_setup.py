"""
Test script to verify all components are working correctly
Run this before deploying to catch any issues early
"""
import os
import sys

def test_imports():
    """Test all required imports"""
    print("🔍 Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        from utils.llm_client import GeminiClient
        print("✅ LLM client imported successfully")
    except ImportError as e:
        print(f"❌ LLM client import failed: {e}")
        return False
    
    try:
        from utils.document_processor import DocumentProcessor
        print("✅ Document processor imported successfully")
    except ImportError as e:
        print(f"❌ Document processor import failed: {e}")
        return False
    
    try:
        from utils.vector_store import VectorStore
        print("✅ Vector store imported successfully")
    except ImportError as e:
        print(f"❌ Vector store import failed: {e}")
        return False
    
    try:
        from utils.web_search import WebSearcher
        print("✅ Web searcher imported successfully")
    except ImportError as e:
        print(f"❌ Web searcher import failed: {e}")
        return False
    
    return True

def test_dependencies():
    """Test critical dependencies"""
    print("\n🔍 Testing dependencies...")
    
    try:
        import google.generativeai as genai
        print("✅ Google Generative AI available")
    except ImportError:
        print("❌ Google Generative AI not installed")
        return False
    
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ Sentence Transformers available")
    except ImportError:
        print("❌ Sentence Transformers not installed")
        return False
    
    try:
        import chromadb
        print("✅ ChromaDB available")
    except ImportError:
        print("❌ ChromaDB not installed")
        return False
    
    try:
        import PyPDF2
        print("✅ PyPDF2 available")
    except ImportError:
        print("❌ PyPDF2 not installed")
        return False
    
    return True

def test_configuration():
    """Test configuration files"""
    print("\n🔍 Testing configuration...")
    
    # Check if config files exist
    config_files = [
        ".streamlit/config.toml",
        ".streamlit/secrets.toml.example",
        "requirements.txt",
        "README.md"
    ]
    
    for file in config_files:
        if os.path.exists(file):
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} missing")
            return False
    
    return True

def test_api_key():
    """Test API key availability"""
    print("\n🔍 Testing API key setup...")
    
    # Check environment variable
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        print("✅ GOOGLE_API_KEY found in environment")
        return True
    
    # Check if secrets file exists (for deployment)
    if os.path.exists(".streamlit/secrets.toml"):
        print("✅ Secrets file exists for deployment")
        return True
    
    print("⚠️  No API key found. Set GOOGLE_API_KEY environment variable or add to Streamlit secrets")
    return False

def main():
    """Run all tests"""
    print("🚀 Running Enhanced RAG App Setup Test\n")
    
    tests = [
        ("Import Test", test_imports),
        ("Dependencies Test", test_dependencies),
        ("Configuration Test", test_configuration),
        ("API Key Test", test_api_key),
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "="*50)
    print("📊 TEST RESULTS")
    print("="*50)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Ready for deployment!")
        print("\nNext steps:")
        print("1. Get your Google API key: https://makersuite.google.com/app/apikey")
        print("2. Push to GitHub repository")
        print("3. Deploy to Streamlit Community Cloud")
        print("4. Add API key to Streamlit secrets")
    else:
        print("❌ SOME TESTS FAILED! Please fix issues before deploying.")
        print("\nTroubleshooting:")
        print("- Install missing dependencies: pip install -r requirements.txt")
        print("- Set up API key: export GOOGLE_API_KEY='your_key_here'")
        print("- Check file structure and imports")
    
    print("="*50)

if __name__ == "__main__":
    main()