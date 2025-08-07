# 🧠 RAG - AI as a Research Assistant

A powerful document Q&A system with AI-powered responses and web search enhancement. Built for hackathons and deployed on Streamlit Community Cloud.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.40.1-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🚀 Live Demo

**[Try the app here!](https://hackathon-rag.streamlit.app/)** - Experience the RAG assistant in action!

## ✨ Features

### 🎯 Core RAG Capabilities
- **PDF Upload & Processing**: Upload multiple PDFs (up to 10 files, 200MB each)
- **Intelligent Chunking**: Smart text segmentation with overlap for better context
- **Vector Search**: Semantic similarity search using sentence transformers
- **AI-Powered Q&A**: Natural language responses using Google Gemini API

### 🌐 Enhanced with Web Search
- **Real-time Information**: Augment document knowledge with live web search
- **SerpAPI Primary**: Premium Google search for reliable results
- **DuckDuckGo Fallback**: Free search when SerpAPI is unavailable

### 🎭 Customizable Response Style
- **Perspective Control**: Choose AI persona (scientist, teacher, consultant, etc.)
- **Audience Targeting**: Tailor responses for different audiences
- **Temperature Control**: Adjust creativity vs. focus in responses

### 💡 Advanced Features
- **Multi-model Support**: Choose between Gemini 1.5 Flash, Pro, and 2.0 models
- **Debug Mode**: Detailed processing information for developers
- **Clean Interface**: Streamlined UI focused on essential functionality
- **Source Attribution**: See which documents and web sources informed each answer

## 🛠️ Technology Stack

- **Frontend**: Streamlit with custom CSS styling
- **LLM**: Google Gemini API (1.5 Flash/Pro, 2.0 Flash)
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Store**: ChromaDB (in-memory)
- **PDF Processing**: PyPDF2
- **Web Search**: DuckDuckGo API + optional SerpAPI
- **Deployment**: Streamlit Community Cloud

## 🚀 Quick Start

### 1. Get Your API Key
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a free Google API key for Gemini
3. (Optional) Get a [SerpAPI key](https://serpapi.com/) for enhanced web search

### 2. Local Development

```bash
# Clone the repository
git clone https://github.com/yourusername/hackathon-rag-app.git
cd hackathon-rag-app

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export GOOGLE_API_KEY="your_gemini_api_key_here"
export SERP_API_KEY="your_serp_api_key_here"  # Optional

# Run the app
streamlit run streamlit_app.py
```

### 3. Deploy to Streamlit Community Cloud

1. **Fork this repository** to your GitHub account
2. **Create Streamlit Account** at [share.streamlit.io](https://share.streamlit.io)
3. **Connect GitHub** and select your forked repository
4. **Add Secrets** in your Streamlit app settings:
   ```toml
   GOOGLE_API_KEY = "your_gemini_api_key_here"
   SERP_API_KEY = "your_serp_api_key_here"  # Optional
   ```
5. **Deploy** - Your app will be live in minutes!

## 📋 Usage Guide

### Step 1: Upload Documents
- Click "Browse files" and select up to 10 PDF files
- Hit "🚀 Process Documents" to extract and index content
- View document statistics and processing status

### Step 2: Configure Settings
- **Model**: Choose between Gemini models based on your needs
- **Temperature**: Adjust response creativity (0.0 = focused, 1.0 = creative)
- **Perspective**: Select AI persona (scientist, teacher, etc.)
- **Audience**: Choose target audience (students, professionals, etc.)
- **Search**: Enable/disable web search enhancement

### Step 3: Ask Questions
- Type your question in the text area
- Click "🔍 Get Answer" to get AI-powered responses
- View sources and chat history

## 💰 Cost Estimation

### Google Gemini API Pricing (as of 2024)
- **Free Tier**: 15 requests/minute, 1,500/day, 1M/month
- **Gemini 1.5 Flash**: $0.075/$0.30 per 1M tokens (input/output)
- **Gemini 1.5 Pro**: $1.25/$5.00 per 1M tokens (input/output)

### Typical Usage Costs
- **Light Usage** (100 queries/day): FREE (within limits)
- **Medium Usage** (500 queries/day): $5-15/month
- **Heavy Usage** (2000 queries/day): $20-50/month

*Note: Costs depend on document size, query complexity, and model choice*

## 🏗️ Architecture

```
User Interface (Streamlit)
    ↓
Document Processing (PyPDF2)
    ↓
Text Chunking & Embeddings (SentenceTransformers)
    ↓
Vector Storage (ChromaDB)
    ↓
Query Processing & Search
    ↓
Context Enhancement (Web Search)
    ↓
Response Generation (Google Gemini API)
    ↓
Formatted Output & Source Attribution
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 Example Use Cases

### 🎓 Education
- **Student**: Upload textbooks, ask study questions
- **Teacher**: Process curriculum materials, generate explanations
- **Researcher**: Analyze papers, get contextual insights

### 💼 Business
- **Consultant**: Process client documents, generate insights
- **Analyst**: Review reports, extract key findings
- **Manager**: Understand policies, get quick summaries

### 🔬 Research
- **Scientist**: Analyze research papers, compare methodologies
- **Engineer**: Review technical specifications, troubleshoot issues
- **Policy Expert**: Process regulations, understand implications

## ⚠️ Limitations

- **Document Types**: Currently supports PDF only
- **File Size**: Maximum 200MB per file, 10 files total
- **Language**: Optimized for English text
- **Memory**: Vector store is in-memory (resets on restart)
- **Rate Limits**: Subject to Gemini API rate limits

## 🔧 Configuration Options

### Environment Variables
```bash
GOOGLE_API_KEY=your_gemini_api_key     # Required
SERP_API_KEY=your_serp_api_key         # Optional, for enhanced web search
```

### Model Options
- `gemini-1.5-flash`: Fast and efficient (recommended)
- `gemini-1.5-pro`: Most capable for complex tasks
- `gemini-2.0-flash`: Latest generation model

### Customization
- Modify chunk size/overlap in `utils/document_processor.py`
- Adjust embedding model in `utils/vector_store.py`
- Customize UI styling in `streamlit_app.py`

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/hackathon-rag-app/issues)
- **Documentation**: This README and code comments
- **API Docs**: [Google Gemini API](https://ai.google.dev/docs)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google**: For the powerful Gemini API
- **Streamlit**: For the amazing deployment platform
- **Hugging Face**: For sentence transformers
- **ChromaDB**: For vector storage capabilities

---

**Built with ❤️ for hackathons and rapid prototyping!**

*Ready to deploy and impress? Fork this repo and get your RAG app live in minutes!* 🚀