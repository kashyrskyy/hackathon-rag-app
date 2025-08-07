# 🚀 Quick Start Guide

Get your RAG - AI Research Assistant running in **under 10 minutes**!

## 📝 What You'll Need

1. **GitHub account** (free)
2. **Google account** (for Gemini API key)
3. **5 minutes** of your time

## ⚡ Super Quick Deploy (3 steps)

### Step 1: Get Your API Key (2 minutes)
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key" → "Create API key in new project"
4. **Copy the key** (keep it safe!)

### Step 2: Deploy to Streamlit (2 minutes)
1. **Fork this repository** on GitHub (click Fork button)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app" → Select your forked repository
5. Set main file path: `streamlit_app.py`
6. Click "Deploy!"

### Step 3: Add Your API Key (1 minute)
1. In your deployed app, click the hamburger menu (☰)
2. Go to "Settings" → "Secrets"
3. Add this:
```toml
GOOGLE_API_KEY = "paste_your_api_key_here"
```
4. Click "Save"
5. Click "Reboot app"

## 🎉 You're Done!

Your app is now live at: `https://your-app-name.streamlit.app`

## 🧪 Test Your App

1. **Upload a PDF** (try a research paper or document)
2. **Ask a question** like "What is this document about?"
3. **Watch the magic** happen! ✨

## 💡 Pro Tips

### For Better Results:
- Use **specific questions** rather than vague ones
- Try different **perspectives** (scientist, teacher, etc.)
- Adjust **temperature** for more creative/focused answers
- Enable **web search** for current information

### Example Questions:
- "Summarize the key findings of this research"
- "What are the practical implications?"
- "Explain this concept for a high school student"
- "What are the limitations mentioned?"

## 🆘 Something Not Working?

### Common Issues:

**"Google API key not found"**
- Double-check your API key in Streamlit secrets
- Make sure there are no extra spaces
- Try regenerating the key

**"File upload failed"**
- Check file size (max 200MB)
- Try a different PDF
- Ensure PDF is not password-protected

**"Slow responses"**
- Use Gemini 1.5 Flash (faster)
- Reduce number of search results
- Try shorter documents

### Still Stuck?
- Check the full [README.md](README.md) for detailed instructions
- Look at [DEPLOYMENT.md](DEPLOYMENT.md) for troubleshooting
- Create an issue in the GitHub repository

## 🎯 Hackathon Tips

### For Your Presentation:
1. **Demo the key features**:
   - PDF upload and processing
   - Intelligent Q&A with context
   - Web search enhancement
   - Perspective/audience customization

2. **Show the technology**:
   - Google Gemini API integration
   - Vector search with embeddings
   - Real-time web search
   - Clean, modern UI

3. **Highlight the value**:
   - Free to deploy and run
   - Scalable cloud architecture
   - Production-ready code
   - Easy to customize and extend

### Customization Ideas:
- Change the UI theme/colors
- Add new document types
- Implement user authentication
- Add more search providers
- Create specialized prompts for your domain

## 📈 Usage Costs

**Free Tier (Perfect for Hackathons):**
- Streamlit Community Cloud: **FREE**
- Google Gemini API: **1M requests/month FREE**
- Estimated usage: **$0/month** for typical hackathon demo

**Production Scale:**
- Light usage (100 queries/day): **$0-5/month**
- Medium usage (500 queries/day): **$5-15/month**
- Heavy usage (2000 queries/day): **$20-50/month**

## 🔗 Useful Links

- **Live Demo**: [Your app URL after deployment]
- **Google AI Studio**: https://makersuite.google.com/app/apikey
- **Streamlit Cloud**: https://share.streamlit.io
- **GitHub Repository**: [Your forked repo URL]

---

**Ready to impress the judges? Your RAG app is just 3 steps away! 🚀**

*Questions? Check the detailed guides or create an issue!*