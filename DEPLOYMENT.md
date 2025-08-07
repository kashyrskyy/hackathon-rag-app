# 🚀 Deployment Guide

Complete guide to deploy your RAG - AI Research Assistant to Streamlit Community Cloud.

## 📋 Pre-Deployment Checklist

### ✅ Required Items
- [ ] GitHub account
- [ ] Google API key for Gemini
- [ ] All files committed to GitHub repository
- [ ] Streamlit Community Cloud account

### 🔑 API Keys Needed
1. **Google Gemini API Key** (Required)
   - Visit: https://makersuite.google.com/app/apikey
   - Create free account and generate API key
   - Free tier: 15 requests/min, 1,500/day, 1M/month

2. **SerpAPI Key** (Optional)
   - Visit: https://serpapi.com/
   - Enhanced web search capabilities
   - Free tier: 100 searches/month

## 🏗️ Step-by-Step Deployment

### Step 1: Prepare Your Repository

1. **Fork or Clone** this repository to your GitHub account
2. **Customize** the app (optional):
   - Update `README.md` with your information
   - Modify UI styling in `streamlit_app.py`
   - Adjust configuration in `.streamlit/config.toml`

3. **Commit and Push** all changes:
```bash
git add .
git commit -m "Initial setup for deployment"
git push origin main
```

### Step 2: Create Streamlit Community Cloud Account

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign up with your GitHub account
3. Authorize Streamlit to access your repositories

### Step 3: Deploy Your App

1. **Click "New app"** in your Streamlit dashboard
2. **Select your repository** from the dropdown
3. **Configure deployment**:
   - Repository: `your-username/hackathon-rag-app`
   - Branch: `main`
   - Main file path: `streamlit_app.py`
4. **Click "Deploy!"**

### Step 4: Add Secrets (API Keys)

1. **Go to your app settings** (click the hamburger menu → Settings)
2. **Navigate to "Secrets"** tab
3. **Add your secrets** in TOML format:

```toml
# Required
GOOGLE_API_KEY = "your_actual_gemini_api_key_here"

# Optional (for enhanced web search)
SERP_API_KEY = "your_actual_serp_api_key_here"
```

4. **Save** the secrets
5. **Reboot** the app to apply changes

## 🔧 Configuration Options

### App Settings
You can customize these in your Streamlit Cloud dashboard:

- **App name**: Change the subdomain (e.g., `my-rag-app.streamlit.app`)
- **Python version**: Use Python 3.8+ (automatically detected)
- **Resource limits**: Streamlit Community Cloud provides sufficient resources

### Custom Domain (Optional)
- Available with Streamlit Cloud Pro
- Point your domain to your Streamlit app
- Configure HTTPS automatically

## 🧪 Testing Your Deployment

### Quick Test Checklist
1. **App loads** without errors
2. **File upload** works for PDFs
3. **Document processing** completes successfully
4. **Questions** generate appropriate responses
5. **Web search** enhances answers (if enabled)
6. **Model selection** changes behavior
7. **Temperature slider** affects response style

### Sample Test Questions
After uploading a PDF, try these:
- "What is the main topic of this document?"
- "Summarize the key findings"
- "What are the implications for [your field]?"

## 🐛 Troubleshooting

### Common Issues

#### 1. "Google API key not found"
**Solution**: 
- Check that `GOOGLE_API_KEY` is added to Streamlit secrets
- Ensure no extra spaces or quotes in the key
- Verify the API key is active in Google AI Studio

#### 2. "Module not found" errors
**Solution**:
- Ensure all dependencies are in `requirements.txt`
- Check for typos in import statements
- Verify Python version compatibility

#### 3. PDF processing fails
**Solution**:
- Check file size limits (200MB per file)
- Ensure PDF is not password-protected
- Try with a different PDF file

#### 4. Slow response times
**Solution**:
- Use Gemini 1.5 Flash for faster responses
- Reduce number of document chunks retrieved
- Disable web search for faster processing

#### 5. Rate limit errors
**Solution**:
- Wait a few minutes and try again
- Consider upgrading to paid Gemini API tier
- Implement request throttling

### Debug Mode

To enable debug information:

1. Add to your secrets:
```toml
DEBUG = "true"
```

2. Check Streamlit logs for detailed error messages

3. Use the app's built-in error handling and status messages

## 📊 Monitoring & Analytics

### Built-in Metrics
Your app automatically tracks:
- Processing status and document statistics
- Debug information (when debug mode enabled)
- Chat history (session-based)

### Streamlit Analytics
Access via your Streamlit dashboard:
- **Usage stats**: Views, users, sessions
- **Performance**: Load times, errors
- **Resource usage**: Memory, CPU

### Custom Analytics (Optional)
Add Google Analytics or other tracking:

```python
# Add to streamlit_app.py
st.markdown("""
<script async src="https://www.googletagmanager.com/gtag/js?id=GA_MEASUREMENT_ID"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'GA_MEASUREMENT_ID');
</script>
""", unsafe_allow_html=True)
```

## 🔄 Updates & Maintenance

### Updating Your App
1. Make changes to your code
2. Commit and push to GitHub
3. Streamlit automatically redeploys (usually within 1-2 minutes)

### Dependency Updates
Update `requirements.txt` and push changes:
```bash
# Update specific package
pip install package_name==new_version
pip freeze > requirements.txt
```

### Monitoring Costs
- **Gemini API**: Monitor usage in Google AI Studio
- **SerpAPI**: Check usage in SerpAPI dashboard
- **Streamlit**: Community Cloud is free with usage limits

## 🚨 Security Best Practices

### API Key Security
- ✅ **DO**: Store API keys in Streamlit secrets
- ❌ **DON'T**: Commit API keys to GitHub
- ✅ **DO**: Use environment variables for local development
- ❌ **DON'T**: Share API keys in screenshots or logs

### Data Privacy
- User-uploaded documents are processed in-memory
- No persistent storage of user data
- Vector embeddings reset on app restart
- Consider adding privacy notice for production use

## 📈 Scaling Considerations

### Free Tier Limits
- **Streamlit Community Cloud**: 1GB RAM, shared CPU
- **Gemini API**: 15 requests/min, 1,500/day, 1M/month
- **SerpAPI**: 100 searches/month (free tier)

### Upgrade Options
- **Streamlit Cloud Pro**: More resources, custom domains
- **Gemini API Paid**: Higher rate limits, more models
- **SerpAPI Pro**: More searches, advanced features

### Performance Optimization
- Use smaller embedding models for faster processing
- Implement caching for repeated queries
- Optimize chunk size and overlap
- Consider persistent vector storage for production

## 🎉 Going Live

### Pre-Launch Checklist
- [ ] Test all features thoroughly
- [ ] Update README with your live app URL
- [ ] Add proper error handling
- [ ] Set up monitoring/alerts
- [ ] Create user documentation
- [ ] Plan for scaling if needed

### Sharing Your App
- **Direct link**: `https://your-app-name.streamlit.app`
- **Social media**: Share screenshots and features
- **GitHub**: Update README with live demo link
- **Hackathon submission**: Include deployment URL

### Success Metrics
Track these for your hackathon presentation:
- Number of documents processed
- Questions answered
- User engagement time
- Unique features demonstrated
- Technical challenges solved

---

**🚀 Ready to deploy? Your RAG - AI Research Assistant will be live in minutes!**

*Need help? Check the troubleshooting section or create an issue in the repository.*