# 🚀 Quick Start Guide - Flask Fashion Sense AI

## ✅ What Was Created

Your Streamlit app has been completely converted into a **Flask web application** with the following structure:

```
Fashion-Sense-AI/
├── flask_app.py              # Main Flask application with REST API
├── run_flask.ps1             # PowerShell startup script
├── requirements_flask.txt    # Flask dependencies
├── FLASK_README.md          # Complete API documentation
│
├── Services/                 # Backend services (modular)
│   ├── __init__.py
│   ├── embedding_service.py  # 896D embeddings (CLIP 512D + SBERT 384D)
│   ├── faiss_service.py      # FAISS IndexFlatL2 search
│   ├── llm_service.py        # Gemma-3 outfit reasoning
│   └── data_service.py       # Product data management
│
├── templates/               # Frontend HTML
│   └── index.html           # Modern single-page application
│
├── static/                  # Frontend assets
│   ├── css/
│   │   └── style.css        # Complete styling
│   └── js/
│       └── app.js           # Frontend logic & API calls
│
├── uploads/                 # Temporary image storage
└── Assets/                  # FAISS index & embeddings
```

---

## 🎯 Key Features Implemented

### ✅ Multimodal 896D Embeddings
- **CLIP (ViT-B/32)**: 512D for images and text
- **SentenceTransformer (all-MiniLM-L6-v2)**: 384D for text
- **Combined**: 896D multimodal representation

### ✅ Three Search Modes
1. **Image Search**: Upload image → Find similar products
2. **Text Search**: Text query → Find matching products  
3. **Multimodal Search**: Image + Text → Combined search

### ✅ FAISS IndexFlatL2
- L2 distance (Euclidean)
- Exact nearest neighbor search
- Returns similarity scores with results

### ✅ LLM Reasoning (Gemma-3)
- Analyzes top-K results
- Explains WHY products match
- Provides outfit completion suggestions
- Styling tips and recommendations

### ✅ REST API Endpoints
- `/api/search/image` - Image-based search
- `/api/search/text` - Text-based search
- `/api/search/multimodal` - Combined search
- `/api/embeddings/update` - Update specific embeddings
- `/api/embeddings/batch` - Batch embedding updates
- `/api/stats` - Dataset statistics
- `/api/health` - Health check

### ✅ Modern UI
- Clean, responsive design
- Image upload with drag-and-drop
- Real-time results with similarity scores
- LLM reasoning display
- Dataset statistics

---

## 🚀 How to Run

### Option 1: Use PowerShell Script (Recommended)
```powershell
.\run_flask.ps1
```

### Option 2: Manual Steps
```powershell
# 1. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 2. Set environment variable
$env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python'

# 3. Install Flask dependencies
pip install -r requirements_flask.txt

# 4. Run the application
python flask_app.py
```

### Access the Application
Open your browser and go to: **http://localhost:5000**

---

## 📡 API Usage Examples

### Image Search (cURL)
```bash
curl -X POST http://localhost:5000/api/search/image \
  -F "image=@path/to/image.jpg" \
  -F "top_k=10" \
  -F "include_reasoning=true" \
  -F "hf_token=your_token"
```

### Text Search (Python)
```python
import requests

response = requests.post('http://localhost:5000/api/search/text', json={
    'query': 'blue denim jeans, skinny fit',
    'top_k': 10,
    'include_reasoning': True,
    'hf_token': 'your_token'
})

results = response.json()
```

### Multimodal Search (JavaScript)
```javascript
const formData = new FormData();
formData.append('image', imageFile);
formData.append('query', 'casual everyday wear');
formData.append('top_k', 10);

const response = await fetch('/api/search/multimodal', {
    method: 'POST',
    body: formData
});

const results = await response.json();
```

---

## 🔧 Configuration

### Hugging Face Token
Get your free token from: https://huggingface.co/settings/tokens

Enter it in the UI or pass it via API:
- **UI**: Enter in the "🔑 Hugging Face Token" field
- **API**: Include `hf_token` in request

### Environment Variables
```powershell
# Protobuf compatibility
$env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python'

# Flask debug mode
$env:FLASK_DEBUG=True
```

---

## 📊 Embedding Update Workflow

Since only ~10% of your dataset was previously embedded, you can:

### 1. Update via UI
- Click "Update Embeddings (10%)" button
- This processes 10% of products automatically

### 2. Update via API
```python
import requests

# Update specific products
response = requests.post('http://localhost:5000/api/embeddings/update', json={
    'product_ids': ['ABC123', 'DEF456']
})

# Batch update
response = requests.post('http://localhost:5000/api/embeddings/batch', json={
    'start_idx': 0,
    'end_idx': 1000,
    'batch_size': 100
})
```

### 3. Monitor Progress
Check `/api/stats` to see:
- Total products
- Total embeddings
- Coverage percentage

---

## 🎨 Frontend Features

### Image Upload
- Click to upload or drag & drop
- Supports PNG, JPG, JPEG (up to 16MB)
- Real-time preview

### Search Modes
- **📷 Image**: Upload fashion item image
- **📝 Text**: Describe what you're looking for
- **🎨 Multimodal**: Combine image + text for refined search

### Results Display
- Product images with similarity scores
- Ranked results (#1, #2, etc.)
- Brand, price, and match percentage
- LLM reasoning (optional)

---

## 🧪 Testing the System

### Test Image Search
1. Upload a jeans or dress image
2. Set top_k to 10
3. Enable "Generate AI Outfit Reasoning"
4. Enter your HF token
5. Click "Search"

### Test Text Search
1. Switch to "📝 Text Search" tab
2. Enter: "blue skinny jeans, high waist"
3. Click "Search"

### Test Multimodal
1. Switch to "🎨 Multimodal" tab
2. Upload an image
3. Add text: "casual weekend outfit"
4. Click "Search"

---

## 📈 Performance

### Embedding Generation
- **Image**: 100-200ms (GPU) / 500-1000ms (CPU)
- **Text**: 50-100ms
- **Multimodal**: 150-300ms (GPU)

### Search
- **FAISS**: <10ms for 10K vectors
- **Top-K retrieval**: Near-instant

### LLM Reasoning
- **Gemma-3 API**: 5-15 seconds
- **Optional**: Can be disabled for faster results

---

## 🐛 Troubleshooting

### Port Already in Use
```powershell
# Change port in flask_app.py (last line)
app.run(debug=True, host='0.0.0.0', port=5001)
```

### Models Not Loading
- Check internet connection (models download from HuggingFace)
- Verify disk space (models ~2-3GB total)
- Check CUDA availability for GPU

### FAISS Index Not Found
- Normal on first run - will be created automatically
- Upload embeddings using batch API endpoint
- Index saved to `Assets/faiss_index_896d.index`

---

## 📚 Next Steps

### 1. Generate Initial Embeddings
Run batch embedding for your entire dataset:
```python
import requests

response = requests.post('http://localhost:5000/api/embeddings/batch', json={
    'start_idx': 0,
    'batch_size': 100
})
```

### 2. Test All Features
- Try all three search modes
- Test with different images
- Experiment with text queries
- Check LLM reasoning quality

### 3. Expand Dataset
- Add more product categories
- Update embeddings for new products
- Monitor via `/api/stats`

---

## 🎓 For Your MMAL Project

This implementation covers:
- ✅ **Multimodal Learning**: CLIP + SBERT embeddings
- ✅ **Vector Search**: FAISS with L2 distance
- ✅ **LLM Integration**: Gemma-3 for reasoning
- ✅ **REST API**: Complete backend
- ✅ **Web Interface**: Modern frontend
- ✅ **Scalability**: Modular, extensible design
- ✅ **Dataset Expansion**: Easy to add embeddings

---

## 📞 Support

For issues or questions:
1. Check `FLASK_README.md` for detailed API docs
2. Review error messages in terminal
3. Verify all dependencies are installed

---

**Happy Fashion Searching! 👗✨**
