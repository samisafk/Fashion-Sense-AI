# Fashion Sense AI - Flask API

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- CUDA-enabled GPU (optional, for faster processing)
- Hugging Face API Token ([Get one here](https://huggingface.co/settings/tokens))

### Installation

1. **Install Flask dependencies**:
```bash
pip install -r requirements_flask.txt
```

2. **Set environment variable for protobuf** (if needed):
```bash
# Windows PowerShell
$env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python'

# Linux/Mac
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

3. **Run the Flask application**:
```bash
python flask_app.py
```

The application will be available at: http://localhost:5000

---

## 📡 API Endpoints

### Health Check
**GET** `/api/health`

Check the health status of all services.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-11-01T12:00:00",
  "services": {
    "embedding": true,
    "faiss": true,
    "llm": true,
    "data": true
  }
}
```

---

### Image Search
**POST** `/api/search/image`

Search for similar fashion items using an uploaded image.

**Form Data:**
- `image` (file, required): Image file (PNG, JPG, JPEG)
- `top_k` (int, optional): Number of results (default: 10)
- `include_reasoning` (bool, optional): Generate LLM reasoning (default: true)
- `hf_token` (string, optional): Hugging Face API token

**Response:**
```json
{
  "query_type": "image",
  "results": [
    {
      "product_id": "ABC123",
      "product_name": "Blue Denim Jeans",
      "brand": "Levi's",
      "selling_price": 2999,
      "feature_image_s3": "https://...",
      "similarity_score": 0.234,
      "rank": 1
    }
  ],
  "total_results": 10,
  "llm_reasoning": "Based on your uploaded image...",
  "timestamp": "2025-11-01T12:00:00"
}
```

---

### Text Search
**POST** `/api/search/text`

Search for fashion items using text description.

**Request Body:**
```json
{
  "query": "blue denim jeans, skinny fit",
  "top_k": 10,
  "include_reasoning": true,
  "hf_token": "your_hf_token"
}
```

**Response:** Same as Image Search

---

### Multimodal Search
**POST** `/api/search/multimodal`

Search using both image and text (combined multimodal embedding).

**Form Data:**
- `image` (file, required): Image file
- `query` (string, optional): Text description
- `top_k` (int, optional): Number of results
- `include_reasoning` (bool, optional): Generate LLM reasoning
- `hf_token` (string, optional): Hugging Face API token

**Response:** Same as Image Search

---

### Update Embeddings
**POST** `/api/embeddings/update`

Update embeddings for specific products.

**Request Body:**
```json
{
  "product_ids": ["ABC123", "DEF456"],
  "force_update": false
}
```

**Response:**
```json
{
  "status": "success",
  "updated_count": 2,
  "failed_count": 0,
  "total_requested": 2,
  "timestamp": "2025-11-01T12:00:00"
}
```

---

### Batch Update Embeddings
**POST** `/api/embeddings/batch`

Batch update embeddings for multiple products.

**Request Body:**
```json
{
  "start_idx": 0,
  "end_idx": 100,
  "batch_size": 50
}
```

**Response:**
```json
{
  "status": "success",
  "updated_count": 100,
  "failed_count": 0,
  "total_processed": 100,
  "timestamp": "2025-11-01T12:00:00"
}
```

---

### Get Statistics
**GET** `/api/stats`

Get dataset and index statistics.

**Response:**
```json
{
  "total_products": 1000,
  "total_embeddings": 100,
  "embedding_dimension": 896,
  "index_type": "IndexFlatL2",
  "models": {
    "clip": "openai/clip-vit-base-patch32",
    "text": "sentence-transformers/all-MiniLM-L6-v2",
    "llm": "google/gemma-3-27b-it"
  },
  "timestamp": "2025-11-01T12:00:00"
}
```

---

## 🧠 Architecture

### Multimodal Embedding (896D)

The system uses a hybrid embedding approach combining two powerful models:

1. **CLIP (ViT-B/32)** - 512D
   - Image embeddings from visual features
   - Text embeddings from CLIP's text encoder
   - Trained on 400M image-text pairs

2. **SentenceTransformer (all-MiniLM-L6-v2)** - 384D
   - Specialized text embeddings
   - Better semantic understanding for fashion descriptions

**Final Embedding:** [CLIP 512D | SentenceTransformer 384D] = **896D**

### FAISS IndexFlatL2

- Uses L2 (Euclidean) distance for similarity
- Exact nearest neighbor search
- Fast retrieval even with large datasets
- Ready for expansion to IVF or HNSW for scalability

### LLM Reasoning (Gemma-3)

- Analyzes top-K search results
- Explains WHY products match the query
- Provides outfit completion suggestions
- Styling tips for different occasions

---

## 📊 Dataset

Current dataset includes:
- **Dresses**: Various styles, brands, prices
- **Jeans**: Different fits, washes, styles

The system is designed to be easily expandable to other fashion categories.

---

## 🔧 Configuration

### Environment Variables

```bash
# Protobuf compatibility (if needed)
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Flask settings
FLASK_ENV=development
FLASK_DEBUG=True
```

### File Paths

- **Data**: `Data/` folder
- **Assets**: `Assets/` folder (FAISS index, embeddings)
- **Uploads**: `uploads/` folder (temporary image storage)
- **Templates**: `templates/` folder (HTML)
- **Static**: `static/` folder (CSS, JS)

---

## 🚀 Performance

### Embedding Generation
- **Image**: ~100-200ms (GPU) / ~500-1000ms (CPU)
- **Text**: ~50-100ms
- **Multimodal**: ~150-300ms (GPU)

### Search
- **FAISS L2**: <10ms for 10K vectors
- **Top-K retrieval**: O(1) with IndexFlatL2

### LLM Reasoning
- **Gemma-3 via API**: 5-15 seconds
- Optional (can be disabled for faster results)

---

## 🛠️ Troubleshooting

### Protobuf Error
If you see `TypeError: Descriptors cannot be created directly`:
```bash
pip install protobuf==3.20.3
$env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python'
```

### CUDA Out of Memory
Reduce batch size or use CPU:
```python
device = torch.device('cpu')
```

### Missing Dependencies
```bash
pip install -r requirements_flask.txt --upgrade
```

---

## 📝 License

MIT License

---

## 👥 Contributors

- Your Name - MMAL Project CA4

---

## 🙏 Acknowledgments

- OpenAI CLIP
- Sentence Transformers
- FAISS
- Hugging Face
- Gemma LLM
