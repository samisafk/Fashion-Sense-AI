# 👗 Fashion Sense AI

## (Fashion Visual Search & Personalized Styling Assistant)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Flask](https://img.shields.io/badge/Built%20with-Flask-000000?logo=flask)
![HuggingFace](https://img.shields.io/badge/LLM-Gemma--3-orange?logo=huggingface)
![FAISS](https://img.shields.io/badge/Search-FAISS-green?logo=facebook)
![CLIP](https://img.shields.io/badge/Image%20Encoder-CLIP-lightgrey?logo=openai)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

A modern **AI-powered Flask web application** with REST API that helps users **search for visually similar fashion products** using **896D multimodal embeddings** (CLIP + SentenceTransformer), perform similarity search with **FAISS IndexFlatL2**, and receive **LLM-powered outfit reasoning** using **Gemma-3**.

---

## 🖼️ UI Screenshots

### 🎨 Flask Web Interface - Main Dashboard
![Fashion Sense AI - Main Interface](Src/FireShot%20Capture%20002%20-%20Fashion%20Sense%20AI%20-%20Multimodal%20Search%20-%20%5Blocalhost%5D.png)

*Modern single-page application with three search modes: Image Search, Text Search, and Multimodal Search*

### 🔍 Search Results with Product Grid
![Search Results - Product Grid](Src/FireShot%20Capture%20001%20-%20Fashion%20Sense%20AI%20-%20Multimodal%20Search%20-%20%5Blocalhost%5D.png)

*Top matching products displayed with images, names, brands, prices, and similarity scores*

---

## 🧠 Project Overview

This Flask-based REST API system uses **multimodal embeddings** to power intelligent fashion search:

* **🖼️ Image Search**: Upload a fashion image to find visually similar products using CLIP embeddings (512D)
* **📝 Text Search**: Query with natural language like *"red evening dress"* using SentenceTransformer embeddings (384D)
* **🔄 Multimodal Search**: Combine image + text for more precise results using 896D combined embeddings
* **🤖 LLM Reasoning**: Get intelligent outfit suggestions and styling advice powered by Gemma-3 LLM
* **⚡ Fast Vector Search**: FAISS IndexFlatL2 with L2 distance for efficient similarity search on 17,000+ products
* **🔌 REST API**: Complete API with 9 endpoints for integration into any application

Perfect for **e-commerce platforms**, **fashion recommendation systems**, or **AI-powered styling applications**.

---

## 🚀 Key Features

| Feature                       | Description                                                           |
| ----------------------------- | --------------------------------------------------------------------- |
| 🔍 Multimodal Search          | Image, text, or combined search with 896D embeddings                  |
| 🧠 LLM Reasoning              | Uses **Gemma-3** via Hugging Face API for intelligent outfit advice   |
| � FAISS IndexFlatL2          | Fast L2 distance-based similarity search on 17,000+ products          |
| 🎨 896D Embeddings            | CLIP (512D) + SentenceTransformer (384D) combined representation     |
| � REST API                   | 9 endpoints for search, stats, health, and embedding management       |
| ⚡ Batch Processing           | Generate and update embeddings for entire dataset                     |
| 🌐 Modern Web UI              | Single-page application with three search modes                       |
| ✅ Modular Architecture       | Service-based design for easy testing and extension                   |

---

## 🗂️ Project Structure

```bash
Fashion-Sense-AI/
├── flask_app.py                  # Main Flask REST API server
├── generate_embeddings.py        # Script to generate 896D embeddings for dataset
├── run_flask.ps1                 # PowerShell script to run Flask app
├── requirements_flask.txt        # Flask dependencies
├── FLASK_README.md               # Detailed API documentation
├── QUICK_START.md                # Quick start guide
├── Assets/                       # FAISS index and embeddings
│   ├── faiss_index_896d.index   # FAISS IndexFlatL2 (896D, 17,474 vectors)
│   └── product_id_mapping.pkl   # Product ID mappings
├── Data/                         # Product datasets
│   ├── dresses_bd_processed_data.csv  (14,609 products)
│   └── jeans_bd_processed_data.csv    (2,874 products)
├── Services/                     # Modular service layer
│   ├── embedding_service.py     # 896D multimodal embeddings (CLIP + SBERT)
│   ├── faiss_service.py         # FAISS IndexFlatL2 vector search
│   ├── llm_service.py           # Gemma-3 LLM reasoning
│   └── data_service.py          # Product data management
├── templates/                    # HTML templates
│   └── index.html               # Modern SPA interface
├── static/                       # Static assets
│   ├── css/style.css            # Styling
│   └── js/app.js                # Frontend logic
├── Modules/                      # Legacy Streamlit modules
├── Test_Images/                  # Sample test images
└── Notebooks/                    # Jupyter notebooks
```

---

## 🛠️ Tech Stack

* **Backend**: [Flask 3.0.0](https://flask.palletsprojects.com/) with REST API
* **Image Embeddings**: [CLIP (ViT-B/32)](https://github.com/openai/CLIP) - 512D
* **Text Embeddings**: [SentenceTransformers (all-MiniLM-L6-v2)](https://www.sbert.net/) - 384D
* **Combined Embeddings**: 896D multimodal representation (512D + 384D)
* **Vector Search**: [FAISS IndexFlatL2](https://github.com/facebookresearch/faiss) with L2 distance
* **LLM Engine**: [Gemma-3-27b-it](https://huggingface.co/google/gemma-2-27b-it) via Hugging Face Inference API
* **Frontend**: Modern HTML/CSS/JavaScript SPA
* **Deep Learning**: PyTorch 2.9.0, Transformers 4.57.1
* **Dataset**: 17,483 fashion products (14,609 dresses + 2,874 jeans)

---

## 📦 Installation & Setup

> **Recommended Python version**: `3.10+` (Tested on Python 3.10.11)

### 🚀 Quick Start

1. **Clone the repository**

```bash
git clone https://github.com/samisafk/Fashion-Sense-AI.git
cd Fashion-Sense-AI
```

2. **Create and activate virtual environment**

```bash
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. **Install Flask dependencies**

```bash
pip install -r requirements_flask.txt
```

4. **Set environment variable (Windows PowerShell)**

```bash
$env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python'
```

5. **Generate embeddings for the dataset** (First-time setup)

```bash
# Quick mode - Generate embeddings for 50 products (for testing)
python generate_embeddings.py --quick

# Full mode - Generate embeddings for all 17,483 products (takes 2-4 hours)
python generate_embeddings.py
```

6. **Run the Flask application**

```bash
# Windows PowerShell
.\run_flask.ps1

# Or manually
python flask_app.py
```

7. **Access the application**

Open your browser and navigate to: **http://localhost:5000**

8. **(Optional) Configure Hugging Face Token for LLM**

* Get your free token from: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
* Set it in the UI settings or as environment variable: `HF_TOKEN=your_token_here`

---

## 🧭 How to Use

### 🌐 Web Interface

Once the Flask app is running at **http://localhost:5000**:

1. **🖼️ Image Search**
   - Click "Upload Image" and select a fashion item photo
   - Adjust the number of results (default: 10)
   - Click "Search by Image"
   - View visually similar products with similarity scores

2. **📝 Text Search**
   - Enter a description like `"red evening dress"` or `"blue denim jacket"`
   - Adjust the number of results
   - Click "Search by Text"
   - View semantically similar products

3. **🔄 Multimodal Search**
   - Upload an image AND enter text description
   - Combines visual and textual features for more precise results
   - Click "Search Multimodal"
   - View products matching both image and text criteria

4. **🤖 LLM Reasoning** (Optional - requires HF token)
   - After search results appear, view AI-generated outfit suggestions
   - Get styling advice and complementary item recommendations

### 🔌 API Usage

The REST API provides 9 endpoints for programmatic access:

```bash
# Health check
GET /api/health

# Get statistics
GET /api/stats

# Search by image
POST /api/search/image
Content-Type: multipart/form-data
Body: image (file), top_k (optional)

# Search by text
POST /api/search/text
Content-Type: application/json
Body: {"query": "red dress", "top_k": 10, "use_llm": false}

# Multimodal search
POST /api/search/multimodal
Content-Type: multipart/form-data
Body: image (file), text (string), top_k (optional)

# Update single embedding
POST /api/embeddings/update

# Batch update embeddings
POST /api/embeddings/batch
```

See **[FLASK_README.md](FLASK_README.md)** for complete API documentation.

---

## 🧪 Sample Use Cases

---

## 🖼️ UI Screenshots

### 🔍 Search Results with Product Grid
![Search Results - Product Grid](Src/FireShot%20Capture%20002%20-%20Fashion%20Sense%20AI%20-%20Multimodal%20Search%20-%20%5Blocalhost%5D.png)

*Top matching products displayed with images, names, brands, prices, and similarity scores*

---

### 🎨 Flask Web Interface - Main Dashboard
![Fashion Sense AI - Main Interface](Src/FireShot%20Capture%20001%20-%20Fashion%20Sense%20AI%20-%20Multimodal%20Search%20-%20%5Blocalhost%5D.png)

*Modern single-page application with three search modes: Image Search, Text Search, and Multimodal Search*

--- 

## 🧪 Sample Use Cases

* 🔍 A user uploads a black denim jacket → finds 10 similar styles instantly.
* 🛍️ A shopper queries for `"Off Shoulder dresses"` → retrieves relevant dresses.
* 🤖 Fake user browsing history is generated → recommendations are shown.
* 🎨 LLM completes the outfit with accessories and layering items based on the uploaded look.

---

## 🏗️ Architecture

### Embedding Pipeline (896D)

```
Image Input → CLIP (ViT-B/32) → 512D vector
                                              ↘
Text Input → SentenceTransformer → 384D vector → Concatenate → 896D embedding
```

### Search Flow

```
Query (Image/Text/Both) → Generate 896D embedding → FAISS IndexFlatL2 search
→ Top-K products → (Optional) LLM reasoning → Return results + suggestions
```

### Service Architecture

- **EmbeddingService**: Handles CLIP + SentenceTransformer encoding
- **FAISSService**: Manages vector indexing and similarity search
- **LLMService**: Generates outfit reasoning using Gemma-3
- **DataService**: Loads and manages product catalog

## 🔐 Hugging Face Token

* The `HF_TOKEN` is optional but required for LLM-powered outfit suggestions
* Get your free token: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
* Set it as environment variable or configure in the UI
* Token is used only for Gemma API inference, never stored permanently

---

## 📊 Performance Metrics

- **Dataset Size**: 17,483 products (14,609 dresses + 2,874 jeans)
- **Embedding Dimension**: 896D (CLIP 512D + SBERT 384D)
- **Index Type**: FAISS IndexFlatL2
- **Total Vectors**: 17,474 embeddings
- **Search Time**: < 100ms for top-10 results
- **Embedding Generation**: ~10-15 seconds per product (including image download)
- **Memory Usage**: ~2.5GB with all models loaded

## 🎯 Future Improvements

* 🔒 User authentication with persistent search history
* 🛒 E-commerce integration with add-to-cart functionality
* 📱 Mobile-responsive design improvements
* 🎨 Advanced filters (price range, brand, color, size)
* 🗣️ Voice-based search using speech-to-text
* 🔄 Real-time embedding updates for new products
* 🚀 GPU acceleration for faster embedding generation
* 📈 Analytics dashboard for search patterns
* 🌍 Multi-language support
* 🤖 Enhanced LLM prompts for better recommendations

## 📚 Documentation

- **[FLASK_README.md](FLASK_README.md)** - Complete API documentation with examples
- **[QUICK_START.md](QUICK_START.md)** - Quick start guide for developers
- **[README_CONVERSION.md](README_CONVERSION.md)** - Detailed conversion overview from Streamlit to Flask

## � Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the MIT License.

---

##  Acknowledgments

- OpenAI for the CLIP model
- Sentence-Transformers team for text embeddings
- Facebook Research for FAISS
- Hugging Face for model hosting and LLM API
- Google for the Gemma model

---

⭐ **Star this repo if you found it helpful!**
