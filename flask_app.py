"""
Flask Application for Fashion Sense AI
REST API with multimodal embeddings (896D), FAISS search, and LLM reasoning
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import json
from datetime import datetime
import traceback

# Import custom services
from Services.embedding_service import EmbeddingService
from Services.faiss_service import FAISSService
from Services.llm_service import LLMService
from Services.data_service import DataService

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('Assets', exist_ok=True)

# Initialize services
embedding_service = EmbeddingService()
faiss_service = FAISSService()
llm_service = LLMService()
data_service = DataService()


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'services': {
            'embedding': embedding_service.is_ready(),
            'faiss': faiss_service.is_ready(),
            'llm': llm_service.is_ready(),
            'data': data_service.is_ready()
        }
    })


@app.route('/api/search/image', methods=['POST'])
def search_by_image():
    """
    Search for similar fashion items using an uploaded image
    Returns top-K results with similarity scores and LLM reasoning
    """
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Get parameters
        top_k = int(request.form.get('top_k', 10))
        include_reasoning = request.form.get('include_reasoning', 'true').lower() == 'true'
        hf_token = request.form.get('hf_token', '')
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Generate embedding (896D)
        embedding = embedding_service.encode_image(filepath)
        
        # Search in FAISS index
        results = faiss_service.search(embedding, top_k)
        
        # Get product details
        products = data_service.get_products_by_ids(results['product_ids'])
        
        # Add similarity scores to products
        for i, product in enumerate(products):
            product['similarity_score'] = float(results['distances'][i])
            product['rank'] = i + 1
        
        response = {
            'query_type': 'image',
            'results': products,
            'total_results': len(products),
            'timestamp': datetime.now().isoformat()
        }
        
        # Generate LLM reasoning if requested
        if include_reasoning and products and hf_token:
            reasoning = llm_service.generate_outfit_reasoning(
                products=products,
                query_type='image',
                image_path=filepath,
                hf_token=hf_token
            )
            response['llm_reasoning'] = reasoning
        
        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error in search_by_image: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/search/text', methods=['POST'])
def search_by_text():
    """
    Search for similar fashion items using text query
    Returns top-K results with similarity scores and LLM reasoning
    """
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({'error': 'No query text provided'}), 400
        
        query = data['query']
        top_k = int(data.get('top_k', 10))
        include_reasoning = data.get('include_reasoning', True)
        hf_token = data.get('hf_token', '')
        
        # Generate embedding (896D)
        embedding = embedding_service.encode_text(query)
        
        # Search in FAISS index
        results = faiss_service.search(embedding, top_k)
        
        # Get product details
        products = data_service.get_products_by_ids(results['product_ids'])
        
        # Add similarity scores to products
        for i, product in enumerate(products):
            product['similarity_score'] = float(results['distances'][i])
            product['rank'] = i + 1
        
        response = {
            'query': query,
            'query_type': 'text',
            'results': products,
            'total_results': len(products),
            'timestamp': datetime.now().isoformat()
        }
        
        # Generate LLM reasoning if requested
        if include_reasoning and products and hf_token:
            reasoning = llm_service.generate_outfit_reasoning(
                products=products,
                query_type='text',
                query_text=query,
                hf_token=hf_token
            )
            response['llm_reasoning'] = reasoning
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error in search_by_text: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/search/multimodal', methods=['POST'])
def search_multimodal():
    """
    Search using both image and text (multimodal search)
    Combines embeddings from both modalities
    """
    try:
        # Check for image
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Get parameters
        text_query = request.form.get('query', '')
        top_k = int(request.form.get('top_k', 10))
        include_reasoning = request.form.get('include_reasoning', 'true').lower() == 'true'
        hf_token = request.form.get('hf_token', '')
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Generate combined embedding
        embedding = embedding_service.encode_multimodal(filepath, text_query)
        
        # Search in FAISS index
        results = faiss_service.search(embedding, top_k)
        
        # Get product details
        products = data_service.get_products_by_ids(results['product_ids'])
        
        # Add similarity scores to products
        for i, product in enumerate(products):
            product['similarity_score'] = float(results['distances'][i])
            product['rank'] = i + 1
        
        response = {
            'query': text_query,
            'query_type': 'multimodal',
            'results': products,
            'total_results': len(products),
            'timestamp': datetime.now().isoformat()
        }
        
        # Generate LLM reasoning if requested
        if include_reasoning and products and hf_token:
            reasoning = llm_service.generate_outfit_reasoning(
                products=products,
                query_type='multimodal',
                image_path=filepath,
                query_text=text_query,
                hf_token=hf_token
            )
            response['llm_reasoning'] = reasoning
        
        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error in search_multimodal: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/embeddings/update', methods=['POST'])
def update_embeddings():
    """
    Update or add embeddings for products
    Useful for expanding the dataset
    """
    try:
        data = request.get_json()
        
        if not data or 'product_ids' not in data:
            return jsonify({'error': 'No product_ids provided'}), 400
        
        product_ids = data['product_ids']
        force_update = data.get('force_update', False)
        
        # Get products from database
        products = data_service.get_products_by_ids(product_ids)
        
        if not products:
            return jsonify({'error': 'No valid products found'}), 404
        
        # Generate embeddings for products
        updated_count = 0
        failed_count = 0
        
        for product in products:
            try:
                # Generate embedding from product image and text
                image_url = product.get('feature_image_s3', '')
                text_desc = f"{product.get('product_name', '')} {product.get('description', '')}"
                
                embedding = embedding_service.encode_product(image_url, text_desc)
                
                # Add to FAISS index
                faiss_service.add_embedding(product['product_id'], embedding)
                updated_count += 1
                
            except Exception as e:
                print(f"Failed to update embedding for {product['product_id']}: {str(e)}")
                failed_count += 1
        
        # Save updated index
        if updated_count > 0:
            faiss_service.save_index()
        
        return jsonify({
            'status': 'success',
            'updated_count': updated_count,
            'failed_count': failed_count,
            'total_requested': len(product_ids),
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        print(f"Error in update_embeddings: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/embeddings/batch', methods=['POST'])
def batch_update_embeddings():
    """
    Batch update embeddings for all products or a specific range
    Useful for initial dataset embedding
    """
    try:
        data = request.get_json() or {}
        
        start_idx = int(data.get('start_idx', 0))
        end_idx = data.get('end_idx', None)
        batch_size = int(data.get('batch_size', 100))
        
        # Get all product IDs
        all_products = data_service.get_all_products()
        
        if end_idx:
            products_to_process = all_products[start_idx:int(end_idx)]
        else:
            products_to_process = all_products[start_idx:]
        
        total = len(products_to_process)
        updated_count = 0
        failed_count = 0
        
        # Process in batches
        for i in range(0, len(products_to_process), batch_size):
            batch = products_to_process[i:i+batch_size]
            
            for product in batch:
                try:
                    image_url = product.get('feature_image_s3', '')
                    text_desc = f"{product.get('product_name', '')} {product.get('description', '')}"
                    
                    embedding = embedding_service.encode_product(image_url, text_desc)
                    faiss_service.add_embedding(product['product_id'], embedding)
                    updated_count += 1
                    
                except Exception as e:
                    print(f"Failed for {product['product_id']}: {str(e)}")
                    failed_count += 1
            
            # Save after each batch
            if updated_count % batch_size == 0:
                faiss_service.save_index()
                print(f"Progress: {updated_count}/{total} completed")
        
        # Final save
        faiss_service.save_index()
        
        return jsonify({
            'status': 'success',
            'updated_count': updated_count,
            'failed_count': failed_count,
            'total_processed': total,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        print(f"Error in batch_update_embeddings: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get statistics about the dataset and index"""
    try:
        stats = {
            'total_products': data_service.get_total_products(),
            'total_embeddings': faiss_service.get_total_embeddings(),
            'embedding_dimension': 896,
            'index_type': 'IndexFlatL2',
            'models': {
                'clip': 'openai/clip-vit-base-patch32',
                'text': 'sentence-transformers/all-MiniLM-L6-v2',
                'llm': 'google/gemma-3-27b-it'
            },
            'timestamp': datetime.now().isoformat()
        }
        return jsonify(stats)
    
    except Exception as e:
        print(f"Error in get_stats: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/static/<path:path>')
def send_static(path):
    """Serve static files"""
    return send_from_directory('static', path)


if __name__ == '__main__':
    print("=" * 60)
    print("Fashion Sense AI - Flask API Server")
    print("=" * 60)
    print("Initializing services...")
    
    # Initialize services
    try:
        embedding_service.initialize()
        faiss_service.initialize()
        data_service.initialize()
        print("✓ All services initialized successfully")
    except Exception as e:
        print(f"✗ Error initializing services: {str(e)}")
        traceback.print_exc()
    
    print("\nServer starting on http://localhost:5000")
    print("API Documentation: http://localhost:5000/api/health")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
