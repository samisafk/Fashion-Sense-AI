// Fashion Sense AI - Frontend JavaScript

// Global state
let currentMode = 'image';
let currentImage = null;
let currentImageMulti = null;

// API Base URL
const API_BASE = '';

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
    loadStats();
});

// Event Listeners
function initializeEventListeners() {
    // Tab switching
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => switchMode(tab.dataset.mode));
    });

    // Image upload - single
    const uploadArea = document.getElementById('upload-area');
    const imageInput = document.getElementById('image-input');
    
    uploadArea.addEventListener('click', () => imageInput.click());
    imageInput.addEventListener('change', (e) => handleImageSelect(e, false));
    
    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('drag-over');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        if (e.dataTransfer.files.length) {
            handleImageSelect({ target: { files: e.dataTransfer.files } }, false);
        }
    });

    // Image upload - multimodal
    const uploadAreaMulti = document.getElementById('upload-area-multi');
    const imageInputMulti = document.getElementById('image-input-multi');
    
    uploadAreaMulti.addEventListener('click', () => imageInputMulti.click());
    imageInputMulti.addEventListener('change', (e) => handleImageSelect(e, true));

    // Slider
    const topKSlider = document.getElementById('top-k');
    const topKValue = document.getElementById('top-k-value');
    topKSlider.addEventListener('input', (e) => {
        topKValue.textContent = e.target.value;
    });

    // Search button
    document.getElementById('search-btn').addEventListener('click', handleSearch);

    // Update embeddings button
    document.getElementById('update-embeddings-btn').addEventListener('click', updateEmbeddings);
}

// Switch search mode
function switchMode(mode) {
    currentMode = mode;
    
    // Update tabs
    document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.mode === mode);
    });
    
    // Show/hide modes
    document.getElementById('image-mode').style.display = mode === 'image' ? 'block' : 'none';
    document.getElementById('text-mode').style.display = mode === 'text' ? 'block' : 'none';
    document.getElementById('multimodal-mode').style.display = mode === 'multimodal' ? 'block' : 'none';
}

// Handle image selection
function handleImageSelect(event, isMultimodal) {
    const file = event.target.files[0];
    if (!file) return;
    
    // Validate file
    if (!file.type.startsWith('image/')) {
        alert('Please select a valid image file');
        return;
    }
    
    if (file.size > 16 * 1024 * 1024) {
        alert('Image size must be less than 16MB');
        return;
    }
    
    // Store file
    if (isMultimodal) {
        currentImageMulti = file;
    } else {
        currentImage = file;
    }
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        if (isMultimodal) {
            document.getElementById('preview-image-multi').src = e.target.result;
            document.getElementById('preview-image-multi').style.display = 'block';
            document.querySelector('#upload-area-multi .upload-placeholder-small').style.display = 'none';
        } else {
            document.getElementById('preview-image').src = e.target.result;
            document.getElementById('preview-image').style.display = 'block';
            document.querySelector('#upload-area .upload-placeholder').style.display = 'none';
        }
    };
    reader.readAsDataURL(file);
}

// Handle search
async function handleSearch() {
    const searchBtn = document.getElementById('search-btn');
    const btnText = searchBtn.querySelector('.btn-text');
    const btnLoader = searchBtn.querySelector('.btn-loader');
    
    // Get parameters
    const topK = parseInt(document.getElementById('top-k').value);
    const includeReasoning = document.getElementById('include-reasoning').checked;
    const hfToken = document.getElementById('hf-token').value;
    
    // Validate inputs
    if (currentMode === 'image' && !currentImage) {
        alert('Please upload an image');
        return;
    }
    
    if (currentMode === 'text' && !document.getElementById('text-query').value.trim()) {
        alert('Please enter a search query');
        return;
    }
    
    if (currentMode === 'multimodal' && !currentImageMulti) {
        alert('Please upload an image for multimodal search');
        return;
    }
    
    if (includeReasoning && !hfToken) {
        if (!confirm('HF Token is required for AI reasoning. Continue without reasoning?')) {
            return;
        }
    }
    
    // Show loading
    searchBtn.disabled = true;
    btnText.style.display = 'none';
    btnLoader.style.display = 'inline-flex';
    
    try {
        let response;
        
        if (currentMode === 'image') {
            response = await searchByImage(currentImage, topK, includeReasoning, hfToken);
        } else if (currentMode === 'text') {
            const query = document.getElementById('text-query').value;
            response = await searchByText(query, topK, includeReasoning, hfToken);
        } else {
            const query = document.getElementById('text-query-multi').value;
            response = await searchMultimodal(currentImageMulti, query, topK, includeReasoning, hfToken);
        }
        
        displayResults(response);
        
    } catch (error) {
        console.error('Search error:', error);
        alert(`Search failed: ${error.message}`);
    } finally {
        // Hide loading
        searchBtn.disabled = false;
        btnText.style.display = 'inline';
        btnLoader.style.display = 'none';
    }
}

// API Calls
async function searchByImage(imageFile, topK, includeReasoning, hfToken) {
    const formData = new FormData();
    formData.append('image', imageFile);
    formData.append('top_k', topK);
    formData.append('include_reasoning', includeReasoning);
    if (hfToken) formData.append('hf_token', hfToken);
    
    const response = await fetch(`${API_BASE}/api/search/image`, {
        method: 'POST',
        body: formData
    });
    
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Search failed');
    }
    
    return await response.json();
}

async function searchByText(query, topK, includeReasoning, hfToken) {
    const response = await fetch(`${API_BASE}/api/search/text`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            query,
            top_k: topK,
            include_reasoning: includeReasoning,
            hf_token: hfToken
        })
    });
    
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Search failed');
    }
    
    return await response.json();
}

async function searchMultimodal(imageFile, query, topK, includeReasoning, hfToken) {
    const formData = new FormData();
    formData.append('image', imageFile);
    formData.append('query', query);
    formData.append('top_k', topK);
    formData.append('include_reasoning', includeReasoning);
    if (hfToken) formData.append('hf_token', hfToken);
    
    const response = await fetch(`${API_BASE}/api/search/multimodal`, {
        method: 'POST',
        body: formData
    });
    
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Search failed');
    }
    
    return await response.json();
}

// Display results
function displayResults(data) {
    // Show results section
    document.getElementById('results-section').style.display = 'block';
    
    // Scroll to results
    document.getElementById('results-section').scrollIntoView({ behavior: 'smooth' });
    
    // Update count
    document.getElementById('results-count').textContent = 
        `${data.total_results} result${data.total_results !== 1 ? 's' : ''}`;
    
    // Display LLM reasoning if available
    if (data.llm_reasoning) {
        document.getElementById('reasoning-card').style.display = 'block';
        document.getElementById('reasoning-content').textContent = data.llm_reasoning;
    } else {
        document.getElementById('reasoning-card').style.display = 'none';
    }
    
    // Display products
    const grid = document.getElementById('products-grid');
    grid.innerHTML = '';
    
    if (data.results && data.results.length > 0) {
        data.results.forEach(product => {
            grid.appendChild(createProductCard(product));
        });
    } else {
        grid.innerHTML = '<p style="text-align: center; color: #6b7280;">No products found. Try a different search.</p>';
    }
}

// Create product card
function createProductCard(product) {
    const card = document.createElement('div');
    card.className = 'product-card';
    
    const similarityPercent = (100 - (product.similarity_score * 10)).toFixed(1);
    
    card.innerHTML = `
        <img src="${product.feature_image_s3 || ''}" 
             alt="${product.product_name || 'Product'}" 
             class="product-image"
             onerror="this.src='https://via.placeholder.com/250x250?text=No+Image'">
        <div class="product-info">
            <span class="product-rank">#${product.rank}</span>
            <div class="product-name">${product.product_name || 'Unknown Product'}</div>
            <div class="product-brand">${product.brand || 'Unknown Brand'}</div>
            <div class="product-footer">
                <span class="product-price">₹${product.selling_price || 'N/A'}</span>
                <span class="product-score">${similarityPercent}% match</span>
            </div>
        </div>
    `;
    
    return card;
}

// Load statistics
async function loadStats() {
    try {
        const response = await fetch(`${API_BASE}/api/stats`);
        if (response.ok) {
            const stats = await response.json();
            document.getElementById('total-products').textContent = stats.total_products || 0;
            document.getElementById('total-embeddings').textContent = stats.total_embeddings || 0;
        }
    } catch (error) {
        console.error('Failed to load stats:', error);
    }
}

// Update embeddings
async function updateEmbeddings() {
    const btn = document.getElementById('update-embeddings-btn');
    const originalText = btn.textContent;
    
    if (!confirm('This will update embeddings for 10% of the dataset. This may take a few minutes. Continue?')) {
        return;
    }
    
    btn.disabled = true;
    btn.textContent = 'Updating embeddings...';
    
    try {
        // Get 10% of products
        const statsResponse = await fetch(`${API_BASE}/api/stats`);
        const stats = await statsResponse.json();
        const totalProducts = stats.total_products;
        const batchSize = Math.ceil(totalProducts * 0.1);
        
        const response = await fetch(`${API_BASE}/api/embeddings/batch`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                start_idx: 0,
                end_idx: batchSize,
                batch_size: 50
            })
        });
        
        if (response.ok) {
            const result = await response.json();
            alert(`Successfully updated ${result.updated_count} embeddings!\nFailed: ${result.failed_count}`);
            loadStats(); // Refresh stats
        } else {
            const error = await response.json();
            throw new Error(error.error || 'Update failed');
        }
        
    } catch (error) {
        console.error('Update error:', error);
        alert(`Update failed: ${error.message}`);
    } finally {
        btn.disabled = false;
        btn.textContent = originalText;
    }
}
