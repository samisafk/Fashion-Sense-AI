"""
Embedding Service - Generates 896D multimodal embeddings
CLIP (ViT-B/32): 512D for images
SentenceTransformer (all-MiniLM-L6-v2): 384D for text
Combined: 896D multimodal representation
"""

import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
import requests
from io import BytesIO


class EmbeddingService:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.clip_model = None
        self.clip_processor = None
        self.text_model = None
        self._ready = False
        
    def initialize(self):
        """Initialize the embedding models"""
        print("Loading CLIP model (ViT-B/32 - 512D)...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.eval()
        
        print("Loading SentenceTransformer model (all-MiniLM-L6-v2 - 384D)...")
        self.text_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.text_model.to(self.device)
        
        self._ready = True
        print(f"✓ Embedding models loaded on {self.device}")
        print(f"  - CLIP embedding dimension: 512D")
        print(f"  - Text embedding dimension: 384D")
        print(f"  - Combined dimension: 896D")
    
    def is_ready(self):
        """Check if service is initialized"""
        return self._ready
    
    def _load_image(self, image_source):
        """
        Load image from file path or URL
        Args:
            image_source: file path (str) or URL (str)
        Returns:
            PIL Image
        """
        if image_source.startswith('http://') or image_source.startswith('https://'):
            # Load from URL
            response = requests.get(image_source, timeout=10)
            image = Image.open(BytesIO(response.content))
        else:
            # Load from file path
            image = Image.open(image_source)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    
    def encode_image_clip(self, image_source):
        """
        Generate 512D CLIP embedding from image
        Args:
            image_source: file path or URL
        Returns:
            numpy array of shape (512,)
        """
        if not self._ready:
            raise RuntimeError("Embedding service not initialized. Call initialize() first.")
        
        image = self._load_image(image_source)
        
        # Process image with CLIP
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
            # Normalize the features
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        
        embedding = image_features.cpu().numpy().flatten()
        
        return embedding  # Shape: (512,)
    
    def encode_text_sbert(self, text):
        """
        Generate 384D text embedding using SentenceTransformer
        Args:
            text: text string
        Returns:
            numpy array of shape (384,)
        """
        if not self._ready:
            raise RuntimeError("Embedding service not initialized. Call initialize() first.")
        
        if not text or not text.strip():
            # Return zero vector if no text
            return np.zeros(384, dtype=np.float32)
        
        embedding = self.text_model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        return embedding  # Shape: (384,)
    
    def encode_text_clip(self, text):
        """
        Generate 512D CLIP text embedding
        Args:
            text: text string
        Returns:
            numpy array of shape (512,)
        """
        if not self._ready:
            raise RuntimeError("Embedding service not initialized. Call initialize() first.")
        
        if not text or not text.strip():
            return np.zeros(512, dtype=np.float32)
        
        # Process text with CLIP (truncate to max_length=77 to avoid errors)
        inputs = self.clip_processor(
            text=[text], 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=77
        ).to(self.device)
        
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)
            # Normalize the features
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        
        embedding = text_features.cpu().numpy().flatten()
        
        return embedding  # Shape: (512,)
    
    def encode_image(self, image_source, include_text=False, text=""):
        """
        Generate 896D embedding for image
        CLIP (512D) + SentenceTransformer text (384D) = 896D
        
        Args:
            image_source: file path or URL
            include_text: whether to include text embedding
            text: optional text to encode
        Returns:
            numpy array of shape (896,)
        """
        clip_embedding = self.encode_image_clip(image_source)  # 512D
        
        if include_text and text:
            text_embedding = self.encode_text_sbert(text)  # 384D
        else:
            text_embedding = np.zeros(384, dtype=np.float32)
        
        # Concatenate: [CLIP 512D | Text 384D] = 896D
        combined_embedding = np.concatenate([clip_embedding, text_embedding])
        
        # Normalize the combined embedding
        norm = np.linalg.norm(combined_embedding)
        if norm > 0:
            combined_embedding = combined_embedding / norm
        
        return combined_embedding  # Shape: (896,)
    
    def encode_text(self, text):
        """
        Generate 896D embedding for text query
        CLIP text (512D) + SentenceTransformer (384D) = 896D
        
        Args:
            text: text string
        Returns:
            numpy array of shape (896,)
        """
        clip_text_embedding = self.encode_text_clip(text)  # 512D
        sbert_embedding = self.encode_text_sbert(text)  # 384D
        
        # Concatenate: [CLIP 512D | SBERT 384D] = 896D
        combined_embedding = np.concatenate([clip_text_embedding, sbert_embedding])
        
        # Normalize
        norm = np.linalg.norm(combined_embedding)
        if norm > 0:
            combined_embedding = combined_embedding / norm
        
        return combined_embedding  # Shape: (896,)
    
    def encode_multimodal(self, image_source, text):
        """
        Generate 896D multimodal embedding combining image and text
        
        Strategy: Average the image and text embeddings in each modality
        - CLIP image (512D) + CLIP text (512D) -> averaged -> 512D
        - SBERT text (384D) -> 384D
        Total: 896D
        
        Args:
            image_source: file path or URL
            text: text string
        Returns:
            numpy array of shape (896,)
        """
        # Get CLIP embeddings
        clip_image_emb = self.encode_image_clip(image_source)  # 512D
        clip_text_emb = self.encode_text_clip(text) if text else np.zeros(512, dtype=np.float32)  # 512D
        
        # Average CLIP embeddings (both image and text in same space)
        clip_combined = (clip_image_emb + clip_text_emb) / 2.0  # 512D
        
        # Get text-specific embedding
        sbert_emb = self.encode_text_sbert(text) if text else np.zeros(384, dtype=np.float32)  # 384D
        
        # Concatenate: [CLIP 512D | SBERT 384D] = 896D
        combined_embedding = np.concatenate([clip_combined, sbert_emb])
        
        # Normalize
        norm = np.linalg.norm(combined_embedding)
        if norm > 0:
            combined_embedding = combined_embedding / norm
        
        return combined_embedding  # Shape: (896,)
    
    def encode_product(self, image_source, text_description):
        """
        Generate 896D embedding for a product (image + text description)
        
        Args:
            image_source: product image path or URL
            text_description: product name + description
        Returns:
            numpy array of shape (896,)
        """
        return self.encode_multimodal(image_source, text_description)
    
    def batch_encode_products(self, products, batch_size=32):
        """
        Encode multiple products in batches
        
        Args:
            products: list of dicts with 'image_url' and 'text_description'
            batch_size: number of products to process at once
        Returns:
            numpy array of shape (n_products, 896)
        """
        embeddings = []
        
        for i in range(0, len(products), batch_size):
            batch = products[i:i+batch_size]
            
            for product in batch:
                try:
                    embedding = self.encode_product(
                        product['image_url'],
                        product['text_description']
                    )
                    embeddings.append(embedding)
                except Exception as e:
                    print(f"Error encoding product: {str(e)}")
                    # Add zero vector for failed products
                    embeddings.append(np.zeros(896, dtype=np.float32))
        
        return np.array(embeddings, dtype=np.float32)
    
    def get_embedding_dimension(self):
        """Get the dimension of generated embeddings"""
        return 896
