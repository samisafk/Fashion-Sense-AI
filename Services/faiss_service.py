"""
FAISS Service - Vector similarity search using IndexFlatL2
Handles 896D embeddings with L2 distance metric
"""

import numpy as np
import faiss
import os
import pickle


class FAISSService:
    def __init__(self, index_path='Assets/faiss_index_896d.index', 
                 mapping_path='Assets/product_id_mapping.pkl'):
        self.dimension = 896
        self.index = None
        self.product_ids = []
        self.id_to_idx = {}
        self.index_path = index_path
        self.mapping_path = mapping_path
        self._ready = False
        
    def initialize(self):
        """Initialize or load existing FAISS index"""
        if os.path.exists(self.index_path) and os.path.exists(self.mapping_path):
            print(f"Loading existing FAISS index from {self.index_path}...")
            self.load_index()
        else:
            print("Creating new FAISS IndexFlatL2 (896D)...")
            self.create_new_index()
        
        self._ready = True
        print(f"✓ FAISS service ready")
        print(f"  - Index type: IndexFlatL2")
        print(f"  - Dimension: {self.dimension}D")
        print(f"  - Total vectors: {self.get_total_embeddings()}")
    
    def is_ready(self):
        """Check if service is initialized"""
        return self._ready
    
    def create_new_index(self):
        """Create a new FAISS index using L2 distance"""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.product_ids = []
        self.id_to_idx = {}
        print(f"Created new IndexFlatL2 with dimension {self.dimension}")
    
    def load_index(self):
        """Load existing FAISS index and product ID mapping"""
        try:
            self.index = faiss.read_index(self.index_path)
            
            with open(self.mapping_path, 'rb') as f:
                mapping_data = pickle.load(f)
                self.product_ids = mapping_data['product_ids']
                self.id_to_idx = mapping_data['id_to_idx']
            
            print(f"✓ Loaded FAISS index with {len(self.product_ids)} vectors")
            
        except Exception as e:
            print(f"Error loading index: {str(e)}")
            print("Creating new index...")
            self.create_new_index()
    
    def save_index(self):
        """Save FAISS index and product ID mapping to disk"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, self.index_path)
            
            # Save product ID mapping
            mapping_data = {
                'product_ids': self.product_ids,
                'id_to_idx': self.id_to_idx
            }
            with open(self.mapping_path, 'wb') as f:
                pickle.dump(mapping_data, f)
            
            print(f"✓ Saved FAISS index ({len(self.product_ids)} vectors) to {self.index_path}")
            
        except Exception as e:
            print(f"Error saving index: {str(e)}")
            raise
    
    def add_embedding(self, product_id, embedding):
        """
        Add a single embedding to the index
        
        Args:
            product_id: unique product identifier
            embedding: numpy array of shape (896,)
        """
        if not self._ready:
            raise RuntimeError("FAISS service not initialized. Call initialize() first.")
        
        # Ensure embedding is the correct shape
        embedding = np.array(embedding, dtype=np.float32).reshape(1, -1)
        
        if embedding.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embedding.shape[1]} doesn't match index dimension {self.dimension}")
        
        # Check if product already exists
        if product_id in self.id_to_idx:
            # Update existing embedding
            idx = self.id_to_idx[product_id]
            # FAISS doesn't support direct update, so we'll need to rebuild
            # For now, we'll skip duplicates
            print(f"Warning: Product {product_id} already exists in index. Skipping.")
            return
        
        # Add to index
        self.index.add(embedding)
        
        # Update mappings
        idx = len(self.product_ids)
        self.product_ids.append(product_id)
        self.id_to_idx[product_id] = idx
    
    def add_embeddings_batch(self, product_ids, embeddings):
        """
        Add multiple embeddings to the index in batch
        
        Args:
            product_ids: list of product identifiers
            embeddings: numpy array of shape (n, 896)
        """
        if not self._ready:
            raise RuntimeError("FAISS service not initialized. Call initialize() first.")
        
        embeddings = np.array(embeddings, dtype=np.float32)
        
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} doesn't match index dimension {self.dimension}")
        
        if len(product_ids) != embeddings.shape[0]:
            raise ValueError("Number of product IDs must match number of embeddings")
        
        # Filter out products that already exist
        new_ids = []
        new_embeddings = []
        
        for pid, emb in zip(product_ids, embeddings):
            if pid not in self.id_to_idx:
                new_ids.append(pid)
                new_embeddings.append(emb)
        
        if not new_ids:
            print("All products already exist in index.")
            return
        
        # Add to index
        new_embeddings = np.array(new_embeddings, dtype=np.float32)
        self.index.add(new_embeddings)
        
        # Update mappings
        start_idx = len(self.product_ids)
        for i, pid in enumerate(new_ids):
            self.product_ids.append(pid)
            self.id_to_idx[pid] = start_idx + i
        
        print(f"Added {len(new_ids)} new embeddings to index")
    
    def search(self, query_embedding, top_k=10):
        """
        Search for similar vectors using L2 distance
        
        Args:
            query_embedding: numpy array of shape (896,)
            top_k: number of nearest neighbors to return
        
        Returns:
            dict with 'product_ids' and 'distances'
        """
        if not self._ready:
            raise RuntimeError("FAISS service not initialized. Call initialize() first.")
        
        if self.index.ntotal == 0:
            return {
                'product_ids': [],
                'distances': []
            }
        
        # Ensure query is correct shape
        query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        
        if query_embedding.shape[1] != self.dimension:
            raise ValueError(f"Query dimension {query_embedding.shape[1]} doesn't match index dimension {self.dimension}")
        
        # Ensure top_k doesn't exceed index size
        top_k = min(top_k, self.index.ntotal)
        
        # Search
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Convert indices to product IDs
        result_product_ids = [self.product_ids[idx] for idx in indices[0]]
        result_distances = distances[0].tolist()
        
        return {
            'product_ids': result_product_ids,
            'distances': result_distances
        }
    
    def search_with_threshold(self, query_embedding, top_k=10, distance_threshold=None):
        """
        Search with optional distance threshold filtering
        
        Args:
            query_embedding: numpy array of shape (896,)
            top_k: number of nearest neighbors to return
            distance_threshold: maximum L2 distance (None = no threshold)
        
        Returns:
            dict with 'product_ids' and 'distances'
        """
        results = self.search(query_embedding, top_k)
        
        if distance_threshold is not None:
            # Filter by threshold
            filtered_ids = []
            filtered_distances = []
            
            for pid, dist in zip(results['product_ids'], results['distances']):
                if dist <= distance_threshold:
                    filtered_ids.append(pid)
                    filtered_distances.append(dist)
            
            return {
                'product_ids': filtered_ids,
                'distances': filtered_distances
            }
        
        return results
    
    def get_total_embeddings(self):
        """Get the total number of vectors in the index"""
        if self.index is None:
            return 0
        return self.index.ntotal
    
    def get_product_embedding(self, product_id):
        """
        Retrieve the embedding for a specific product
        
        Args:
            product_id: product identifier
        
        Returns:
            numpy array of shape (896,) or None if not found
        """
        if product_id not in self.id_to_idx:
            return None
        
        idx = self.id_to_idx[product_id]
        embedding = self.index.reconstruct(int(idx))
        
        return embedding
    
    def remove_product(self, product_id):
        """
        Remove a product from the index
        Note: FAISS doesn't support direct removal, so this requires rebuilding
        
        Args:
            product_id: product identifier
        """
        if product_id not in self.id_to_idx:
            print(f"Product {product_id} not found in index")
            return
        
        # Get all embeddings except the one to remove
        embeddings = []
        new_product_ids = []
        
        for pid in self.product_ids:
            if pid != product_id:
                idx = self.id_to_idx[pid]
                emb = self.index.reconstruct(int(idx))
                embeddings.append(emb)
                new_product_ids.append(pid)
        
        # Rebuild index
        self.create_new_index()
        
        if embeddings:
            embeddings = np.array(embeddings, dtype=np.float32)
            self.add_embeddings_batch(new_product_ids, embeddings)
        
        print(f"Removed product {product_id} and rebuilt index")
    
    def rebuild_index(self, product_ids, embeddings):
        """
        Completely rebuild the index from scratch
        
        Args:
            product_ids: list of all product identifiers
            embeddings: numpy array of shape (n, 896)
        """
        print(f"Rebuilding index with {len(product_ids)} vectors...")
        
        self.create_new_index()
        self.add_embeddings_batch(product_ids, embeddings)
        
        print("✓ Index rebuilt successfully")
    
    def get_statistics(self):
        """Get statistics about the index"""
        return {
            'total_vectors': self.get_total_embeddings(),
            'dimension': self.dimension,
            'index_type': 'IndexFlatL2',
            'metric': 'L2 (Euclidean distance)',
            'is_trained': self.index.is_trained if self.index else False
        }
