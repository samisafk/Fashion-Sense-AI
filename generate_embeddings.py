"""
Generate initial embeddings for the Fashion Sense AI dataset
This script will create embeddings for products and build the FAISS index
"""

import sys
import os
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Services.embedding_service import EmbeddingService
from Services.faiss_service import FAISSService
from Services.data_service import DataService

def generate_embeddings(batch_size=10, max_products=None):
    """
    Generate embeddings for all products in the dataset
    
    Args:
        batch_size: Number of products to process at once
        max_products: Maximum number of products to process (None = all)
    """
    print("=" * 60)
    print("Fashion Sense AI - Embedding Generation")
    print("=" * 60)
    print()
    
    # Initialize services
    print("Initializing services...")
    embedding_service = EmbeddingService()
    embedding_service.initialize()
    
    faiss_service = FAISSService()
    faiss_service.initialize()
    
    data_service = DataService()
    data_service.initialize()
    
    print()
    
    # Get all products
    all_products = data_service.get_all_products()
    
    if not all_products:
        print("❌ No products found in dataset!")
        print("   Make sure Data/ folder contains CSV files")
        return
    
    if max_products:
        all_products = all_products[:max_products]
    
    total_products = len(all_products)
    print(f"📊 Found {total_products} products to process")
    print()
    
    # Generate embeddings in batches
    print("🔄 Generating embeddings...")
    print(f"   Batch size: {batch_size}")
    print(f"   Device: {embedding_service.device}")
    print()
    
    successful = 0
    failed = 0
    
    for i in tqdm(range(0, len(all_products), batch_size), desc="Processing batches"):
        batch = all_products[i:i+batch_size]
        
        for product in batch:
            try:
                product_id = product['product_id']
                
                # Check if already embedded
                if product_id in faiss_service.id_to_idx:
                    continue
                
                # Get image URL and text description
                image_url = product.get('feature_image_s3', '')
                product_name = product.get('product_name', '')
                description = product.get('description', '')
                brand = product.get('brand', '')
                
                # Combine text fields
                text_desc = f"{product_name} {brand} {description}".strip()
                
                if not image_url:
                    print(f"⚠️  Skipping {product_id}: No image URL")
                    failed += 1
                    continue
                
                # Generate embedding
                embedding = embedding_service.encode_product(image_url, text_desc)
                
                # Add to FAISS index
                faiss_service.add_embedding(product_id, embedding)
                
                successful += 1
                
            except Exception as e:
                print(f"❌ Error processing {product.get('product_id', 'unknown')}: {str(e)}")
                failed += 1
        
        # Save after each batch
        if successful > 0 and successful % (batch_size * 5) == 0:
            print(f"\n💾 Saving progress... ({successful} embeddings)")
            faiss_service.save_index()
    
    # Final save
    print()
    print("💾 Saving final index...")
    faiss_service.save_index()
    
    # Summary
    print()
    print("=" * 60)
    print("✅ Embedding Generation Complete!")
    print("=" * 60)
    print(f"   Total products: {total_products}")
    print(f"   ✓ Successfully embedded: {successful}")
    print(f"   ✗ Failed: {failed}")
    print(f"   📁 Index saved to: {faiss_service.index_path}")
    print("=" * 60)
    
    return successful, failed


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate embeddings for Fashion Sense AI')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for processing')
    parser.add_argument('--max-products', type=int, default=None, help='Maximum products to process')
    parser.add_argument('--quick', action='store_true', help='Quick mode: process first 50 products')
    
    args = parser.parse_args()
    
    if args.quick:
        print("🚀 Quick mode: Processing first 50 products")
        max_products = 50
    else:
        max_products = args.max_products
    
    try:
        generate_embeddings(batch_size=args.batch_size, max_products=max_products)
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        print("   Partial embeddings have been saved")
    except Exception as e:
        print(f"\n\n❌ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
