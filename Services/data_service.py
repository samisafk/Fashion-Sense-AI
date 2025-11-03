"""
Data Service - Handles product data loading and retrieval
"""

import pandas as pd
import os


class DataService:
    def __init__(self):
        self.df = None
        self._ready = False
        self.data_paths = [
            'Data/dresses_bd_processed_data.csv',
            'Data/jeans_bd_processed_data.csv'
        ]
        
    def initialize(self):
        """Load product data from CSV files"""
        print("Loading product data...")
        
        dataframes = []
        for path in self.data_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                dataframes.append(df)
                print(f"  ✓ Loaded {len(df)} products from {path}")
            else:
                print(f"  ⚠ File not found: {path}")
        
        if not dataframes:
            print("  ⚠ No data files found, creating empty dataframe")
            self.df = pd.DataFrame()
        else:
            self.df = pd.concat(dataframes, ignore_index=True)
            
            # Clean price fields - extract INR value from dictionary string
            if 'selling_price' in self.df.columns:
                def extract_price(price_str):
                    """Extract numeric price from dictionary string like "{'INR': 474848.9539}" """
                    if pd.isna(price_str) or price_str == '':
                        return None
                    try:
                        # Try to evaluate the string as a dict
                        if isinstance(price_str, str) and '{' in price_str:
                            import ast
                            price_dict = ast.literal_eval(price_str)
                            if isinstance(price_dict, dict) and 'INR' in price_dict:
                                return float(price_dict['INR'])
                        # If already a number, return it
                        return float(price_str)
                    except:
                        return None
                
                self.df['selling_price'] = self.df['selling_price'].apply(extract_price)
            
            # Clean MRP field the same way
            if 'mrp' in self.df.columns:
                def extract_price(price_str):
                    """Extract numeric price from dictionary string"""
                    if pd.isna(price_str) or price_str == '':
                        return None
                    try:
                        if isinstance(price_str, str) and '{' in price_str:
                            import ast
                            price_dict = ast.literal_eval(price_str)
                            if isinstance(price_dict, dict) and 'INR' in price_dict:
                                return float(price_dict['INR'])
                        return float(price_str)
                    except:
                        return None
                
                self.df['mrp'] = self.df['mrp'].apply(extract_price)
            
            # Fill missing values (but not prices, keep them as None/NaN)
            non_numeric_cols = self.df.select_dtypes(exclude=['number']).columns
            self.df[non_numeric_cols] = self.df[non_numeric_cols].fillna('')
            
            print(f"✓ Total products loaded: {len(self.df)}")
        
        self._ready = True
    
    def is_ready(self):
        """Check if service is initialized"""
        return self._ready
    
    def get_products_by_ids(self, product_ids):
        """
        Get product details by product IDs
        
        Args:
            product_ids: list of product IDs
        
        Returns:
            list of product dictionaries
        """
        if not self._ready or self.df is None or self.df.empty:
            return []
        
        # Filter dataframe
        products = self.df[self.df['product_id'].isin(product_ids)]
        
        # Maintain order of input product_ids
        products = products.set_index('product_id')
        ordered_products = []
        
        for pid in product_ids:
            if pid in products.index:
                product_dict = products.loc[pid].to_dict()
                product_dict['product_id'] = pid
                ordered_products.append(product_dict)
        
        return ordered_products
    
    def get_product_by_id(self, product_id):
        """
        Get a single product by ID
        
        Args:
            product_id: product ID
        
        Returns:
            product dictionary or None
        """
        if not self._ready or self.df is None or self.df.empty:
            return None
        
        products = self.df[self.df['product_id'] == product_id]
        
        if products.empty:
            return None
        
        return products.iloc[0].to_dict()
    
    def get_all_products(self):
        """
        Get all products
        
        Returns:
            list of product dictionaries
        """
        if not self._ready or self.df is None or self.df.empty:
            return []
        
        return self.df.to_dict('records')
    
    def get_total_products(self):
        """Get total number of products"""
        if not self._ready or self.df is None:
            return 0
        return len(self.df)
    
    def search_by_text(self, query, limit=100):
        """
        Simple text search in product names and descriptions
        
        Args:
            query: search query string
            limit: maximum number of results
        
        Returns:
            list of product dictionaries
        """
        if not self._ready or self.df is None or self.df.empty:
            return []
        
        query_lower = query.lower()
        
        # Search in product_name and description columns
        mask = (
            self.df['product_name'].str.lower().str.contains(query_lower, na=False) |
            self.df['description'].str.lower().str.contains(query_lower, na=False)
        )
        
        results = self.df[mask].head(limit)
        
        return results.to_dict('records')
    
    def get_products_by_category(self, category, limit=None):
        """
        Get products by category
        
        Args:
            category: category name
            limit: maximum number of results
        
        Returns:
            list of product dictionaries
        """
        if not self._ready or self.df is None or self.df.empty:
            return []
        
        if 'category' not in self.df.columns:
            return []
        
        products = self.df[self.df['category'].str.lower() == category.lower()]
        
        if limit:
            products = products.head(limit)
        
        return products.to_dict('records')
    
    def get_products_by_brand(self, brand, limit=None):
        """
        Get products by brand
        
        Args:
            brand: brand name
            limit: maximum number of results
        
        Returns:
            list of product dictionaries
        """
        if not self._ready or self.df is None or self.df.empty:
            return []
        
        if 'brand' not in self.df.columns:
            return []
        
        products = self.df[self.df['brand'].str.lower() == brand.lower()]
        
        if limit:
            products = products.head(limit)
        
        return products.to_dict('records')
    
    def get_random_products(self, n=10):
        """
        Get random products
        
        Args:
            n: number of products to return
        
        Returns:
            list of product dictionaries
        """
        if not self._ready or self.df is None or self.df.empty:
            return []
        
        n = min(n, len(self.df))
        products = self.df.sample(n=n)
        
        return products.to_dict('records')
    
    def get_statistics(self):
        """Get dataset statistics"""
        if not self._ready or self.df is None or self.df.empty:
            return {}
        
        stats = {
            'total_products': len(self.df),
            'columns': list(self.df.columns),
        }
        
        if 'category' in self.df.columns:
            stats['categories'] = self.df['category'].value_counts().to_dict()
        
        if 'brand' in self.df.columns:
            stats['top_brands'] = self.df['brand'].value_counts().head(10).to_dict()
        
        if 'selling_price' in self.df.columns:
            stats['price_range'] = {
                'min': float(self.df['selling_price'].min()),
                'max': float(self.df['selling_price'].max()),
                'mean': float(self.df['selling_price'].mean())
            }
        
        return stats
