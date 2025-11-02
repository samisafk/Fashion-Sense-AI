"""
LLM Service - Generate outfit reasoning using Gemma-3
Explains how search results complement or enhance the user's query
"""

import requests
import json


class LLMService:
    def __init__(self, model_id="google/gemma-3-27b-it"):
        self.model_id = model_id
        self.api_url = "https://router.huggingface.co/v1/chat/completions"
        self._ready = True
        
    def initialize(self):
        """Initialize the LLM service"""
        print(f"✓ LLM service ready (Model: {self.model_id})")
    
    def is_ready(self):
        """Check if service is available"""
        return self._ready
    
    def _format_product_list(self, products):
        """Format products into a readable list for the LLM"""
        formatted = []
        for i, product in enumerate(products[:5], 1):  # Limit to top 5 for context
            formatted.append(
                f"{i}. {product.get('product_name', 'Unknown')} "
                f"(Brand: {product.get('brand', 'N/A')}, "
                f"Price: ₹{product.get('selling_price', 'N/A')}, "
                f"Similarity: {product.get('similarity_score', 0):.3f})"
            )
        return "\n".join(formatted)
    
    def generate_outfit_reasoning(self, products, query_type='image', 
                                  image_path=None, query_text=None, hf_token=None):
        """
        Generate outfit reasoning and styling suggestions based on search results
        
        Args:
            products: list of product dictionaries with similarity scores
            query_type: 'image', 'text', or 'multimodal'
            image_path: path to uploaded image (for multimodal context)
            query_text: text query (for text/multimodal context)
            hf_token: Hugging Face API token
        
        Returns:
            string with LLM-generated reasoning
        """
        if not hf_token:
            return "⚠️ HF Token required for LLM reasoning. Please provide your Hugging Face API token."
        
        if not products:
            return "No products found to generate reasoning."
        
        # Build the prompt based on query type
        product_list = self._format_product_list(products)
        
        if query_type == 'image':
            prompt = self._build_image_query_prompt(product_list, products)
        elif query_type == 'text':
            prompt = self._build_text_query_prompt(product_list, query_text, products)
        else:  # multimodal
            prompt = self._build_multimodal_prompt(product_list, query_text, products)
        
        # Call Gemma API
        try:
            response = self._call_gemma_api(prompt, image_path, hf_token)
            return response
        except Exception as e:
            return f"❌ Error generating reasoning: {str(e)}"
    
    def _build_image_query_prompt(self, product_list, products):
        """Build prompt for image-based search"""
        prompt = f"""You are an expert fashion stylist and AI assistant. A user has uploaded an image of a fashion item, 
and our AI system has found the following similar products from our database (currently containing skirts and jeans):

**Top Matching Products:**
{product_list}

**Your Task:**
1. Analyze the similarity scores and product attributes
2. Explain WHY these products are good matches (consider style, color, fit, brand positioning)
3. Provide outfit completion suggestions - what other items would complement these pieces?
4. Suggest how to style these items for different occasions (casual, formal, party, etc.)
5. Even though our current dataset has limited categories (skirts/jeans), explain how these pieces can be versatile building blocks for complete outfits

**Important:** Focus on practical styling advice, color coordination, layering suggestions, and how to create cohesive looks. 
Be creative and suggest accessories, tops, shoes, or outerwear that would complete the outfit, even if they're not in our current search results.

Provide your response in a friendly, conversational tone with clear sections."""

        return prompt
    
    def _build_text_query_prompt(self, product_list, query_text, products):
        """Build prompt for text-based search"""
        prompt = f"""You are an expert fashion stylist and AI assistant. A user searched for: "{query_text}"

Our AI system found these matching products (similarity-based ranking):

**Top Matching Products:**
{product_list}

**Your Task:**
1. Explain how these products match the user's search intent "{query_text}"
2. Highlight the key style attributes that make these good matches
3. Suggest complete outfit ideas incorporating these items
4. Recommend complementary pieces (tops, shoes, accessories, outerwear) to create full looks
5. Provide styling tips for different contexts (work, weekend, evening out, etc.)

**Note:** Our current database contains skirts and jeans, but think creatively about how to build versatile, 
stylish outfits around these core pieces.

Provide detailed, actionable fashion advice in a warm, helpful tone."""

        return prompt
    
    def _build_multimodal_prompt(self, product_list, query_text, products):
        """Build prompt for multimodal (image + text) search"""
        prompt = f"""You are an expert fashion stylist and AI assistant. A user has provided both an image AND a text query: "{query_text}"

This multimodal search combines visual similarity with semantic understanding. Here are the top matches:

**Top Matching Products:**
{product_list}

**Your Task:**
1. Explain how these products satisfy BOTH the visual style from the image AND the text description
2. Analyze the synergy between visual and textual features in the search results
3. Create complete outfit concepts that honor both the aesthetic from the image and the keywords from the text
4. Suggest complementary items (tops, footwear, accessories, jackets) to complete the look
5. Provide occasion-specific styling variations

**Dataset Context:** Our current inventory focuses on skirts and jeans, but use your fashion expertise to suggest 
how to build complete, cohesive outfits around these foundation pieces.

Deliver your advice in an engaging, professional manner with clear formatting."""

        return prompt
    
    def _call_gemma_api(self, prompt, image_path=None, hf_token=None):
        """
        Call Hugging Face API for Gemma model
        
        Args:
            prompt: text prompt for the model
            image_path: optional image path (for multimodal models)
            hf_token: Hugging Face API token
        
        Returns:
            Generated text response
        """
        headers = {
            "Authorization": f"Bearer {hf_token}",
            "Content-Type": "application/json",
        }
        
        # Build message content
        message_content = [{"type": "text", "text": prompt}]
        
        # Add image if provided (for multimodal support)
        if image_path and image_path.startswith('http'):
            message_content.append({
                "type": "image_url",
                "image_url": {"url": image_path}
            })
        
        payload = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": message_content if len(message_content) > 1 else prompt
                }
            ],
            "stream": False,
            "max_tokens": 1000,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            response.raise_for_status()
            data = response.json()
            
            return data["choices"][0]["message"]["content"]
            
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP {response.status_code}: {response.text}"
            raise Exception(error_msg)
        except requests.exceptions.Timeout:
            raise Exception("Request timeout - please try again")
        except Exception as e:
            raise Exception(f"API call failed: {str(e)}")
    
    def generate_product_description(self, product, hf_token):
        """
        Generate an enhanced product description using LLM
        
        Args:
            product: product dictionary
            hf_token: Hugging Face API token
        
        Returns:
            Enhanced description string
        """
        if not hf_token:
            return product.get('description', '')
        
        prompt = f"""As a fashion copywriter, create an engaging product description for:

Product Name: {product.get('product_name', 'Unknown')}
Brand: {product.get('brand', 'N/A')}
Category: {product.get('category', 'Fashion Item')}
Current Description: {product.get('description', 'N/A')}
Price: ₹{product.get('selling_price', 'N/A')}

Write a compelling, concise description (2-3 sentences) that highlights:
- Key style features
- Versatility and styling potential
- Why this item is a wardrobe essential

Keep it professional yet engaging."""

        try:
            return self._call_gemma_api(prompt, hf_token=hf_token)
        except Exception as e:
            return product.get('description', f'Error: {str(e)}')
    
    def generate_trend_analysis(self, products, hf_token):
        """
        Analyze trending patterns in the product set
        
        Args:
            products: list of products
            hf_token: Hugging Face API token
        
        Returns:
            Trend analysis string
        """
        if not hf_token or not products:
            return "Insufficient data for trend analysis."
        
        # Extract key attributes
        brands = [p.get('brand', '') for p in products[:10]]
        styles = [p.get('style_attributes', '') for p in products[:10]]
        
        prompt = f"""As a fashion trend analyst, analyze these top products and identify emerging trends:

Brands: {', '.join(filter(None, brands))}
Style Attributes: {', '.join(filter(None, styles))}

Provide a brief trend analysis covering:
1. Popular styles or themes
2. Color/pattern trends
3. Price point insights
4. Styling recommendations based on these trends

Keep it concise (3-4 sentences)."""

        try:
            return self._call_gemma_api(prompt, hf_token=hf_token)
        except Exception as e:
            return f"Trend analysis unavailable: {str(e)}"
