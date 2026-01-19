"""
CLIP-Based Visual Logo Detection
Uses OpenAI's CLIP model for brand logo recognition
"""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class CLIPLogoDetector:
    """
    Visual brand logo detection using CLIP (Contrastive Language-Image Pre-training).
    
    Uses precomputed embeddings of brand logos for fast similarity matching.
    """
    
    def __init__(self, min_confidence: float = 0.50, verbose: bool = False):
        """
        Initialize CLIP logo detector.
        
        Args:
            min_confidence: Minimum confidence threshold (0.0-1.0)
            verbose: Enable verbose logging
        """
        self.script_dir = Path(__file__).parent
        self.min_confidence = max(0.0, min(1.0, min_confidence))
        self.verbose = verbose
        
        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        
        # Load CLIP model
        self.model = None
        self.preprocess = None
        self.device = None
        self._load_clip_model()
        
        # Load brand embeddings
        self.brand_embeddings = {}
        self.brands_data = self._load_brands_data()
        self._load_or_compute_embeddings()
    
    def _load_clip_model(self):
        """Load CLIP model (ViT-B/32)"""
        try:
            import torch
            import clip
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading CLIP model on {self.device}...")
            
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            logger.info("✅ CLIP model loaded successfully")
            
        except ImportError:
            logger.error("CLIP not installed. Install with: pip install git+https://github.com/openai/CLIP.git")
            raise
        except Exception as e:
            logger.error(f"Error loading CLIP model: {e}")
            raise
    
    def _load_brands_data(self) -> Dict:
        """Load brands.json"""
        brands_file = self.script_dir / "brands.json"
        
        if not brands_file.exists():
            logger.error(f"brands.json not found at {brands_file}")
            return {}
        
        with open(brands_file, 'r', encoding='utf-8') as f:
            brands = json.load(f)
        
        logger.info(f"Loaded {len(brands)} brands from database")
        return brands
    
    def _get_logo_paths(self, brand_id: str) -> List[Path]:
        """Get all logo image paths for a brand"""
        brands_dir = self.script_dir / "brands" / brand_id
        
        if not brands_dir.exists():
            return []
        
        # Find all image files
        logo_paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.webp']:
            logo_paths.extend(brands_dir.glob(ext))
        
        return logo_paths
    
    def _compute_image_embedding(self, image_path: Path) -> np.ndarray:
        """Compute CLIP embedding for an image"""
        import torch
        from PIL import Image
        
        try:
            image = Image.open(image_path).convert('RGB')
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            
            return image_features.cpu().numpy()[0]
            
        except Exception as e:
            logger.error(f"Error computing embedding for {image_path}: {e}")
            return None
    
    def _load_or_compute_embeddings(self):
        """Load precomputed embeddings or compute them"""
        embeddings_file = self.script_dir / "brand_embeddings.pkl"
        
        # Try to load existing embeddings
        if embeddings_file.exists():
            try:
                with open(embeddings_file, 'rb') as f:
                    self.brand_embeddings = pickle.load(f)
                logger.info(f"Loaded precomputed embeddings for {len(self.brand_embeddings)} brands")
                return
            except Exception as e:
                logger.warning(f"Failed to load embeddings: {e}. Recomputing...")
        
        # Compute embeddings for all brand logos
        logger.info("Computing CLIP embeddings for brand logos...")
        
        for brand_id, brand_data in self.brands_data.items():
            logo_paths = self._get_logo_paths(brand_id)
            
            if not logo_paths:
                logger.warning(f"No logos found for brand: {brand_id}")
                continue
            
            embeddings = []
            for logo_path in logo_paths:
                embedding = self._compute_image_embedding(logo_path)
                if embedding is not None:
                    embeddings.append(embedding)
                    logger.debug(f"Computed embedding for {logo_path.name}")
            
            if embeddings:
                # Average all logo embeddings for this brand
                avg_embedding = np.mean(embeddings, axis=0)
                # Normalize
                avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
                
                self.brand_embeddings[brand_id] = {
                    'embedding': avg_embedding,
                    'name': brand_data['name'],
                    'num_logos': len(embeddings)
                }
                logger.info(f"✅ {brand_data['name']}: {len(embeddings)} logos averaged")
        
        # Save embeddings for future use
        try:
            with open(embeddings_file, 'wb') as f:
                pickle.dump(self.brand_embeddings, f)
            logger.info(f"Saved embeddings to {embeddings_file}")
        except Exception as e:
            logger.error(f"Failed to save embeddings: {e}")
    
    def detect(self, image_path: str) -> Dict:
        """
        Detect brand logo using CLIP visual similarity.
        
        Args:
            image_path: Path to product image
            
        Returns:
            Dictionary with detection results
        """
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            return {'matches': [], 'total_brands_detected': 0, 'detection_method': 'visual_clip'}
        
        if not self.brand_embeddings:
            logger.error("No brand embeddings available")
            return {'matches': [], 'total_brands_detected': 0, 'detection_method': 'visual_clip'}
        
        logger.info(f"Running CLIP visual detection on: {image_path}")
        
        # Compute embedding for input image
        query_embedding = self._compute_image_embedding(Path(image_path))
        
        if query_embedding is None:
            logger.error("Failed to compute image embedding")
            return {'matches': [], 'total_brands_detected': 0, 'detection_method': 'visual_clip'}
        
        # Compute similarities with all brand embeddings
        similarities = []
        
        for brand_id, brand_info in self.brand_embeddings.items():
            brand_embedding = brand_info['embedding']
            
            # Cosine similarity (already normalized)
            similarity = float(np.dot(query_embedding, brand_embedding))
            
            similarities.append({
                'brand_id': brand_id,
                'brand': brand_info['name'],
                'confidence': round(similarity, 2),
                'method': 'visual_clip'
            })
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Filter by confidence threshold
        matches = [s for s in similarities if s['confidence'] >= self.min_confidence]
        
        # Format results
        formatted_matches = []
        for match in matches:
            formatted_matches.append({
                'brand': match['brand'],
                'confidence': match['confidence'],
                'method': match['method'],
                'matched_text': []  # Visual detection doesn't use text
            })
        
        logger.info(f"CLIP detected {len(formatted_matches)} brands above {self.min_confidence} confidence")
        if formatted_matches:
            logger.info(f"Top match: {formatted_matches[0]['brand']} ({formatted_matches[0]['confidence']:.2f})")
        
        return {
            'matches': formatted_matches,
            'total_brands_detected': len(formatted_matches),
            'detection_method': 'visual_clip',
            'min_confidence_used': self.min_confidence
        }


def detect_logo_visual(image_path: str, min_confidence: float = 0.50, verbose: bool = False) -> Dict:
    """
    Convenience function for CLIP-based logo detection.
    
    Args:
        image_path: Path to product image
        min_confidence: Minimum confidence threshold
        verbose: Enable verbose logging
        
    Returns:
        Detection results dictionary
    """
    detector = CLIPLogoDetector(min_confidence=min_confidence, verbose=verbose)
    return detector.detect(image_path)
