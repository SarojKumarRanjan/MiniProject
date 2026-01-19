"""
OCR-Based Brand Recognition System
Uses advanced OCR techniques (Tesseract + EasyOCR) for brand detection from product images
"""

import os
import json
import logging
from pathlib import Path
import re
from typing import List, Dict, Optional, Tuple

# Configure logging
logger = logging.getLogger(__name__)


class OCRBrandDetector:
    """
    OCR-based brand detection using text matching against known brand database.
    
    This detector uses two OCR methods:
    1. Tesseract OCR (pre-extracted text from app.py)
    2. EasyOCR (advanced logo text recognition)
    
    Brand matching uses exact and fuzzy text matching with confidence scoring.
    """
    
    def __init__(self, min_confidence: float = 0.50, verbose: bool = False):
        """
        Initialize the OCR-based brand detector.
        
        Args:
            min_confidence: Minimum confidence threshold for matches (0.0-1.0). Default: 0.50
            verbose: Enable verbose logging. Default: False
        """
        self.script_dir = Path(__file__).parent
        self.min_confidence = max(0.0, min(1.0, min_confidence))  # Clamp between 0 and 1
        self.verbose = verbose
        
        # Set logging level
        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        
        # Load brand database
        self.brands = self._load_brand_data()
        logger.info(f"Loaded {len(self.brands)} brands from database")
        
    def _load_brand_data(self) -> Dict:
        """
        Load brand information from brands.json.
        
        Returns:
            Dictionary of brand data with brand_id as keys
            
        Raises:
            ValueError: If brands.json is invalid or missing required fields
        """
        brands_file = self.script_dir / "brands.json"
        
        # Default brand database (fallback)
        default_brands = {
            "amul": {"name": "Amul", "aliases": ["Amul"]},
            "parle": {"name": "Parle", "aliases": ["Parle", "Parle-G", "Parle G"]},
            "britannia": {"name": "Britannia", "aliases": ["Britannia"]},
            "nestle": {"name": "Nestlé", "aliases": ["Nestle", "Nestlé"]},
            "aashirvaad": {"name": "Aashirvaad", "aliases": ["Aashirvaad", "ITC"]},
            "hul": {"name": "Hindustan Unilever", "aliases": ["HUL", "Hindustan Unilever", "Unilever"]},
            "dabur": {"name": "Dabur", "aliases": ["Dabur"]},
            "patanjali": {"name": "Patanjali", "aliases": ["Patanjali"]},
            "godrej": {"name": "Godrej", "aliases": ["Godrej"]},
            "marico": {"name": "Marico", "aliases": ["Marico"]},
            "colgate": {"name": "Colgate", "aliases": ["Colgate"]},
            "himalaya": {"name": "Himalaya", "aliases": ["Himalaya"]},
            "cocacola": {"name": "Coca-Cola", "aliases": ["Coca Cola", "Coca-Cola", "Coke"]},
            "pepsi": {"name": "PepsiCo", "aliases": ["Pepsi", "PepsiCo"]},
        }
        
        if brands_file.exists():
            try:
                with open(brands_file, 'r', encoding='utf-8') as f:
                    brands = json.load(f)
                
                # Validate brand data
                for brand_id, brand_data in brands.items():
                    if 'name' not in brand_data:
                        raise ValueError(f"Brand '{brand_id}' missing required 'name' field")
                    if 'aliases' not in brand_data:
                        logger.warning(f"Brand '{brand_id}' missing 'aliases' field, using name only")
                        brand_data['aliases'] = [brand_data['name']]
                
                logger.info(f"Loaded brands from {brands_file}")
                return brands
                
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in {brands_file}: {e}")
                logger.warning("Using default brand database")
                return default_brands
            except Exception as e:
                logger.error(f"Error loading brands.json: {e}")
                logger.warning("Using default brand database")
                return default_brands
        else:
            # Create default brands file
            logger.warning(f"brands.json not found at {brands_file}, creating default")
            try:
                with open(brands_file, 'w', encoding='utf-8') as f:
                    json.dump(default_brands, f, indent=2, ensure_ascii=False)
                logger.info("Created default brands.json")
            except Exception as e:
                logger.error(f"Failed to create brands.json: {e}")
            
            return default_brands
    
    def _calculate_confidence(self, brand_name: str, match_type: str) -> float:
        """
        Calculate confidence score based on brand name length and match type.
        
        Args:
            brand_name: The brand name that was matched
            match_type: 'exact' or 'fuzzy'
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        name_length = len(brand_name)
        
        if match_type == 'exact':
            # Exact match: 0.75-0.90 range
            # Longer brand names get higher confidence (less likely to be false positive)
            base_confidence = 0.75
            length_bonus = min((name_length / 30.0) * 0.15, 0.15)
            confidence = base_confidence + length_bonus
            return min(confidence, 0.90)
        else:  # fuzzy
            # Fuzzy match: 0.60-0.70 range
            base_confidence = 0.60
            length_bonus = min((name_length / 40.0) * 0.10, 0.10)
            confidence = base_confidence + length_bonus
            return min(confidence, 0.70)
    
    def detect_from_text(self, ocr_text: str) -> List[Dict]:
        """
        Detect brands from OCR text using pattern matching.
        
        Uses two matching strategies:
        1. Exact match with word boundaries (higher confidence)
        2. Fuzzy substring match (lower confidence, catches OCR errors)
        
        Args:
            ocr_text: Extracted OCR text from image
            
        Returns:
            List of brand matches with confidence scores
        """
        if not ocr_text or not isinstance(ocr_text, str) or len(ocr_text.strip()) < 2:
            logger.warning("Invalid or empty OCR text provided")
            return []
        
        matches = []
        text_lower = ocr_text.lower()
        
        # Common words to ignore in fuzzy matching (prevent false positives)
        generic_words = {
            'product', 'products', 'india', 'consumer', 'limited', 'ltd',
            'company', 'co', 'pvt', 'private', 'corporation', 'corp',
            'industries', 'industry', 'foods', 'food'
        }
        
        logger.debug(f"Searching for brands in text: {ocr_text[:100]}...")
        
        for brand_id, brand_data in self.brands.items():
            # Check brand name and all aliases
            all_names = [brand_data['name']] + brand_data.get('aliases', [])
            
            for name in all_names:
                name_lower = name.lower()
                
                # Method 1: Exact match with word boundaries
                pattern = r'\b' + re.escape(name_lower) + r'\b'
                if re.search(pattern, text_lower):
                    confidence = self._calculate_confidence(name, 'exact')
                    
                    if confidence >= self.min_confidence:
                        matches.append({
                            'brand': brand_data['name'],
                            'brand_id': brand_id,
                            'confidence': round(confidence, 2),
                            'method': 'text_exact',
                            'matched_text': name
                        })
                        logger.debug(f"Exact match: {brand_data['name']} ({confidence:.2f})")
                    break
                
                # Method 2: Fuzzy match (substring, allows for OCR errors)
                elif name_lower in text_lower:
                    # Full name substring match
                    confidence = self._calculate_confidence(name, 'fuzzy')
                    
                    if confidence >= self.min_confidence:
                        matches.append({
                            'brand': brand_data['name'],
                            'brand_id': brand_id,
                            'confidence': round(confidence, 2),
                            'method': 'text_fuzzy',
                            'matched_text': name
                        })
                        logger.debug(f"Fuzzy match: {brand_data['name']} ({confidence:.2f})")
                    break
                
                # Method 3: Word-level fuzzy match (ONLY for non-generic words)
                else:
                    words_in_name = [w for w in name_lower.split() if w not in generic_words and len(w) > 2]
                    significant_matches = sum(1 for word in words_in_name if word in text_lower)
                    
                    # Require at least one significant word match AND it must be >50% of significant words
                    if significant_matches > 0 and len(words_in_name) > 0:
                        match_ratio = significant_matches / len(words_in_name)
                        if match_ratio >= 0.5:  # At least 50% of significant words must match
                            confidence = self._calculate_confidence(name, 'fuzzy')
                            confidence = confidence * 0.9  # Slightly reduce confidence for partial matches
                            
                            if confidence >= self.min_confidence:
                                matches.append({
                                    'brand': brand_data['name'],
                                    'brand_id': brand_id,
                                    'confidence': round(confidence, 2),
                                    'method': 'text_fuzzy',
                                    'matched_text': name
                                })
                                logger.debug(f"Word-level fuzzy match: {brand_data['name']} ({confidence:.2f})")
                            break
        
        logger.info(f"Text matching found {len(matches)} potential brands")
        return matches
    
    def detect_from_image_ocr(self, image_path: str) -> List[Dict]:
        """
        Use EasyOCR for advanced logo text recognition.
        
        EasyOCR is better at recognizing stylized text in logos compared to Tesseract.
        Requires: pip install easyocr
        
        Args:
            image_path: Path to the product image
            
        Returns:
            List of brand matches from EasyOCR text
        """
        if not image_path or not os.path.exists(image_path):
            logger.warning(f"Invalid image path: {image_path}")
            return []
        
        try:
            import easyocr
            
            logger.info("Initializing EasyOCR reader...")
            reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            
            # Read text from image
            logger.info(f"Processing image with EasyOCR: {image_path}")
            results = reader.readtext(image_path)
            
            # Filter by confidence and combine text
            detected_texts = [text for (bbox, text, conf) in results if conf > 0.3]
            combined_text = ' '.join(detected_texts)
            
            logger.info(f"EasyOCR detected {len(detected_texts)} text segments")
            logger.debug(f"Combined text: {combined_text[:100]}...")
            
            # Use text matching on EasyOCR results
            return self.detect_from_text(combined_text)
            
        except ImportError:
            logger.warning("EasyOCR not available. Install with: pip install easyocr")
            return []
        except Exception as e:
            logger.error(f"EasyOCR processing error: {e}", exc_info=True)
            return []
    
    def _merge_duplicates(self, matches: List[Dict]) -> List[Dict]:
        """
        Merge duplicate brand detections, keeping the highest confidence match.
        
        Args:
            matches: List of brand matches
            
        Returns:
            Deduplicated list with best match per brand
        """
        brand_best_matches = {}
        
        for match in matches:
            brand_id = match['brand_id']
            
            if brand_id not in brand_best_matches:
                brand_best_matches[brand_id] = match
            else:
                # Keep match with higher confidence
                if match['confidence'] > brand_best_matches[brand_id]['confidence']:
                    logger.debug(f"Replacing {brand_id}: {brand_best_matches[brand_id]['confidence']:.2f} -> {match['confidence']:.2f}")
                    brand_best_matches[brand_id] = match
        
        return list(brand_best_matches.values())
    
    def detect(self, image_path: Optional[str] = None, ocr_text: Optional[str] = None) -> Dict:
        """
        Main detection method combining Tesseract OCR text and EasyOCR.
        
        Detection pipeline:
        1. Match brands from pre-extracted Tesseract OCR text
        2. Perform EasyOCR on image for logo text recognition
        3. Merge duplicate detections
        4. Filter by minimum confidence threshold
        5. Sort by confidence (descending)
        
        Args:
            image_path: Path to product image (optional, for EasyOCR)
            ocr_text: Pre-extracted OCR text from Tesseract (optional)
            
        Returns:
            Dictionary containing:
            - matches: List of detected brands with confidence scores
            - total_brands_detected: Number of unique brands found
            - detection_method: 'ocr_based'
            - min_confidence_used: Threshold applied
            
        Example:
            >>> detector = OCRBrandDetector(min_confidence=0.6)
            >>> result = detector.detect(image_path="product.jpg", ocr_text="Nestle Maggi")
            >>> print(result['matches'])
            [{'brand': 'Nestlé', 'confidence': 0.85, ...}]
        """
        logger.info("="*60)
        logger.info("OCR-BASED BRAND DETECTION")
        logger.info("="*60)
        logger.info(f"Image path: {image_path}")
        logger.info(f"Image exists: {os.path.exists(image_path) if image_path else False}")
        logger.info(f"OCR text length: {len(ocr_text) if ocr_text else 0} chars")
        logger.info(f"Min confidence threshold: {self.min_confidence}")
        
        all_matches = []
        
        # Method 1: Tesseract OCR text matching
        if ocr_text:
            logger.info("[1] Processing Tesseract OCR text...")
            text_matches = self.detect_from_text(ocr_text)
            logger.info(f"    Tesseract matches: {len(text_matches)}")
            all_matches.extend(text_matches)
        else:
            logger.info("[1] Skipping Tesseract (no OCR text provided)")
        
        # Method 2: EasyOCR for logo text recognition
        if image_path and os.path.exists(image_path):
            logger.info("[2] Processing with EasyOCR...")
            easyocr_matches = self.detect_from_image_ocr(image_path)
            logger.info(f"    EasyOCR matches: {len(easyocr_matches)}")
            all_matches.extend(easyocr_matches)
        else:
            logger.info("[2] Skipping EasyOCR (no valid image path)")
        
        # Merge duplicates and keep highest confidence
        merged_matches = self._merge_duplicates(all_matches)
        logger.info(f"After merging duplicates: {len(merged_matches)} unique brands")
        
        # Sort by confidence (descending)
        sorted_matches = sorted(merged_matches, key=lambda x: x['confidence'], reverse=True)
        
        # Format for API response
        formatted_matches = []
        for match in sorted_matches:
            formatted_matches.append({
                'brand': match['brand'],
                'confidence': match['confidence'],
                'method': match['method'],
                'matched_text': [match.get('matched_text', '')],  # Return as array for frontend
            })
        
        logger.info(f"Final result: {len(formatted_matches)} brands detected")
        for match in formatted_matches:
            logger.info(f"  - {match['brand']}: {match['confidence']:.2f} ({match['method']})")
        
        return {
            'matches': formatted_matches,
            'total_brands_detected': len(formatted_matches),
            'detection_method': 'ocr_based',
            'min_confidence_used': self.min_confidence
        }


def detect_brands(image_path: Optional[str] = None, 
                  ocr_text: Optional[str] = None,
                  min_confidence: float = 0.50,
                  verbose: bool = False) -> Dict:
    """
    Convenience function for brand detection.
    
    Args:
        image_path: Path to product image (for EasyOCR)
        ocr_text: Pre-extracted OCR text from Tesseract
        min_confidence: Minimum confidence threshold (0.0-1.0). Default: 0.50
        verbose: Enable verbose logging. Default: False
    
    Returns:
        Detection results dictionary with matches and metadata
        
    Example:
        >>> result = detect_brands(
        ...     image_path="product.jpg",
        ...     ocr_text="Amul Butter",
        ...     min_confidence=0.6
        ... )
        >>> print(f"Found {result['total_brands_detected']} brands")
    """
    detector = OCRBrandDetector(min_confidence=min_confidence, verbose=verbose)
    return detector.detect(image_path=image_path, ocr_text=ocr_text)
