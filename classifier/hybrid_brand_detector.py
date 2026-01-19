"""
Hybrid Brand Detection Pipeline
Combines OCR-based text detection with CLIP visual logo detection
"""

import logging
from typing import Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class HybridBrandDetector:
    """
    Hybrid brand detection using OCR + CLIP fallback.
    
    Pipeline:
    1. Run OCR detection (EasyOCR + text matching)
    2. If OCR confidence >= 0.7 → return OCR result
    3. If OCR confidence < 0.7 → fallback to CLIP visual detection
    """
    
    def __init__(self, ocr_threshold: float = 0.70, clip_threshold: float = 0.50, verbose: bool = False):
        """
        Initialize hybrid detector.
        
        Args:
            ocr_threshold: Minimum OCR confidence to accept (default: 0.70)
            clip_threshold: Minimum CLIP confidence to accept (default: 0.50)
            verbose: Enable verbose logging
        """
        self.ocr_threshold = ocr_threshold
        self.clip_threshold = clip_threshold
        self.verbose = verbose
        
        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        
        # Initialize detectors
        self.ocr_detector = None
        self.clip_detector = None
        self._init_detectors()
    
    def _init_detectors(self):
        """Initialize OCR and CLIP detectors"""
        try:
            from classifier.brand_detector import OCRBrandDetector
            self.ocr_detector = OCRBrandDetector(
                min_confidence=0.50,  # Lower threshold for OCR (we'll check later)
                verbose=self.verbose
            )
            logger.info("✅ OCR detector initialized")
        except Exception as e:
            logger.error(f"Failed to initialize OCR detector: {e}")
            raise
        
        try:
            from classifier.visual_logo_detector import CLIPLogoDetector
            self.clip_detector = CLIPLogoDetector(
                min_confidence=self.clip_threshold,
                verbose=self.verbose
            )
            logger.info("✅ CLIP detector initialized")
        except Exception as e:
            logger.warning(f"CLIP detector not available: {e}")
            logger.warning("Falling back to OCR-only detection")
            self.clip_detector = None
    
    def detect(self, image_path: str, ocr_text: Optional[str] = None) -> Dict:
        """
        Hybrid brand detection pipeline.
        
        Args:
            image_path: Path to product image
            ocr_text: Pre-extracted OCR text (optional)
            
        Returns:
            Detection results with method indicator
        """
        logger.info("="*60)
        logger.info("HYBRID BRAND DETECTION PIPELINE")
        logger.info("="*60)
        logger.info(f"Image: {image_path}")
        logger.info(f"OCR text provided: {len(ocr_text) if ocr_text else 0} chars")
        logger.info(f"OCR threshold: {self.ocr_threshold}")
        logger.info(f"CLIP threshold: {self.clip_threshold}")
        
        # Step 1: Run OCR Detection
        logger.info("\n[STEP 1] Running OCR detection...")
        ocr_result = self.ocr_detector.detect(image_path=image_path, ocr_text=ocr_text)
        
        # Check OCR confidence
        ocr_best_confidence = 0.0
        ocr_best_match = None
        
        if ocr_result['matches']:
            ocr_best_match = ocr_result['matches'][0]
            ocr_best_confidence = ocr_best_match['confidence']
            logger.info(f"OCR best match: {ocr_best_match['brand']} ({ocr_best_confidence:.2f})")
        else:
            logger.info("OCR: No matches found")
        
        # Decision: Use OCR or fallback to CLIP?
        if ocr_best_confidence >= self.ocr_threshold:
            logger.info(f"\n✅ OCR confidence {ocr_best_confidence:.2f} >= {self.ocr_threshold}")
            logger.info(f"DECISION: Using OCR result")
            logger.info("="*60)
            return ocr_result
        
        # Step 2: Fallback to CLIP Visual Detection
        logger.info(f"\n⚠️ OCR confidence {ocr_best_confidence:.2f} < {self.ocr_threshold}")
        logger.info(f"DECISION: Falling back to CLIP visual detection")
        
        if not self.clip_detector:
            logger.warning("CLIP detector not available. Returning OCR result anyway.")
            logger.info("="*60)
            return ocr_result
        
        logger.info("\n[STEP 2] Running CLIP visual detection...")
        clip_result = self.clip_detector.detect(image_path)
        
        # Check CLIP confidence
        if clip_result['matches']:
            clip_best = clip_result['matches'][0]
            logger.info(f"CLIP best match: {clip_best['brand']} ({clip_best['confidence']:.2f})")
            
            # Return only top CLIP match (highest confidence)
            clip_result['matches'] = [clip_best]
            clip_result['total_brands_detected'] = 1
            
            logger.info(f"\n✅ Using CLIP visual result (top match only)")
            logger.info("="*60)
            return clip_result
        
        # Fallback: Return OCR result if CLIP also failed
        logger.warning("Both OCR and CLIP have low confidence")
        logger.info("Returning best available result (OCR)")
        logger.info("="*60)
        return ocr_result


def detect_brands_hybrid(
    image_path: str, 
    ocr_text: Optional[str] = None,
    ocr_threshold: float = 0.70,
    clip_threshold: float = 0.50,
    verbose: bool = False
) -> Dict:
    """
    Convenience function for hybrid brand detection.
    
    Args:
        image_path: Path to product image
        ocr_text: Pre-extracted OCR text (optional)
        ocr_threshold: Min OCR confidence to accept (default: 0.70)
        clip_threshold: Min CLIP confidence to accept (default: 0.50)
        verbose: Enable verbose logging
        
    Returns:
        Detection results dictionary
        
    Example:
        >>> result = detect_brands_hybrid(
        ...     image_path="product.jpg",
        ...     ocr_text="Some text",
        ...     ocr_threshold=0.70,
        ...     verbose=True
        ... )
        >>> print(result['matches'][0]['method'])  # 'ocr' or 'visual_clip'
    """
    detector = HybridBrandDetector(
        ocr_threshold=ocr_threshold,
        clip_threshold=clip_threshold,
        verbose=verbose
    )
    return detector.detect(image_path=image_path, ocr_text=ocr_text)
