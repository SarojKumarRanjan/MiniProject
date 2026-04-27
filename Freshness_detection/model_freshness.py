import cv2
import numpy as np
import os
from roboflow import Roboflow

# Global variable to cache the Roboflow model
_rf_model = None

def get_roboflow_model():
    global _rf_model
    if _rf_model is None:
        try:
            # Initialize Roboflow with the API key from your provided code
            rf = Roboflow(api_key="RBMCiagFraeIHPvptwcS")
            project = rf.workspace().project("freshness-fruits-and-vegetables")
            _rf_model = project.version(7).model
            print("Successfully initialized Roboflow model.")
        except Exception as e:
            print(f"Error initializing Roboflow: {e}")
            return None
    return _rf_model

def predict_freshness(image_path):
    # Try to get the Roboflow model
    model = get_roboflow_model()
    
    if model is None:
        return "Fresh (API Offline)"

    try:
        # Read and prepare the image
        image = cv2.imread(image_path)
        if image is None:
            return "Error loading image"
            
        # Perform inference using Roboflow
        # We use the same confidence and overlap from your code
        response = model.predict(image, confidence=40, overlap=30).json()
        predictions = response.get('predictions', [])

        if not predictions:
            return "No items detected"

        # Sort by confidence to get the most certain prediction
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        top_prediction = predictions[0]
        
        label = top_prediction.get('class', 'Unknown')
        confidence = top_prediction.get('confidence', 0)
        
        print(f"Roboflow detection: {label} ({confidence:.2f})")
        return label, confidence
        
    except Exception as e:
        print(f"Roboflow inference error: {e}")
        return "Detection error", 0.0
