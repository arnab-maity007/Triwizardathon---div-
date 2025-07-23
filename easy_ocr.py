import json
import cv2
import numpy as np
import easyocr
import os
from PIL import Image, ImageEnhance, ImageFilter

# Initialize EasyOCR reader globally to avoid reloading
reader = easyocr.Reader(['en'], gpu=True)  # Set gpu=False if no GPU available

def preprocess_for_game_ui(image):
    """
    Specialized preprocessing for game UI text with clear backgrounds
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Scale up significantly - game UI text is often small
    height, width = gray.shape
    scale_factor = 4  # Increase scaling
    scaled = cv2.resize(gray, (width * scale_factor, height * scale_factor), interpolation=cv2.INTER_CUBIC)
    
    # Apply slight gaussian blur to smooth pixelated edges
    blurred = cv2.GaussianBlur(scaled, (3, 3), 0)
    
    # Enhance contrast specifically for white text on dark backgrounds
    # Check if background is darker than text
    mean_val = np.mean(blurred)
    if mean_val < 128:  # Dark background, light text
        # Invert for better OCR
        inverted = 255 - blurred
        return inverted
    else:
        return blurred

def preprocess_simple_threshold(image):
    """
    Simple but effective threshold for clear UI text
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Scale up
    height, width = gray.shape
    scaled = cv2.resize(gray, (width * 3, height * 3), interpolation=cv2.INTER_CUBIC)
    
    # Simple binary threshold - works well for clear text
    _, binary = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Check if we need to invert
    white_pixels = np.sum(binary == 255)
    black_pixels = np.sum(binary == 0)
    
    if black_pixels > white_pixels:  # More black pixels, probably inverted
        binary = 255 - binary
    
    return binary

def preprocess_morphology(image):
    """
    Use morphological operations to clean up text
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Scale up
    height, width = gray.shape
    scaled = cv2.resize(gray, (width * 4, height * 4), interpolation=cv2.INTER_CUBIC)
    
    # Binary threshold
    _, binary = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Check if we need to invert (text should be white on black for morphology)
    white_pixels = np.sum(binary == 255)
    black_pixels = np.sum(binary == 0)
    
    if black_pixels > white_pixels:
        binary = 255 - binary
    
    # Morphological operations to connect text components
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Remove small noise
    kernel_open = np.ones((1, 1), np.uint8)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_open)
    
    return cleaned

def extract_text_with_easyocr(image, method_name=""):
    """
    Extract text using EasyOCR - returns best result only
    """
    try:
        # Try with character restrictions first
        ocr_results = reader.readtext(
            image,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ ',
            width_ths=0.7,
            height_ths=0.7,
            detail=1,
            paragraph=False
        )
        
        if ocr_results:
            # Get the result with highest confidence
            best_result = max(ocr_results, key=lambda x: x[2])
            text = best_result[1].strip()
            confidence = best_result[2]
            
            if text and len(text) > 1:
                return text, confidence
        
        # Fallback: Try without character restrictions
        ocr_results = reader.readtext(
            image,
            width_ths=0.7,
            height_ths=0.7,
            detail=1,
            paragraph=False
        )
        
        if ocr_results:
            # Get the result with highest confidence
            best_result = max(ocr_results, key=lambda x: x[2])
            text = best_result[1].strip()
            confidence = best_result[2]
            
            # Clean up common OCR artifacts
            text = text.replace('|', 'I').replace('0', 'O').replace('5', 'S')
            
            if text and len(text) > 1:
                return text, confidence
                
    except Exception as e:
        pass
    
    return "", 0.0

def extract_game_text(image_region):
    """
    Simplified text extraction for game UI elements using EasyOCR
    """
    # Single method: Game UI optimized preprocessing
    try:
        processed = preprocess_for_game_ui(image_region)
        text, confidence = extract_text_with_easyocr(processed, "GameUI")
        
        if text:
            return text, confidence
                
    except Exception as e:
        pass
        
    # Fallback: Try original image if preprocessing fails
    try:
        text, confidence = extract_text_with_easyocr(image_region, "Original")
        return text, confidence
    except Exception as e:
        pass
    
    return "", 0.0

def run_game_ui_ocr(predictions_json_path, base_image_path="", output_json_path="game_ocr_results.json"):
    """
    Specialized OCR for game UI elements using EasyOCR - silent mode, JSON output only
    """
    
    # Load predictions data
    with open(predictions_json_path, 'r') as f:
        predictions_data = json.load(f)
    
    results = []
    
    for image_data in predictions_data:
        filename = image_data['filename']
        image_path = os.path.join(base_image_path, image_data['path'])
        predictions = image_data['result']['predictions']
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            continue
        
        image_result = {
            'image_name': filename,
            'event_boxes': []
        }
        
        # Process each event box
        for i, pred in enumerate(predictions):
            # Calculate crop coordinates with smaller padding for clearer crops
            padding = 5
            x = int(pred['x'] - pred['width'] / 2) - padding
            y = int(pred['y'] - pred['height'] / 2) - padding
            w = int(pred['width']) + (2 * padding)
            h = int(pred['height']) + (2 * padding)
            
            # Ensure coordinates are within image bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, image.shape[1] - x)
            h = min(h, image.shape[0] - y)
            
            if w <= 0 or h <= 0:
                continue
            
            # Crop the region
            cropped = image[y:y+h, x:x+w]
            
            if cropped.size == 0:
                continue
            
            # Extract text using game-specialized methods
            best_text, best_confidence = extract_game_text(cropped)
            
            # Clean and validate text
            if best_text:
                best_text = ' '.join(best_text.split())  # Clean whitespace
                best_text = best_text.upper()  # Game UI is often uppercase
                
                # Game UI specific filtering
                if (len(best_text) <= 1 or 
                    best_text in ['_', '-', '~', '=', '<', '>', '|', '/', '\\'] or
                    best_text.isspace() or
                    all(c in '.,;:!?' for c in best_text)):  # Just punctuation
                    best_text = ""
                    best_confidence = 0.0
            
            event_box_result = {
                'box_id': i,
                'coordinates': {
                    'x': pred['x'],
                    'y': pred['y'],
                    'width': pred['width'],
                    'height': pred['height']
                },
                'confidence': pred['confidence'],
                'extracted_text': best_text,
                'ocr_confidence': best_confidence
            }
            
            image_result['event_boxes'].append(event_box_result)
        
        results.append(image_result)
    
    # Save results
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return results

# Usage example:
if __name__ == "__main__":
    predictions_file = r"results\buy1_result.json"
    base_path = ""
    output_file = "game_ocr_results.json"
    
    print("Running Specialized Game UI OCR with EasyOCR...")
    print("This version is optimized for clear game interface text")
    print("Note: First run may take longer as EasyOCR downloads models")
    print()
    
    results = run_game_ui_ocr(
        predictions_file, 
        base_path, 
        output_file
    )