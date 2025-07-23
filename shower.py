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

def extract_text_with_easyocr_multi_line(image, method_name=""):
    """
    Extract multi-line text using EasyOCR - returns all detected text
    """
    try:
        # Use paragraph mode to capture multi-line text
        ocr_results = reader.readtext(
            image,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 ',  # Allow more characters
            width_ths=0.7,
            height_ths=0.7,
            detail=1,
            paragraph=True  # Enable paragraph detection to capture multiple lines
        )
        
        if ocr_results:
            # Collect all the text in the result
            full_text = " ".join([result[1] for result in ocr_results]).strip()
            confidence = sum([result[2] for result in ocr_results]) / len(ocr_results)  # Average confidence
            
            if full_text and len(full_text) > 1:
                return full_text, confidence
        
    except Exception as e:
        print(f"Error in OCR extraction: {e}")
    
    return "", 0.0

def extract_game_text_multi_line(image_region):
    """
    Simplified text extraction for game UI elements, optimized for multi-line text
    """
    try:
        processed = preprocess_for_game_ui(image_region)
        text, confidence = extract_text_with_easyocr_multi_line(processed, "GameUI")
        
        if text:
            return text, confidence
                
    except Exception as e:
        pass
        
    # Fallback: Try original image if preprocessing fails
    try:
        text, confidence = extract_text_with_easyocr_multi_line(image_region, "Original")
        return text, confidence
    except Exception as e:
        pass
    
    return "", 0.0

def draw_preview_boxes(image, predictions, ocr_results=None):
    """
    Draw bounding boxes and OCR results on the image for preview
    """
    preview_image = image.copy()
    
    for i, pred in enumerate(predictions):
        # Calculate box coordinates
        x = int(pred['x'] - pred['width'] / 2)
        y = int(pred['y'] - pred['height'] / 2)
        w = int(pred['width'])
        h = int(pred['height'])
        
        # Draw bounding box
        color = (0, 255, 0)  # Green for detected boxes
        cv2.rectangle(preview_image, (x, y), (x + w, y + h), color, 2)
        
        # Add box ID
        cv2.putText(preview_image, f"#{i}", (x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Add OCR text if available
        if ocr_results and i < len(ocr_results):
            ocr_text = ocr_results[i].get('extracted_text', '')
            ocr_conf = ocr_results[i].get('ocr_confidence', 0.0)
            
            if ocr_text:
                # Choose text color based on confidence
                text_color = (0, 255, 0) if ocr_conf > 0.5 else (0, 165, 255)  # Green if good, orange if poor
                
                # Put text below the box
                text_y = y + h + 20
                cv2.putText(preview_image, f"{ocr_text} ({ocr_conf:.2f})", 
                           (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    
    return preview_image

def show_preview_window(image, window_name="OCR Preview", max_width=1200, max_height=800):
    """
    Display image in a resized window that fits the screen
    """
    h, w = image.shape[:2]
    
    # Calculate scaling factor to fit within max dimensions
    scale = min(max_width / w, max_height / h, 1.0)
    
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        resized = image
    
    cv2.imshow(window_name, resized)
    return resized

def run_game_ui_ocr(predictions_json_path, base_image_path="", output_json_path="game_ocr_results.json", 
                   show_preview=True, save_preview_images=False, preview_output_dir="preview_images"):
    """
    Specialized OCR for game UI elements using EasyOCR with optional preview
    """
    
    # Load predictions data
    with open(predictions_json_path, 'r') as f:
        predictions_data = json.load(f)
    
    results = []
    
    # Create preview directory if needed
    if save_preview_images:
        os.makedirs(preview_output_dir, exist_ok=True)
    
    print(f"Processing {len(predictions_data)} images...")
    print("Controls: SPACE = next image, ESC = quit, S = save current preview")
    print()
    
    for idx, image_data in enumerate(predictions_data):
        filename = image_data['filename']
        image_path = os.path.join(base_image_path, image_data['path'])
        predictions = image_data['result']['predictions']
        
        print(f"Processing {idx + 1}/{len(predictions_data)}: {filename}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"  Could not load image: {image_path}")
            continue
        
        image_result = {
            'image_name': filename,
            'event_boxes': []
        }
        
        # Process each event box
        box_results = []
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
            best_text, best_confidence = extract_game_text_multi_line(cropped)
            
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
            box_results.append(event_box_result)
            
            # Print results for this box
            if best_text:
                print(f"  Box #{i}: '{best_text}' (confidence: {best_confidence:.2f})")
        
        results.append(image_result)
        
        # Show preview if requested
        if show_preview:
            preview_img = draw_preview_boxes(image, predictions, box_results)
            show_preview_window(preview_img, f"OCR Preview - {filename}")
            
            while True:
                key = cv2.waitKey(0) & 0xFF
                
                if key == 27:  # ESC key - quit
                    cv2.destroyAllWindows()
                    print("Preview interrupted by user")
                    return results
                elif key == ord('s') or key == ord('S'):  # S key - save preview
                    if save_preview_images:
                        preview_path = os.path.join(preview_output_dir, f"preview_{filename}")
                        cv2.imwrite(preview_path, preview_img)
                        print(f"  Saved preview: {preview_path}")
                    else:
                        # Save even if save_preview_images is False
                        os.makedirs(preview_output_dir, exist_ok=True)
                        preview_path = os.path.join(preview_output_dir, f"preview_{filename}")
                        cv2.imwrite(preview_path, preview_img)
                        print(f"  Saved preview: {preview_path}")
                elif key == 32:  # SPACE key - next image
                    break
            
            cv2.destroyAllWindows()
        
        print()
    
    # Save results
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {output_json_path}")
    if save_preview_images:
        print(f"Preview images saved to: {preview_output_dir}/")
    
    return results

# Usage example:
if __name__ == "__main__":
    predictions_file = r"results\buy1_result.json"
    base_path = ""
    output_file = "game_ocr_results.json"
    
    print("Running Specialized Game UI OCR with EasyOCR and Preview...")
    print("This version shows bounding boxes and extracted text for verification")
    print("Note: First run may take longer as EasyOCR downloads models")
    print()
    
    results = run_game_ui_ocr(
        predictions_file, 
        base_path, 
        output_file,
        show_preview=True,          # Set to False to disable preview
        save_preview_images=True,   # Set to True to save preview images
        preview_output_dir="preview_images"
    )