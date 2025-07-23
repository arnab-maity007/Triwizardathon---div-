import pytesseract
from PIL import Image
import json
import cv2
import numpy as np
import pytesseract
import os
from PIL import Image, ImageEnhance, ImageFilter

# Set the path for Tesseract (if it's not in your PATH)
# For Windows, make sure to point to the correct path where Tesseract is installed
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Adjust the path for your system

def ocr_image(image_path: str):
    """Run OCR on an image file and return the extracted text."""
    try:
        # Open the image using Pillow
        image = Image.open(image_path)

        # Run OCR on the image
        text = pytesseract.image_to_string(image)

        # Return the extracted text
        return text

    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None


def run_ocr_on_event_boxes_enhanced(predictions_json_path, base_image_path="", output_json_path="ocr_results_enhanced.json", show_preview=True, auto_preview=False):
    """
    Enhanced OCR function with multiple preprocessing methods and detailed preview
    """
    
    # Load predictions data
    with open(predictions_json_path, 'r') as f:
        predictions_data = json.load(f)
    
    results = []
    skip_previews = False
    
    for image_data in predictions_data:
        filename = image_data['filename']
        image_path = os.path.join(base_image_path, image_data['path'])
        predictions = image_data['result']['predictions']
        
        print(f"Processing {filename}...")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load {image_path}")
            continue
        
        image_result = {
            'image_name': filename,
            'event_boxes': []
        }
        
        # Process each event box
        for i, pred in enumerate(predictions):
            print(f"  Processing box {i}...")
            
            # Calculate crop coordinates with padding
            padding = 10
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
            
            # Extract text using multiple methods
            extracted_texts, processed_images = ocr_image(cropped)
            
            # Choose the best result
            best_text = ""
            if extracted_texts:
                # Filter out single charac_ters and noise
                filtered_texts = []
                for text_with_method in extracted_texts:
                    # Extract just the text part (after the colon)
                    if ': ' in text_with_method:
                        text = text_with_method.split(': ', 1)[1]
                    else:
                        text = text_with_method
                    
                    # Filter criteria
                    if (len(text) > 1 and 
                        not text in ['_', '-', '~', '=', '<', '>', '|', '/', '\\', '..', '...'] and
                        not text.isspace()):
                        filtered_texts.append(text)
                
                if filtered_texts:
                    # Choose the longest meaningful text
                    best_text = max(filtered_texts, key=len)
                    print(f"    Best result: '{best_text}'")
                else:
                    print(f"    No meaningful text found")
            else:
                print(f"    No text extracted")
            
            # Clean up the text
            best_text = ' '.join(best_text.split())
            
            # Show enhanced preview if enabled
            if show_preview and not skip_previews:
                coordinates_with_confidence = {
                    'x': pred['x'],
                    'y': pred['y'],
                    'width': pred['width'],
                    'height': pred['height'],
                    'confidence': pred['confidence']
                }
                from ocr import show_ocr_preview_enhanced  # Import from your previous function
                if auto_preview:
                    show_ocr_preview_enhanced(image, cropped, processed_images, coordinates_with_confidence, extracted_texts, i, filename)
                    cv2.waitKey(3000)  # Wait 3 seconds
                    cv2.destroyAllWindows()
                else:
                    skip_previews = show_ocr_preview_enhanced(image, cropped, processed_images, coordinates_with_confidence, extracted_texts, i, filename)
            
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
                'all_extraction_attempts': extracted_texts  # Store all attempts for debugging
            }
            
            image_result['event_boxes'].append(event_box_result)
        
        results.append(image_result)
    
    # Close any remaining windows
    cv2.destroyAllWindows()
    
    # Save results
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Enhanced OCR results saved to {output_json_path}")
    
    # Print summary
    total_images = len(results)
    total_boxes = sum(len(img['event_boxes']) for img in results)
    boxes_with_text = sum(len([box for box in img['event_boxes'] if box['extracted_text']]) for img in results)
    boxes_with_good_text = sum(
        len([box for box in img['event_boxes'] if len(box['extracted_text']) > 2]) 
        for img in results
    )
    
    print(f"\nSummary:")
    print(f"Processed {total_images} images")
    print(f"Found {total_boxes} event boxes")
    print(f"Extracted text from {boxes_with_text} boxes")
    print(f"Boxes with meaningful text (>2 chars): {boxes_with_good_text}")
    
    return results

# Usage example:
if __name__ == "__main__":
    predictions_file = r"results\folder_results.json"
    base_path = ""
    output_file = "ocr_results_enhanced.json"
    
    print("Running Enhanced OCR with multiple preprocessing methods...")
    print("Controls:")
    print("- Press any key to continue to next box")
    print("- Press 'q' to skip remaining previews")
    print("- Press 's' to save current preview image")
    print()
    
    results = run_ocr_on_event_boxes_enhanced(
        predictions_file, 
        base_path, 
        output_file, 
        show_preview=True,
        auto_preview=False
    )