import json
import cv2
import numpy as np
import easyocr
import os

reader = easyocr.Reader(['en'], gpu=True)

def preprocess_for_game_ui(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    scale_factor = 4
    height, width = gray.shape
    scaled = cv2.resize(gray, (width * scale_factor, height * scale_factor), interpolation=cv2.INTER_CUBIC)
    
    blurred = cv2.GaussianBlur(scaled, (3, 3), 0)
    
    mean_val = np.mean(blurred)
    if mean_val < 128:
        return 255 - blurred
    return blurred

def extract_text_with_easyocr(image):
    try:
        ocr_results = reader.readtext(image, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ ', width_ths=0.7, detail=1, paragraph=False)
        if ocr_results:
            best_result = max(ocr_results, key=lambda x: x[2])
            text, confidence = best_result[1].strip(), best_result[2]
            if text and len(text) > 1:
                return text, confidence
    except Exception:
        pass
    return "", 0.0

def run_game_ui_ocr(predictions_json_path, base_image_path="", output_json_path="game_ocr_results.json"):
    with open(predictions_json_path, 'r') as f:
        predictions_data = json.load(f)
    
    results = []
    
    for image_data in predictions_data:
        filename = image_data['filename']
        image_path = os.path.join(base_image_path, image_data['path'])
        predictions = image_data['result']['predictions']
        image = cv2.imread(image_path)
        if image is None:
            continue
        
        image_result = {'image_name': filename, 'event_boxes': []}
        
        for i, pred in enumerate(predictions):
            padding = 5
            x, y = int(pred['x'] - pred['width'] / 2) - padding, int(pred['y'] - pred['height'] / 2) - padding
            w, h = int(pred['width']) + (2 * padding), int(pred['height']) + (2 * padding)
            x, y = max(0, x), max(0, y)
            w, h = min(w, image.shape[1] - x), min(h, image.shape[0] - y)
            
            if w <= 0 or h <= 0: continue
            cropped = image[y:y+h, x:x+w]
            if cropped.size == 0: continue
            
            best_text, best_confidence = extract_text_with_easyocr(preprocess_for_game_ui(cropped))
            
            if best_text and (len(best_text) <= 1 or best_text.isspace() or all(c in '.,;:!?' for c in best_text)):
                best_text, best_confidence = "", 0.0
            
            event_box_result = {
                'box_id': i, 'coordinates': pred, 'confidence': pred['confidence'],
                'extracted_text': best_text, 'ocr_confidence': best_confidence
            }
            image_result['event_boxes'].append(event_box_result)
        results.append(image_result)
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return results

if _name_ == "_main_":
    predictions_file = "results/buy1_result.json" # Use a relative path
    if os.path.exists(predictions_file):
        run_game_ui_ocr(predictions_file, "", "game_ocr_results.json")
    else:
        print(f"File not found: {predictions_file}")