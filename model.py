import os
import json
from PIL import Image, ImageDraw, ImageFont
from inference_sdk import InferenceHTTPClient
import pytesseract
import cv2
import numpy as np

# Use environment variables for API keys
API_KEY = os.environ.get("ROBOFLOW_API_KEY", "hY9qOmC03Dpg4JNVNeOp")

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY
)

valorant_characters = [
    "Astra", "Brimstone", "Breach", "Cypher", "Jett", "Killjoy", "Omen",
    "Phoenix", "Raze", "Reyna", "Sage", "Sova", "Viper", "Yoru",
    "KAY/O", "Skye", "Neon", "Fade", "Harbor", "Waylay", "Gekko", "Chamber"
]

def preprocess_for_ocr(image_crop):
    gray = cv2.cvtColor(np.array(image_crop), cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return thresh

def extract_text_from_region(cropped_image):
    preprocessed_image = preprocess_for_ocr(cropped_image)
    text = pytesseract.image_to_string(preprocessed_image, config='--psm 6').strip()
    return text

def process_image(image_path):
    image = Image.open(image_path).convert("RGB")
    result = client.infer(image_path, model_id="agent-selection-phase/2")

    detections_with_text = []

    for pred in result.get("predictions", []):
        x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
        left, top = x - w / 2, y - h / 2
        right, bottom = x + w / 2, y + h / 2

        cropped_image = image.crop((left, top, right, bottom))
        extracted_text = extract_text_from_region(cropped_image)

        detection = {
            "class": pred['class'],
            "confidence": pred['confidence'],
            "bounding_box": {"x": x, "y": y, "width": w, "height": h},
            "extracted_text": extracted_text
        }
        detections_with_text.append(detection)
        pred['extracted_text'] = extracted_text

    draw = ImageDraw.Draw(image)
    for detection in detections_with_text:
        x, y, w, h = detection['bounding_box']['x'], detection['bounding_box']['y'], detection['bounding_box']['width'], detection['bounding_box']['height']
        left, top = x - w / 2, y - h / 2
        right, bottom = x + w / 2, y + h / 2

        draw.rectangle([left, top, right, bottom], outline="red", width=3)
        class_label = f"{detection['class']} ({detection['confidence']:.2f})"
        draw.text((left, top - 25), class_label, fill="red")
        extracted_text = detection['extracted_text']
        if extracted_text:
            display_text = (extracted_text[:30] + "...") if len(extracted_text) > 30 else extracted_text
            draw.text((left, top - 10), f"Text: {display_text}", fill="blue")

    image.show()

if _name_ == "_main_":
    image_path = input("Please enter the path to the input image: ")
    if os.path.isfile(image_path):
        print(f"[INFO] Running inference on: {image_path}")
        process_image(image_path)
    else:
        print(f"[ERROR] File not found: {image_path}")