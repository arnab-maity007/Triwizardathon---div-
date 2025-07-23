import os
import json
from PIL import Image, ImageDraw, ImageFont
from inference_sdk import InferenceHTTPClient
import pytesseract
import cv2

# Initialize the Roboflow client
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="hY9qOmC03Dpg4JNVNeOp"
)

# List of Valorant characters (for matching OCR results)
valorant_characters = [
    "Astra", "Brimstone", "Breach", "Cypher", "Jett", "Killjoy", "Omen", 
    "Phoenix", "Raze", "Reyna", "Sage", "Sova", "Viper", "Yoru", 
    "KAY/O", "Skye", "Neon", "Fade", "Harbor", "Waylay", "Gekko", "Chamber"
]

# Function to preprocess the cropped image for OCR
def preprocess_for_ocr(image_crop):
    gray = cv2.cvtColor(np.array(image_crop), cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return thresh

# Function to extract text from a specific region of the image
def extract_text_from_region(cropped_image):
    preprocessed_image = preprocess_for_ocr(cropped_image)
    text = pytesseract.image_to_string(preprocessed_image, config='--psm 6').strip()
    return text

# Function to process the image: Perform object detection and OCR on detected regions
def process_image(image_path):
    image = Image.open(image_path).convert("RGB")
    result = client.infer(image_path, model_id="agent-selection-phase/2")  # Update model_id if necessary

    detections_with_text = []

    # Process each detection and extract text from the detected regions
    for pred in result["predictions"]:
        x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
        left, top = x - w / 2, y - h / 2
        right, bottom = x + w / 2, y + h / 2
        
        # Crop image based on the bounding box
        cropped_image = image.crop((left, top, right, bottom))

        # Extract text from the cropped region
        extracted_text = extract_text_from_region(cropped_image)

        # Add OCR results to the prediction
        detection = {
            "class": pred['class'],
            "confidence": pred['confidence'],
            "bounding_box": {"x": x, "y": y, "width": w, "height": h},
            "extracted_text": extracted_text
        }

        detections_with_text.append(detection)
        pred['extracted_text'] = extracted_text  # Add extracted text to original prediction

    # Draw bounding boxes and OCR results on the image
    draw = ImageDraw.Draw(image)
    for detection in detections_with_text:
        x, y, w, h = detection['bounding_box']['x'], detection['bounding_box']['y'], detection['bounding_box']['width'], detection['bounding_box']['height']
        left, top = x - w / 2, y - h / 2
        right, bottom = x + w / 2, y + h / 2

        # Draw bounding box
        draw.rectangle([left, top, right, bottom], outline="red", width=3)

        # Draw class label and confidence
        class_label = f"{detection['class']} ({detection['confidence']:.2f})"
        draw.text((left, top - 25), class_label, fill="red")

        # Draw extracted text
        extracted_text = detection['extracted_text']
        if extracted_text:
            display_text = extracted_text[:30] + "..." if len(extracted_text) > 30 else extracted_text
            draw.text((left, top - 10), f"Text: {display_text}", fill="blue")

    # Show the image with bounding boxes and OCR text
    image.show()

# Main function to run the process
def main():
    image_path = input("Please enter the path to the input image: ")
    if not os.path.isfile(image_path):
        print(f"[ERROR] File not found: {image_path}")
        return

    print(f"[INFO] Running inference on: {image_path}")
    process_image(image_path)

if __name__ == "__main__":
    main()
