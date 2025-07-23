import os
import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# List of Valorant characters
valorant_characters = [
    "Astra", "Brimstone", "Breach", "Cypher", "Jett", "Killjoy", "Omen", 
    "Phoenix", "Raze", "Reyna", "Sage", "Sova", "Viper", "Yoru"
]

# Function to extract text and match character
def match_character_from_image(image_path):
    # Open image
    image = Image.open(image_path)

    # Extract text using pytesseract
    extracted_text = pytesseract.image_to_string(image)

    # Convert extracted text to lowercase
    extracted_text_lower = extracted_text.lower()

    # Match extracted text with the character list
    for character in valorant_characters:
        if character.lower() in extracted_text_lower:
            return character
    return "No matching character found"

# Function to process all images in the given directory
def process_directory(directory_path):
    # Iterate through all files in the directory
    for file_name in os.listdir(directory_path):
        # Check if the file is an image (can check based on extension or MIME type)
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_path = os.path.join(directory_path, file_name)
            matched_character = match_character_from_image(image_path)
            print(f"Image: {file_name} - Matched Character: {matched_character}")

# Interactive prompt for user input for directory path
directory = input("Please enter the directory path to scan for images: ")

# Process the images in the given directory
process_directory(directory)
