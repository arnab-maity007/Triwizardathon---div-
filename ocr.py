import pytesseract
from PIL import Image

# Set this only if Tesseract isn't in PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def simple_ocr(image_path):
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        print("🔍 Extracted Text:")
        print(text.strip())
    except Exception as e:
        print(f"❌ Error: {e}")

# Example usage
if __name__ == "__main__":
    path = input("Enter image path: ")
    simple_ocr(path)
