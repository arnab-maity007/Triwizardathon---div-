import cv2
import pytesseract
import numpy as np
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# List of Valorant characters
valorant_characters = [
    "Astra", "Brimstone", "Breach", "Cypher", "Jett", "Killjoy", "Omen", 
    "Phoenix", "Raze", "Reyna", "Sage", "Sova", "Viper", "Yoru", 
    "KAY/O", "Skye", "Neon", "Fade", "Harbor", "Waylay", "Gekko", "Chamber"
]

# Variables to store the points selected by the user
points = []

# Mouse callback function to get the points
def select_roi(event, x, y, flags, param):
    global points, frame
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) == 0:
            # Record the first point (top-left corner)
            points = [(x, y)]
            print(f"First point selected: {points[0]}")
        elif len(points) == 1:
            # Record the second point (bottom-right corner) and draw the rectangle
            points.append((x, y))
            print(f"Second point selected: {points[1]}")
            # Draw rectangle on the image
            cv2.rectangle(frame, points[0], points[1], (0, 255, 0), 2)
            cv2.imshow("Select ROI", frame)
    
    # If both points are selected, allow moving to the OCR processing
    if len(points) == 2:
        print(f"Region selected: {points[0]} to {points[1]}")
    
# Function to extract text and match character
def match_character_from_image(image):
    global points
    if len(points) != 2:
        return "No region selected"
    
    # Crop the region defined by the points
    x_start, y_start = points[0]
    x_end, y_end = points[1]
    cropped_image = image[y_start:y_end, x_start:x_end]

    # Convert the cropped region to grayscale (improves OCR performance)
    gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to enhance the text
    _, thresh_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)

    # Extract text using pytesseract
    extracted_text = pytesseract.image_to_string(thresh_image)

    # Print the extracted text to the terminal for debugging
    print(f"Extracted Text: {extracted_text}")

    # Convert extracted text to lowercase
    extracted_text_lower = extracted_text.lower()

    # Match extracted text with the character list
    for character in valorant_characters:
        if character.lower() in extracted_text_lower:
            return character
    return "No matching character found"

# Function to process the video and display results
def process_video(input_video_path):
    global frame
    # Open the video
    cap = cv2.VideoCapture(input_video_path)
    
    # Get video properties
    frame_rate = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set up mouse callback to allow region selection
    cv2.namedWindow("Select ROI")
    cv2.setMouseCallback("Select ROI", select_roi)

    # Process the video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Show the frame for the user to select a region
        cv2.imshow("Select ROI", frame)

        # Once both points are selected, start OCR processing
        if len(points) == 2:
            # Match character from the cropped region of the frame
            matched_character = match_character_from_image(frame)

            # Add text overlay to the frame (top-left corner)
            cv2.putText(frame, f"Detected: {matched_character}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the updated frame with detected text
            cv2.imshow("Video Feed", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close the window
    cap.release()
    cv2.destroyAllWindows()

# Input path for video
input_video_path = input("Please enter the path to the input video: ")

# Process the video and display the feed
process_video(input_video_path)
