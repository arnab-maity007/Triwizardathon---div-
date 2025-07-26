import cv2
import pytesseract
import numpy as np
from PIL import Image
import os

# It is recommended to add Tesseract to your system's PATH
# If you must specify the path, consider using a configuration file
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# List of Valorant characters
valorant_characters = [
    "Astra", "Brimstone", "Breach", "Cypher", "Jett", "Killjoy", "Omen",
    "Phoenix", "Raze", "Reyna", "Sage", "Sova", "Viper", "Yoru",
    "KAY/O", "Skye", "Neon", "Fade", "Harbor", "Waylay", "Gekko", "Chamber"
]

# Variables to store the points selected by the user
points = []
frame = None # It's better to pass the frame as an argument

# Mouse callback function to get the points
def select_roi(event, x, y, flags, param):
    global points, frame
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) == 0:
            points = [(x, y)]
            print(f"First point selected: {points[0]}")
        elif len(points) == 1:
            points.append((x, y))
            print(f"Second point selected: {points[1]}")
            cv2.rectangle(frame, points[0], points[1], (0, 255, 0), 2)
            cv2.imshow("Select ROI", frame)

    if len(points) == 2:
        print(f"Region selected: {points[0]} to {points[1]}")

def match_character_from_image(image, selected_points):
    if len(selected_points) != 2:
        return "No region selected"

    x_start, y_start = selected_points[0]
    x_end, y_end = selected_points[1]
    cropped_image = image[y_start:y_end, x_start:x_end]

    gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    _, thresh_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)
    extracted_text = pytesseract.image_to_string(thresh_image).lower()
    print(f"Extracted Text: {extracted_text}")

    for character in valorant_characters:
        if character.lower() in extracted_text:
            return character
    return "No matching character found"

def process_video(input_video_path):
    global frame
    cap = cv2.VideoCapture(input_video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_skip = 5 # Process every 5th frame
    frame_count = 0

    cv2.namedWindow("Select ROI")
    cv2.setMouseCallback("Select ROI", select_roi)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Select ROI", frame)

        if len(points) == 2:
            if frame_count % frame_skip == 0:
                matched_character = match_character_from_image(frame, points)
                cv2.putText(frame, f"Detected: {matched_character}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Video Feed", frame)

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if _name_ == "_main_":
    input_video_path = input("Please enter the path to the input video: ")
    if os.path.exists(input_video_path):
        process_video(input_video_path)
    else:
        print("File not found.")