# Valorant AI Commentary System

This repository contains a sophisticated AI-powered system designed to analyze gameplay from the popular tactical shooter, *Valorant*. It uses computer vision and Optical Character Recognition (OCR) to understand events happening during the agent selection and buy phases, and then generates dynamic, human-like commentary in real-time.

## Key Features

  * **Real-time Analysis**: The system can process video feeds or directories of images to detect and interpret in-game events as they happen.
  * **Agent Selection Tracking**: Identifies which agent a player is hovering over and when they "lock in" their choice.
  * **Buy Phase Analysis**: Detects all interactive elements in the buy menu, including weapons, shields, and abilities.
  * **Advanced OCR**: Utilizes multiple OCR engines (`easyocr` and `pytesseract`) with specialized image preprocessing to accurately extract text from the game's UI.
  * **Intelligent Event Classification**: Classifies extracted information into structured data, identifying player actions like purchasing, owning, or requesting items.
  * **Dynamic Commentary Generation**:
      * **Dual Caster System**: Generates commentary from two distinct personas: a "Hype" caster for excitement and an "Analyst" for tactical insights.
      * **Context-Aware**: The commentary is tailored to the specific events, such as a player requesting a weapon or purchasing full armor.
  * **Riot API Integration**: Includes scripts to fetch player data, like summoner IDs, directly from the Riot Games API.

## How It Works

The system operates in a multi-stage pipeline:

1.  **Detection**: A trained object detection model (either a cloud-based model from Roboflow or a local YOLO model) scans the game screen (from an image or video frame) to identify key UI elements, referred to as "event-boxes".

2.  **OCR and Text Extraction**: Once an event-box is detected, the system crops that specific region of the image. Advanced image preprocessing techniques are applied, and then OCR is used to extract any text within the box, such as weapon names, prices, or ability statuses.

3.  **Classification and Structuring**:

      * **Buy Phase**: The extracted text from the buy menu is processed by a classifier that uses fuzzy string matching and regular expressions to identify the specific item (e.g., "Vandal", "Heavy Shields") and its status (e.g., `weapon_owned`, `requesting_weapon`). This structured data is then saved to a JSON file.
      * **Agent Selection**: The system identifies the agent's name and looks for the "LOCK IN" button to determine if a player is just considering an agent or has confirmed their choice. This state is also saved.

4.  **Commentary Generation**: The JSON files containing the structured data act as triggers for the commentary engine. The system reads the latest events and:

      * Prioritizes the most important actions.
      * Selects a caster role (alternating between Hype and Analyst).
      * Chooses a suitable commentary template for the event.
      * Fills in the template with the relevant details (player name, weapon, etc.) to generate a final line of commentary.

## Repository Structure

```
.
├── agents/                  # Sample images from the Agent Selection phase
├── buyphase/                # Sample images from the Buy Phase
├── results/                 # Output directory for annotated images and JSON results
├── Output Processed Json/   # Contains the core logic for commentary and JSON processing
│   ├── agent.json           # Example processed data for agent selection
│   ├── buy.json             # Example processed data for the buy phase
│   ├── agentprocessor.py    # Generates commentary for agent selection
│   └── buycommentary.py     # Generates commentary for the buy phase
├── buy_phase.py             # Main script for processing the buy phase using Roboflow API
├── buyphase.py              # Main script for processing the buy phase using a local YOLO model
├── model.py                 # Main script for processing agent selection
├── shower.py                # Enhanced OCR script with a preview window for verification
├── requirements.txt         # Project dependencies
└── ...                      # Other scripts and data files
```

## Setup and Usage

### Prerequisites

  * Python 3.x
  * Tesseract OCR Engine
  * API keys for Roboflow and Riot Games (optional, for full functionality)

### Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install Python dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Tesseract OCR:**

      * Make sure Tesseract is installed and accessible from your system's PATH.
      * Alternatively, you can specify the path to the Tesseract executable directly in the scripts (e.g., in `imp.txt` or at the top of `ocr.py`).

4.  **Set API Keys:**

      * For scripts using the Roboflow or Riot Games API, set your API keys as environment variables or replace the placeholder keys in the code.

### Running the System

You can run the different modules of the system individually. The general workflow is to first process images/videos to generate JSON data, and then use that data to generate commentary.

1.  **Process Buy Phase Images:**

      * To process a folder of images and generate detection results:
        ```bash
        python buy_phase.py
        ```
        (Follow the interactive prompts to select the folder and options)

2.  **Run OCR on Detections:**

      * After generating detection JSON, run the OCR script to extract text. The `shower.py` script is recommended as it provides a visual preview to verify accuracy.

3.  **Generate Commentary:**

      * Once you have processed JSON files (e.g., `buy.json`), you can run the commentary scripts:
        ```bash
        # For buy phase commentary
        python "Output Processed Json/buycommentary.py"

        # For agent selection commentary
        python "Output Processed Json/agentprocessor.py"
        ```

## Key Dependencies

  * `ultralytics`: For running the YOLO model.
  * `opencv-python`: For image and video processing.
  * `numpy`: For numerical operations.
  * `torch`: The backend for the YOLO model.
  * `pytesseract` & `easyocr`: For Optical Character Recognition.
  * `requests`: For making API calls.
  * `fuzzywuzzy`: For string matching in the classification scripts.
