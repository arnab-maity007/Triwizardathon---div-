import json
import re
from difflib import SequenceMatcher
from typing import Dict, List, Tuple, Optional

class ValorantBuyPhaseClassifier:
    def __init__(self):
        # Define weapon categories and their items
        self.weapon_data = {
            'sidearms': {
                'classic': ['classic', 'clasic', 'classik'],
                'shorty': ['shorty', 'shortie', 'shorty'],
                'frenzy': ['frenzy', 'frency', 'frenzi'],
                'ghost': ['ghost', 'ghosT', 'gost'],
                'sheriff': ['sheriff', 'sherif', 'sherrif']
            },
            'smgs': {
                'stinger': ['stinger', 'stingr', 'stingeR'],
                'spectre': ['spectre', 'specter', 'spectr']
            },
            'rifles': {
                'bulldog': ['bulldog', 'buldog', 'bulldg'],
                'guardian': ['guardian', 'guardien', 'guardin'],
                'phantom': ['phantom', 'fantom', 'phaniont'],
                'vandal': ['vandal', 'vandel', 'vandl']
            },
            'sniper_rifles': {
                'marshal': ['marshal', 'marshel', 'marshall'],
                'operator': ['operator', 'operatr', 'opertor']
            },
            'shotguns': {
                'bucky': ['bucky', 'buckie', 'bucky'],
                'judge': ['judge', 'judoe', 'juge']
            },
            'machine_guns': {
                'ares': ['ares', 'are', 'ares'],
                'odin': ['odin', 'odn', 'odin']
            },
            'armor': {
                'light_shields': ['light shields', 'light', 'shields'],
                'heavy_shields': ['heavy shields', 'heavy', 'shields']
            },
            'abilities': {
                'nanoswarm': ['nanoswarm', 'nano', 'swarm'],
                'alarmbot': ['alarmbot', 'alarm', 'bot'],
                'turret': ['turret', 'turet', 'turrent'],
                'incendiary': ['incendiary', 'incendiARY', 'fire'],
                'stim_beacon': ['stim beacon', 'stim', 'beacon'],
                'sky_smoke': ['sky smoke', 'sky', 'smoke']
            }
        }
        
        # Common price patterns
        self.price_patterns = [
            r'(\d{1,2}[,.]?\d{3})',  # 1,600 or 1.600 or 1600
            r'([ødpb]\d{2,4})',      # ø950, d100, p100, b400
            r'(\d{2,4})'             # 950, 100, etc.
        ]
        
        # Updated status indicators to match new categories
        self.status_indicators = {
            'requesting_weapon': ['requesting', 'request', 'req', 'aequesting'],
            'teammate_requesting_weapon': ['teammate requesting', 'team request', 'ally request'],
            'weapon_owned': ['owned', 'ownd', 'own', 'bought', 'buy', 'purchased', 'equipped', 'equip', 'equiped', 'omned'],  # Added 'omned' for OCR errors
            'shield_owned': ['shield owned', 'armor owned', 'shields equipped'],
            'ability_owned': ['ability owned', 'skill owned', 'ability equipped', 'full', 'ful'],  # Added 'full' and 'ful' for abilities
            'side_arm_owned': ['sidearm owned', 'pistol owned', 'secondary owned'],
            'weapon_hover': ['hovering', 'hover', 'selected', 'select']  # removed 'full', 'ful' from here
        }

        # Category mappings for determining specific owned types
        self.category_to_owned_status = {
            'sidearms': 'side_arm_owned',
            'armor': 'shield_owned',
            'abilities': 'ability_owned',
            'smgs': 'weapon_owned',
            'rifles': 'weapon_owned',
            'sniper_rifles': 'weapon_owned',
            'shotguns': 'weapon_owned',
            'machine_guns': 'weapon_owned'
        }

    def similarity(self, a: str, b: str) -> float:
        """Calculate similarity between two strings"""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def extract_price(self, text: str) -> Optional[str]:
        """Extract price from OCR text"""
        for pattern in self.price_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def detect_status(self, text: str, weapon_category: str = None) -> str:
        """Detect the status of an item based on text indicators and weapon category"""
        text_lower = text.lower()
        
        # Special case: "full" for abilities means ability_owned
        if ('full' in text_lower or 'ful' in text_lower) and weapon_category == 'abilities':
            return 'ability_owned'
        
        # Check for each status indicator first
        for status, patterns in self.status_indicators.items():
            for pattern in patterns:
                if pattern in text_lower:
                    # For owned items, map to appropriate category-specific status
                    if status == 'weapon_owned' and weapon_category:
                        return self.category_to_owned_status.get(weapon_category, 'weapon_owned')
                    return status
        
        # Additional owned indicators check with better OCR error handling
        owned_indicators = ['owned', 'bought', 'equipped', 'purchased', 'omned', 'ownd']
        for indicator in owned_indicators:
            if indicator in text_lower:
                # Map to appropriate owned category based on weapon type
                if weapon_category:
                    return self.category_to_owned_status.get(weapon_category, 'weapon_owned')
                return 'weapon_owned'
        
        # Check for request indicators
        request_indicators = ['requesting', 'request', 'req', 'aequesting']
        for indicator in request_indicators:
            if indicator in text_lower:
                return 'requesting_weapon'
        
        # Special case: items with "50/50" pattern are likely owned armor
        if re.search(r'\d+/\d+', text_lower) and weapon_category == 'armor':
            return 'shield_owned'
        
        # If it has a price and no other indicator, likely hovering/available
        if self.extract_price(text):
            return 'weapon_hover'
        
        # Default status is weapon_hover (replaces unknown)
        return 'weapon_hover'

    def is_request(self, text: str) -> bool:
        """Check if text indicates a weapon request"""
        status = self.detect_status(text)
        return status in ['requesting_weapon', 'teammate_requesting_weapon']

    def find_best_weapon_match(self, ocr_text: str) -> Tuple[Optional[str], Optional[str], float]:
        """Find the best matching weapon from all categories"""
        best_match = None
        best_category = None
        best_score = 0.0
        
        # Clean the OCR text
        clean_text = re.sub(r'[^\w\s]', '', ocr_text.lower())
        
        # Special handling for armor types - check for heavy/light keywords first
        if 'armor' in self.weapon_data:
            if 'heavy' in clean_text and 'shields' in clean_text:
                return 'armor', 'heavy_shields', 0.9
            elif 'light' in clean_text and 'shields' in clean_text:
                return 'armor', 'light_shields', 0.9
        
        for category, weapons in self.weapon_data.items():
            for weapon_name, variants in weapons.items():
                for variant in variants:
                    # Check full text similarity
                    score = self.similarity(clean_text, variant)
                    
                    # Also check if variant is contained in the text
                    if variant.lower() in clean_text:
                        score = max(score, 0.8)
                    
                    # Check word-by-word matching
                    words = clean_text.split()
                    for word in words:
                        word_score = self.similarity(word, variant)
                        if word_score > 0.7:
                            score = max(score, word_score)
                    
                    if score > best_score:
                        best_score = score
                        best_match = weapon_name
                        best_category = category
        
        return best_category, best_match, best_score

    def classify_detection(self, detection: Dict) -> Dict:
        """Classify a single detection"""
        ocr_text = detection.get('ocr_result', {}).get('text', '')
        confidence = detection.get('ocr_result', {}).get('confidence', 0.0)
        bbox = detection.get('bbox', [])
        class_name = detection.get('class_name', '')
        
        # Extract price
        price = self.extract_price(ocr_text)
        
        # Find best weapon match first
        category, weapon, match_score = self.find_best_weapon_match(ocr_text)
        
        # Detect status with weapon category context
        status = self.detect_status(ocr_text, category)
        is_request = self.is_request(ocr_text)
        
        # Determine classification confidence
        classification_confidence = match_score * confidence
        
        result = {
            'original_text': ocr_text,
            'ocr_confidence': confidence,
            'bbox': bbox,
            'price': price,
            'status': status,
            'is_request': is_request,
            'category': category,
            'weapon': weapon,
            'match_score': match_score,
            'classification_confidence': classification_confidence,
            'original_class': class_name
        }
        
        return result

    def generate_structured_output(self, results: Dict) -> Dict:
        """Generate a clean structured JSON output with item names and statuses"""
        structured_output = {
            'metadata': {
                'image_name': results['metadata']['image_name'],
                'timestamp': results['metadata']['timestamp'],
                'total_event_boxes_analyzed': results['metadata']['event_boxes_with_text']
            },
            'items': []
        }
        
        # Process all classifications
        for classification in results['classifications']:
            if classification['weapon'] and classification['match_score'] > 0.4:
                item = {
                    'item_name': classification['weapon'],
                    'category': classification['category'],
                    'status': classification['status'],
                    'price': classification['price'],
                    'confidence': round(classification['classification_confidence'], 3),
                    'original_text': classification['original_text']
                }
                structured_output['items'].append(item)
        
        # Sort items by confidence (highest first)
        structured_output['items'].sort(key=lambda x: x['confidence'], reverse=True)
        
        return structured_output

    def process_json_data(self, json_data: Dict) -> Dict:
        """Process the entire JSON data structure"""
        if isinstance(json_data, str):
            json_data = json.loads(json_data)
        
        # Count event-box detections
        event_box_detections = 0
        total_event_boxes = 0
        
        for detection in json_data.get('detections', []):
            if detection.get('class_name') == 'event-box':
                total_event_boxes += 1
                if detection.get('has_text', False):
                    event_box_detections += 1
        
        results = {
            'metadata': {
                'image_name': json_data.get('image_name', ''),
                'timestamp': json_data.get('timestamp', ''),
                'total_detections': json_data.get('total_detections', 0),
                'detections_with_text': json_data.get('detections_with_text', 0),
                'total_event_boxes': total_event_boxes,
                'event_boxes_with_text': event_box_detections
            },
            'classifications': [],
            'summary': {
                # New status categories
                'requesting_weapon': [],
                'teammate_requesting_weapon': [],
                'weapon_owned': [],
                'shield_owned': [],
                'ability_owned': [],
                'side_arm_owned': [],
                'weapon_hover': [],
                # Keep weapon categories for reference
                'sidearms': [],
                'smgs': [],
                'rifles': [],
                'sniper_rifles': [],
                'shotguns': [],
                'machine_guns': [],
                'armor': [],
                'abilities': [],
                'unclassified': []
            }
        }
        
        # Process each detection - ONLY event-box classifications
        for detection in json_data.get('detections', []):
            # Only process if it's an event-box AND has text
            if (detection.get('class_name') == 'event-box' and 
                detection.get('has_text', False)):
                
                classified = self.classify_detection(detection)
                results['classifications'].append(classified)
                
                # Add to appropriate summary category based on status
                status = classified['status']
                if status in results['summary']:
                    results['summary'][status].append(classified)
                elif classified['category'] and classified['match_score'] > 0.5:
                    category_key = classified['category']
                    if category_key in results['summary']:
                        results['summary'][category_key].append(classified)
                else:
                    results['summary']['unclassified'].append(classified)
        
        return results

    def print_summary(self, results: Dict):
        """Print a formatted summary of the classification results"""
        print(f"=== Valorant Buy Phase Analysis ===")
        print(f"Image: {results['metadata']['image_name']}")
        print(f"Total detections: {results['metadata']['total_detections']}")
        print(f"Total event boxes: {results['metadata']['total_event_boxes']}")
        print(f"Event boxes with text: {results['metadata']['event_boxes_with_text']}")
        print(f"Note: Only analyzing event-box classifications")
        print()
        
        # Print structured output
        structured = self.generate_structured_output(results)
        print("=== STRUCTURED OUTPUT ===")
        print(json.dumps(structured, indent=2))
        print()
        
        # Print by new status categories
        status_categories = [
            'requesting_weapon', 
            'teammate_requesting_weapon', 
            'weapon_owned', 
            'shield_owned', 
            'ability_owned', 
            'side_arm_owned', 
            'weapon_hover'
        ]
        total_classified = 0
        
        for status in status_categories:
            items = results['summary'].get(status, [])
            if items:
                total_classified += len(items)
                print(f"{status.upper().replace('_', ' ')}:")
                for item in items:
                    price_str = f" (₹{item['price']})" if item['price'] else ""
                    confidence_str = f" [confidence: {item['classification_confidence']:.2f}]"
                    weapon_str = f" - {item['weapon']}" if item['weapon'] else ""
                    category_str = f" ({item['category']})" if item['category'] else ""
                    print(f"  • {item['original_text']}{weapon_str}{category_str}{price_str}{confidence_str}")
                print()
        
        # Print weapon categories for additional reference
        weapon_categories = ['sidearms', 'smgs', 'rifles', 'sniper_rifles', 'shotguns', 'machine_guns', 'armor', 'abilities']
        category_total = 0
        for category in weapon_categories:
            items = results['summary'].get(category, [])
            if items:
                category_total += len(items)
        
        if category_total > 0:
            print("=== WEAPON CATEGORIES (Reference) ===")
            for category in weapon_categories:
                items = results['summary'].get(category, [])
                if items:
                    print(f"{category.upper().replace('_', ' ')}:")
                    for item in items:
                        price_str = f" (₹{item['price']})" if item['price'] else ""
                        confidence_str = f" [confidence: {item['classification_confidence']:.2f}]"
                        weapon_str = f" - {item['weapon']}" if item['weapon'] else ""
                        status_str = f" [{item['status']}]"
                        print(f"  • {item['original_text']}{weapon_str}{price_str}{status_str}{confidence_str}")
                    print()
        
        # Print unclassified items
        unclassified = results['summary'].get('unclassified', [])
        if unclassified:
            print("UNCLASSIFIED:")
            for item in unclassified:
                status_str = f" [{item['status']}]" if item['status'] != 'weapon_hover' else ""
                print(f"  • {item['original_text']}{status_str}")
            print()
        
        if total_classified == 0:
            print("No event-box items were successfully classified.")
        else:
            print(f"Successfully classified {total_classified} event-box items.")

def main():
    classifier = ValorantBuyPhaseClassifier()
    
    # Get file path from user
    print("=== Valorant Buy Phase OCR Classifier ===")
    print("Please provide the path to your JSON file containing OCR results.")
    print("Example: D:\\dual cast\\dataset\\buy\\ocr_results.json")
    print()
    
    while True:
        file_path = input("Enter JSON file path (or 'quit' to exit): ").strip()
        
        if file_path.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not file_path:
            print("Please enter a valid file path.")
            continue
        
        # Remove quotes if user added them
        file_path = file_path.strip('"').strip("'")
        
        try:
            # Check if file exists
            import os
            if not os.path.exists(file_path):
                print(f"Error: File '{file_path}' not found. Please check the path and try again.")
                continue
            
            # Load and process the JSON data
            print(f"\nLoading file: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print("Processing OCR data...")
            results = classifier.process_json_data(data)
            
            # Print summary
            print("\n" + "="*60)
            classifier.print_summary(results)
            
            # Ask if user wants to save results
            save_choice = input("\nSave detailed results to file? (y/n): ").strip().lower()
            if save_choice in ['y', 'yes']:
                # Generate output filename
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                
                # Save detailed results
                detailed_output_path = f"{base_name}_detailed.json"
                with open(detailed_output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                
                # Save structured results
                structured_output = classifier.generate_structured_output(results)
                structured_output_path = f"{base_name}_structured.json"
                with open(structured_output_path, 'w', encoding='utf-8') as f:
                    json.dump(structured_output, f, indent=2, ensure_ascii=False)
                
                print(f"Detailed results saved to: {detailed_output_path}")
                print(f"Structured results saved to: {structured_output_path}")
            
            # Ask if user wants to process another file
            another = input("\nProcess another file? (y/n): ").strip().lower()
            if another not in ['y', 'yes']:
                print("Goodbye!")
                break
                
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON format in file. {e}")
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found. Please check the path.")
        except PermissionError:
            print(f"Error: Permission denied accessing '{file_path}'.")
        except Exception as e:
            print(f"Error processing file: {e}")
        
        print()  # Add spacing for next iteration

def process_file_directly(file_path: str):
    """
    Helper function to process a file directly without user interaction.
    Useful for batch processing or integration with other scripts.
    """
    classifier = ValorantBuyPhaseClassifier()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = classifier.process_json_data(data)
        classifier.print_summary(results)
        
        return results
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

if __name__ == "__main__":
    import sys
    
    # Check if file path was provided as command line argument
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        print(f"Processing file: {file_path}")
        process_file_directly(file_path)
    else:
        # Interactive mode
        main()