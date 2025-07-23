import os
import json
import cv2
from pathlib import Path
from typing import List, Dict, Any
from inference_sdk import InferenceHTTPClient

class RoboflowProcessor:
    def __init__(self, api_key: str = "hY9qOmC03Dpg4JNVNeOp", model_id: str = "buy-phase-valorant/3", confidence_threshold: float = 0.9, target_class: str = "event-box"):
        """Initialize the Roboflow client"""
        self.client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=api_key
        )
        self.model_id = model_id
        self.confidence_threshold = confidence_threshold
        self.target_class = target_class
        
    def process_single_image(self, image_path: str, save_results: bool = False, output_dir: str = "results") -> Dict[Any, Any]:
        """Process a single image and optionally save results"""
        try:
            print(f"Processing image: {image_path}")
            result = self.client.infer(image_path, model_id=self.model_id)
            
            # Filter predictions by confidence threshold and target class
            if 'predictions' in result:
                original_count = len(result['predictions'])
                filtered_predictions = [
                    pred for pred in result['predictions'] 
                    if (pred.get('confidence', 0) >= self.confidence_threshold and 
                        pred.get('class', '').lower() == self.target_class.lower())
                ]
                result['predictions'] = filtered_predictions
                result['filtered_count'] = len(filtered_predictions)
                result['original_count'] = original_count
                print(f"Found {len(filtered_predictions)} '{self.target_class}' predictions above confidence threshold {self.confidence_threshold}")
                print(f"(Filtered from {original_count} total predictions)")
            
            if save_results:
                self._save_image_result(image_path, result, output_dir)
                
            return result
            
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return {}
    
    def process_image_folder(self, folder_path: str, save_results: bool = False, output_dir: str = "results") -> List[Dict[Any, Any]]:
        """Process all images in a folder"""
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(supported_formats)]
        
        if not image_files:
            print(f"No supported image files found in {folder_path}")
            return []
        
        results = []
        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            result = self.process_single_image(image_path, save_results, output_dir)
            if result:
                results.append({
                    'filename': image_file,
                    'path': image_path,
                    'result': result
                })
        
        if save_results and results:
            self._save_batch_results(results, output_dir, "folder_results.json")
            
        return results
    
    def process_video(self, video_path: str, save_results: bool = False, output_dir: str = "results", 
                     frame_skip: int = 30, create_output_video: bool = True) -> List[Dict[Any, Any]]:
        """Process video frames and optionally save results and create output video"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return []
        
        results = []
        frame_count = 0
        processed_frames = 0
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Processing video: {video_path}")
        print(f"Total frames: {total_frames}, FPS: {fps}, Resolution: {width}x{height}")
        print(f"Processing every {frame_skip} frames...")
        
        # Initialize video writer for output video
        out = None
        if create_output_video and save_results:
            os.makedirs(output_dir, exist_ok=True)
            video_name = Path(video_path).stem
            output_video_path = os.path.join(output_dir, f"{video_name}_with_detections.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            print(f"Output video will be saved to: {output_video_path}")
        
        # Store detection results for all frames (for video output)
        all_frame_detections = {}
        
        # Reset video capture to beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_count = 0
        
        # First pass: run inference on selected frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process every nth frame for inference
            if frame_count % frame_skip == 0:
                # Save frame temporarily
                temp_frame_path = "temp_frame.jpg"
                cv2.imwrite(temp_frame_path, frame)
                
                try:
                    result = self.client.infer(temp_frame_path, model_id=self.model_id)
                    
                    # Filter predictions by confidence threshold and target class
                    if 'predictions' in result:
                        original_count = len(result['predictions'])
                        filtered_predictions = [
                            pred for pred in result['predictions'] 
                            if (pred.get('confidence', 0) >= self.confidence_threshold and 
                                pred.get('class', '').lower() == self.target_class.lower())
                        ]
                        result['predictions'] = filtered_predictions
                    
                    # Store detections for this frame
                    all_frame_detections[frame_count] = result.get('predictions', [])
                    
                    frame_result = {
                        'frame_number': frame_count,
                        'timestamp': frame_count / fps,
                        'result': result
                    }
                    results.append(frame_result)
                    processed_frames += 1
                    
                    pred_count = len(result.get('predictions', []))
                    if pred_count > 0:
                        print(f"Processed frame {frame_count}/{total_frames} (timestamp: {frame_count/fps:.2f}s) - Found {pred_count} '{self.target_class}' detections")
                    else:
                        print(f"Processed frame {frame_count}/{total_frames} (timestamp: {frame_count/fps:.2f}s) - No '{self.target_class}' detected")
                    
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {str(e)}")
                    all_frame_detections[frame_count] = []
                
                # Clean up temporary file
                if os.path.exists(temp_frame_path):
                    os.remove(temp_frame_path)
            
            frame_count += 1
        
        # Second pass: create output video with detections
        if create_output_video and save_results and out is not None:
            print("Creating output video with detections...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Find the closest processed frame for detections
                closest_processed_frame = self._find_closest_processed_frame(frame_count, all_frame_detections, frame_skip)
                
                if closest_processed_frame is not None and closest_processed_frame in all_frame_detections:
                    predictions = all_frame_detections[closest_processed_frame]
                    frame = self._draw_predictions_on_frame(frame, predictions)
                
                out.write(frame)
                frame_count += 1
                
                # Progress indicator
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Video creation progress: {progress:.1f}%")
            
            out.release()
            print(f"✅ Output video created successfully: {output_video_path}")
        
        cap.release()
        
        if save_results and results:
            self._save_video_results(video_path, results, output_dir)
            
        print(f"Video processing complete. Processed {processed_frames} frames.")
        return results
    
    def _find_closest_processed_frame(self, current_frame: int, detections_dict: Dict[int, List], frame_skip: int) -> int:
        """Find the closest processed frame to apply detections"""
        # Find the most recent processed frame
        processed_frames = [f for f in detections_dict.keys() if f <= current_frame]
        if processed_frames:
            return max(processed_frames)
        return None
    
    def _draw_predictions_on_frame(self, frame: Any, predictions: List[Dict[Any, Any]]) -> Any:
        """Draw bounding boxes and labels on frame"""
        for pred in predictions:
            if 'x' in pred and 'y' in pred and 'width' in pred and 'height' in pred:
                x = int(pred['x'])
                y = int(pred['y'])
                w = int(pred['width'])
                h = int(pred['height'])
                
                # Calculate bounding box coordinates
                x1 = int(x - w/2)
                y1 = int(y - h/2)
                x2 = int(x + w/2)
                y2 = int(y + h/2)
                
                # Draw rectangle with thicker line for video
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                # Add label if available
                if 'class' in pred:
                    label = pred['class']
                    if 'confidence' in pred:
                        label += f" ({pred['confidence']:.2f})"
                    
                    # Add background for text readability
                    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 255, 0), -1)
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        return frame
    
    def _save_image_result(self, image_path: str, result: Dict[Any, Any], output_dir: str):
        """Save individual image result"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename for result
        image_name = Path(image_path).stem
        result_filename = f"{image_name}_result.json"
        result_path = os.path.join(output_dir, result_filename)
        
        # Save JSON result
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        print(f"Result saved to: {result_path}")
        
        # Optionally save annotated image if predictions exist
        self._save_annotated_image(image_path, result, output_dir)
    
    def _save_batch_results(self, results: List[Dict[Any, Any]], output_dir: str, filename: str):
        """Save batch processing results"""
        os.makedirs(output_dir, exist_ok=True)
        result_path = os.path.join(output_dir, filename)
        
        with open(result_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Batch results saved to: {result_path}")
    
    def _save_video_results(self, video_path: str, results: List[Dict[Any, Any]], output_dir: str):
        """Save video processing results"""
        os.makedirs(output_dir, exist_ok=True)
        
        video_name = Path(video_path).stem
        result_filename = f"{video_name}_video_results.json"
        result_path = os.path.join(output_dir, result_filename)
        
        # Prepare video metadata
        video_results = {
            'video_path': video_path,
            'video_name': video_name,
            'total_processed_frames': len(results),
            'frame_results': results
        }
        
        with open(result_path, 'w') as f:
            json.dump(video_results, f, indent=2, default=str)
        
        print(f"Video results saved to: {result_path}")
    
    def _save_annotated_image(self, image_path: str, result: Dict[Any, Any], output_dir: str):
        """Save image with bounding boxes drawn"""
        try:
            if 'predictions' not in result or not result['predictions']:
                return
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return
            
            # Draw predictions
            image = self._draw_predictions_on_frame(image, result['predictions'])
            
            # Save annotated image
            image_name = Path(image_path).stem
            annotated_filename = f"{image_name}_annotated.jpg"
            annotated_path = os.path.join(output_dir, annotated_filename)
            cv2.imwrite(annotated_path, image)
            
            print(f"Annotated image saved to: {annotated_path}")
            
        except Exception as e:
            print(f"Error creating annotated image: {str(e)}")


def get_user_input():
    """Interactive function to get user preferences"""
    print("=" * 50)
    print("🤖 ROBOFLOW INFERENCE PROCESSOR")
    print("=" * 50)
    
    # Choose processing type
    print("\n📋 What would you like to process?")
    print("1. Single Image")
    print("2. Folder of Images")
    print("3. Video")
    
    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        if choice in ['1', '2', '3']:
            break
        print("❌ Please enter 1, 2, or 3")
    
    type_map = {'1': 'image', '2': 'folder', '3': 'video'}
    input_type = type_map[choice]
    
    # Get input path
    print(f"\n📁 Enter the path to your {input_type}:")
    if input_type == 'image':
        print("   Example: C:/photos/image.jpg or ./image.png")
    elif input_type == 'folder':
        print("   Example: C:/photos/ or ./images/")
    else:  # video
        print("   Example: C:/videos/video.mp4 or ./video.avi")
    
    while True:
        input_path = input("Path: ").strip().strip('"\'')
        
        if input_type == 'image' and os.path.isfile(input_path):
            break
        elif input_type == 'folder' and os.path.isdir(input_path):
            break
        elif input_type == 'video' and os.path.isfile(input_path):
            break
        else:
            if input_type == 'image':
                print("❌ File not found. Please enter a valid image file path.")
            elif input_type == 'folder':
                print("❌ Folder not found. Please enter a valid folder path.")
            else:
                print("❌ Video file not found. Please enter a valid video file path.")
    
    # Ask about saving results
    print(f"\n💾 Do you want to save the results?")
    save_choice = input("Save results? (y/n): ").strip().lower()
    save_results = save_choice in ['y', 'yes', '1', 'true']
    
    # Get output directory if saving
    output_dir = "results"
    if save_results:
        print(f"\n📂 Where should results be saved?")
        custom_output = input(f"Output directory (press Enter for default '{output_dir}'): ").strip()
        if custom_output:
            output_dir = custom_output
    
    # Video-specific settings
    create_output_video = True
    frame_skip = 30
    if input_type == 'video' and save_results:
        print(f"\n🎬 Video output settings:")
        video_choice = input("Create output video with detections? (y/n): ").strip().lower()
        create_output_video = video_choice in ['y', 'yes', '1', 'true']
        
        if create_output_video:
            frame_input = input(f"Process every nth frame (press Enter for default {frame_skip}): ").strip()
            if frame_input.isdigit():
                frame_skip = int(frame_input)
    
    # Additional settings
    confidence_threshold = 0.9
    target_class = "event-box"
    api_key = "hY9qOmC03Dpg4JNVNeOp"
    model_id = "buy-phase-valorant/3"
    
    # Ask about detection settings
    print(f"\n🎯 Detection settings:")
    conf_input = input(f"Confidence threshold (press Enter for default {confidence_threshold}): ").strip()
    if conf_input:
        try:
            custom_conf = float(conf_input)
            if 0 <= custom_conf <= 1:
                confidence_threshold = custom_conf
            else:
                print("⚠️  Confidence must be between 0 and 1, using default")
        except ValueError:
            print("⚠️  Invalid confidence value, using default")
    
    class_input = input(f"Target class to detect (press Enter for default '{target_class}'): ").strip()
    if class_input:
        target_class = class_input
    
    # Ask about custom API settings
    print(f"\n⚙️  Advanced settings (optional):")
    custom_api = input("Use different API key? (press Enter to skip): ").strip()
    if custom_api:
        api_key = custom_api
    
    custom_model = input("Use different model ID? (press Enter to skip): ").strip()
    if custom_model:
        model_id = custom_model
    
    return {
        'input_type': input_type,
        'input_path': input_path,
        'save_results': save_results,
        'output_dir': output_dir,
        'create_output_video': create_output_video if input_type == 'video' else False,
        'frame_skip': frame_skip,
        'confidence_threshold': confidence_threshold,
        'target_class': target_class,
        'api_key': api_key,
        'model_id': model_id
    }

def main():
    try:
        # Get user input interactively
        config = get_user_input()
        
        print(f"\n🚀 Starting processing...")
        print(f"   Type: {config['input_type']}")
        print(f"   Input: {config['input_path']}")
        print(f"   Target class: '{config['target_class']}'")
        print(f"   Confidence threshold: {config['confidence_threshold']}")
        print(f"   Save results: {'Yes' if config['save_results'] else 'No'}")
        if config['save_results']:
            print(f"   Output directory: {config['output_dir']}")
        if config['input_type'] == 'video' and config.get('create_output_video'):
            print(f"   Create output video: Yes")
        
        # Initialize processor
        processor = RoboflowProcessor(
            api_key=config['api_key'], 
            model_id=config['model_id'],
            confidence_threshold=config['confidence_threshold'],
            target_class=config['target_class']
        )
        
        # Process based on input type
        if config['input_type'] == "image":
            result = processor.process_single_image(
                config['input_path'], 
                config['save_results'], 
                config['output_dir']
            )
            if result:
                predictions_count = len(result.get('predictions', []))
                print(f"\n✅ Processing complete! Found {predictions_count} predictions")
                if predictions_count > 0:
                    print("📋 'event-box' detections found:")
                    for i, pred in enumerate(result.get('predictions', [])[:5], 1):  # Show first 5
                        confidence = pred.get('confidence', 0)
                        print(f"   {i}. event-box (confidence: {confidence:.2f})")
                    if len(result.get('predictions', [])) > 5:
                        print(f"   ... and {len(result.get('predictions', [])) - 5} more event-boxes")
        
        elif config['input_type'] == "folder":
            results = processor.process_image_folder(
                config['input_path'], 
                config['save_results'], 
                config['output_dir']
            )
            total_predictions = sum(len(r['result'].get('predictions', [])) for r in results)
            print(f"\n✅ Processing complete!")
            print(f"   📸 Processed {len(results)} images")
            print(f"   📦 Total 'event-box' detections: {total_predictions}")
        
        elif config['input_type'] == "video":
            results = processor.process_video(
                config['input_path'], 
                config['save_results'], 
                config['output_dir'], 
                config['frame_skip'],
                config.get('create_output_video', True)
            )
            total_predictions = sum(len(r['result'].get('predictions', [])) for r in results)
            print(f"\n✅ Processing complete!")
            print(f"   🎬 Processed {len(results)} frames")
            print(f"   📦 Total 'event-box' detections: {total_predictions}")
            if config.get('create_output_video'):
                video_name = Path(config['input_path']).stem
                output_video = os.path.join(config['output_dir'], f"{video_name}_with_detections.mp4")
                print(f"   🎥 Output video: {output_video}")
    
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Processing interrupted by user")
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
    
    print(f"\n👋 Thank you for using Roboflow Inference Processor!")


if __name__ == "__main__":
    # Run the interactive main function
    main()