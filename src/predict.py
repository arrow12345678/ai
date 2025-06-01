"""
Prediction/Inference module for ALPR project.
Handles end-to-end license plate recognition on new images.
"""

import os
import argparse
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import json

from data_loader import ALPRDataLoader


class ALPRPredictor:
    """End-to-end ALPR prediction system."""
    
    def __init__(self, 
                 detection_model_path: str,
                 ocr_model_path: str,
                 confidence_threshold: float = 0.5,
                 nms_threshold: float = 0.4):
        """
        Initialize ALPR predictor.
        
        Args:
            detection_model_path: Path to trained detection model
            ocr_model_path: Path to trained OCR model
            confidence_threshold: Confidence threshold for detections
            nms_threshold: NMS threshold for duplicate removal
        """
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        # Load models
        print("Loading detection model...")
        self.detection_model = keras.models.load_model(detection_model_path)
        
        print("Loading OCR model...")
        self.ocr_model = keras.models.load_model(ocr_model_path)
        
        # Initialize data loader for preprocessing and text decoding
        self.data_loader = ALPRDataLoader('.')  # Dummy path
        
        print("Models loaded successfully!")
    
    def preprocess_image_for_detection(self, image: np.ndarray, target_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
        """
        Preprocess image for detection model.
        
        Args:
            image: Input image
            target_size: Target size (width, height)
            
        Returns:
            Preprocessed image
        """
        # Resize image while maintaining aspect ratio
        h, w = image.shape[:2]
        scale = min(target_size[0] / w, target_size[1] / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h))
        
        # Create padded image
        padded = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        
        # Center the resized image
        y_offset = (target_size[1] - new_h) // 2
        x_offset = (target_size[0] - new_w) // 2
        padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        # Normalize
        normalized = padded.astype(np.float32) / 255.0
        
        return normalized, scale, (x_offset, y_offset)
    
    def postprocess_detections(self, 
                             predictions: Dict, 
                             original_shape: Tuple[int, int],
                             scale: float,
                             offset: Tuple[int, int]) -> List[Dict]:
        """
        Post-process detection predictions.
        
        Args:
            predictions: Raw model predictions
            original_shape: Original image shape (height, width)
            scale: Scale factor used in preprocessing
            offset: Offset used in preprocessing (x_offset, y_offset)
            
        Returns:
            List of detection dictionaries
        """
        detections = []
        
        # This is a simplified post-processing
        # In practice, you would implement proper YOLO post-processing
        # including anchor decoding, NMS, etc.
        
        # For now, return empty list as placeholder
        # You would implement:
        # 1. Decode predictions from grid format
        # 2. Apply confidence filtering
        # 3. Convert coordinates back to original image space
        # 4. Apply Non-Maximum Suppression
        
        return detections
    
    def extract_license_plate(self, image: np.ndarray, bbox: List[float]) -> np.ndarray:
        """
        Extract license plate region from image.
        
        Args:
            image: Original image
            bbox: Bounding box [x1, y1, x2, y2] in pixel coordinates
            
        Returns:
            Extracted license plate image
        """
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Ensure coordinates are within image bounds
        h, w = image.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        # Extract region
        plate_img = image[y1:y2, x1:x2]
        
        return plate_img
    
    def preprocess_plate_for_ocr(self, plate_image: np.ndarray, target_size: Tuple[int, int] = (128, 32)) -> np.ndarray:
        """
        Preprocess license plate image for OCR.
        
        Args:
            plate_image: License plate image
            target_size: Target size (width, height)
            
        Returns:
            Preprocessed image for OCR
        """
        # Convert to grayscale
        if len(plate_image.shape) == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Resize to target size
        resized = cv2.resize(thresh, target_size)
        
        # Normalize and add channel dimension
        normalized = resized.astype(np.float32) / 255.0
        normalized = np.expand_dims(normalized, axis=-1)
        
        return normalized
    
    def decode_ocr_prediction(self, prediction: np.ndarray) -> str:
        """
        Decode OCR prediction to text.
        
        Args:
            prediction: OCR model prediction
            
        Returns:
            Decoded text string
        """
        # Apply CTC decoding
        input_length = np.array([prediction.shape[0]])
        decoded = keras.backend.ctc_decode(
            prediction[np.newaxis, :, :], 
            input_length, 
            greedy=True
        )[0][0]
        
        # Convert to text
        decoded_text = ""
        for idx in decoded[0]:
            if idx < self.data_loader.num_classes:
                decoded_text += self.data_loader.idx_to_char[idx]
        
        return decoded_text.strip()
    
    def predict_single_image(self, image_path: str) -> List[Dict]:
        """
        Perform ALPR on a single image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            List of detection results with recognized text
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_shape = image.shape[:2]
        
        # Step 1: Detect license plates
        preprocessed, scale, offset = self.preprocess_image_for_detection(image_rgb)
        detection_input = np.expand_dims(preprocessed, axis=0)
        
        # Get detection predictions
        detection_predictions = self.detection_model(detection_input, training=False)
        
        # Post-process detections
        detections = self.postprocess_detections(
            detection_predictions, original_shape, scale, offset
        )
        
        results = []
        
        # Step 2: Recognize text for each detected plate
        for detection in detections:
            bbox = detection['bbox']  # [x1, y1, x2, y2]
            confidence = detection['confidence']
            
            # Extract license plate region
            plate_img = self.extract_license_plate(image_rgb, bbox)
            
            if plate_img.size == 0:
                continue
            
            # Preprocess for OCR
            ocr_input = self.preprocess_plate_for_ocr(plate_img)
            ocr_input = np.expand_dims(ocr_input, axis=0)
            
            # Get OCR prediction
            ocr_prediction = self.ocr_model(ocr_input, training=False)
            
            # Decode text
            recognized_text = self.decode_ocr_prediction(ocr_prediction[0])
            
            # Add to results
            result = {
                'bbox': bbox,
                'confidence': confidence,
                'text': recognized_text,
                'plate_image': plate_img
            }
            results.append(result)
        
        return results
    
    def predict_batch(self, image_paths: List[str]) -> Dict[str, List[Dict]]:
        """
        Perform ALPR on multiple images.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            Dictionary mapping image paths to results
        """
        batch_results = {}
        
        for image_path in image_paths:
            try:
                results = self.predict_single_image(image_path)
                batch_results[image_path] = results
                print(f"Processed: {image_path} - Found {len(results)} plates")
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                batch_results[image_path] = []
        
        return batch_results
    
    def visualize_results(self, image_path: str, results: List[Dict], save_path: str = None):
        """
        Visualize ALPR results on image.
        
        Args:
            image_path: Path to original image
            results: ALPR results
            save_path: Path to save visualization (optional)
        """
        # Load image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        plt.imshow(image_rgb)
        plt.axis('off')
        
        # Draw bounding boxes and text
        for i, result in enumerate(results):
            bbox = result['bbox']
            text = result['text']
            confidence = result['confidence']
            
            x1, y1, x2, y2 = bbox
            
            # Draw bounding box
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                               fill=False, color='red', linewidth=2)
            plt.gca().add_patch(rect)
            
            # Add text label
            label = f"{text} ({confidence:.2f})"
            plt.text(x1, y1-10, label, color='red', fontsize=12, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.title(f"ALPR Results - {os.path.basename(image_path)}")
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Visualization saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    """Main prediction function with CLI interface."""
    parser = argparse.ArgumentParser(description='ALPR Prediction')
    
    parser.add_argument('--detection_model', type=str, required=True,
                       help='Path to detection model')
    parser.add_argument('--ocr_model', type=str, required=True,
                       help='Path to OCR model')
    parser.add_argument('--input', type=str, required=True,
                       help='Input image path or directory')
    parser.add_argument('--output_dir', type=str, default='predictions',
                       help='Output directory for results')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                       help='Confidence threshold for detections')
    parser.add_argument('--visualize', action='store_true',
                       help='Save visualization images')
    parser.add_argument('--save_json', action='store_true',
                       help='Save results as JSON')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize predictor
    predictor = ALPRPredictor(
        detection_model_path=args.detection_model,
        ocr_model_path=args.ocr_model,
        confidence_threshold=args.confidence_threshold
    )
    
    # Get input images
    if os.path.isfile(args.input):
        image_paths = [args.input]
    elif os.path.isdir(args.input):
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_paths.extend(glob.glob(os.path.join(args.input, ext)))
    else:
        raise ValueError(f"Invalid input path: {args.input}")
    
    print(f"Processing {len(image_paths)} images...")
    
    # Run predictions
    all_results = predictor.predict_batch(image_paths)
    
    # Save results
    if args.save_json:
        results_path = os.path.join(args.output_dir, 'predictions.json')
        
        # Convert results to JSON-serializable format
        json_results = {}
        for img_path, results in all_results.items():
            json_results[img_path] = []
            for result in results:
                json_result = {
                    'bbox': result['bbox'],
                    'confidence': float(result['confidence']),
                    'text': result['text']
                }
                json_results[img_path].append(json_result)
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Results saved to: {results_path}")
    
    # Create visualizations
    if args.visualize:
        for img_path, results in all_results.items():
            if results:  # Only visualize if there are detections
                img_name = os.path.splitext(os.path.basename(img_path))[0]
                vis_path = os.path.join(args.output_dir, f"{img_name}_result.png")
                predictor.visualize_results(img_path, results, vis_path)
    
    # Print summary
    total_detections = sum(len(results) for results in all_results.values())
    print(f"\nProcessing completed!")
    print(f"Total images processed: {len(image_paths)}")
    print(f"Total license plates detected: {total_detections}")


if __name__ == '__main__':
    import glob
    main()
