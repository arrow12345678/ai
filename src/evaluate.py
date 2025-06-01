"""
Evaluation module for ALPR project.
Provides comprehensive evaluation metrics for both detection and OCR models.
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from typing import List, Dict, Tuple
import json

from data_loader import ALPRDataLoader
from model_builder import ALPRModelBuilder


class DetectionEvaluator:
    """Evaluator for license plate detection model."""
    
    def __init__(self, model_path: str, data_loader: ALPRDataLoader):
        """
        Initialize detection evaluator.
        
        Args:
            model_path: Path to trained detection model
            data_loader: Data loader instance
        """
        self.model = keras.models.load_model(model_path)
        self.data_loader = data_loader
        
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            box1: First bounding box [x_center, y_center, width, height] (normalized)
            box2: Second bounding box [x_center, y_center, width, height] (normalized)
            
        Returns:
            IoU value
        """
        # Convert center format to corner format
        def center_to_corner(box):
            x_center, y_center, width, height = box
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            return [x1, y1, x2, y2]
        
        box1_corner = center_to_corner(box1)
        box2_corner = center_to_corner(box2)
        
        # Calculate intersection
        x1 = max(box1_corner[0], box2_corner[0])
        y1 = max(box1_corner[1], box2_corner[1])
        x2 = min(box1_corner[2], box2_corner[2])
        y2 = min(box1_corner[3], box2_corner[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        area1 = (box1_corner[2] - box1_corner[0]) * (box1_corner[3] - box1_corner[1])
        area2 = (box2_corner[2] - box2_corner[0]) * (box2_corner[3] - box2_corner[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def evaluate_detection(self, test_dataset, iou_threshold: float = 0.5) -> Dict:
        """
        Evaluate detection model performance.
        
        Args:
            test_dataset: Test dataset
            iou_threshold: IoU threshold for positive detection
            
        Returns:
            Dictionary containing evaluation metrics
        """
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        total_predictions = 0
        iou_scores = []
        
        print("Evaluating detection model...")
        
        for batch_idx, (images, targets) in enumerate(test_dataset):
            # Get predictions
            predictions = self.model(images, training=False)
            
            batch_size = tf.shape(images)[0]
            
            for i in range(batch_size):
                # Extract ground truth bboxes
                gt_bboxes = targets['bboxes'][i].numpy()
                gt_labels = targets['labels'][i].numpy()
                
                # Process predictions (simplified - in practice, you'd need NMS)
                # This is a placeholder for actual prediction processing
                pred_bboxes = self._extract_predictions(predictions, i)
                
                total_predictions += len(pred_bboxes)
                
                # Match predictions with ground truth
                matched_gt = set()
                
                for pred_bbox in pred_bboxes:
                    best_iou = 0
                    best_gt_idx = -1
                    
                    for gt_idx, gt_bbox in enumerate(gt_bboxes):
                        if gt_idx in matched_gt:
                            continue
                            
                        iou = self.calculate_iou(pred_bbox, gt_bbox)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx
                    
                    if best_iou >= iou_threshold:
                        true_positives += 1
                        matched_gt.add(best_gt_idx)
                        iou_scores.append(best_iou)
                    else:
                        false_positives += 1
                
                # Count unmatched ground truth as false negatives
                false_negatives += len(gt_bboxes) - len(matched_gt)
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        mean_iou = np.mean(iou_scores) if iou_scores else 0
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'mean_iou': mean_iou,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'total_predictions': total_predictions
        }
        
        return metrics
    
    def _extract_predictions(self, predictions: Dict, batch_idx: int) -> List[List[float]]:
        """
        Extract bounding box predictions from model output.
        This is a simplified version - in practice, you'd implement proper post-processing.
        
        Args:
            predictions: Model predictions
            batch_idx: Batch index
            
        Returns:
            List of predicted bounding boxes
        """
        # Placeholder implementation
        # In a real implementation, you would:
        # 1. Apply sigmoid/softmax to appropriate outputs
        # 2. Decode the predictions from grid format
        # 3. Apply Non-Maximum Suppression (NMS)
        # 4. Filter by confidence threshold
        
        pred_bboxes = []
        
        # For now, return empty list (you would implement actual prediction extraction)
        return pred_bboxes


class OCREvaluator:
    """Evaluator for OCR model."""
    
    def __init__(self, model_path: str, data_loader: ALPRDataLoader):
        """
        Initialize OCR evaluator.
        
        Args:
            model_path: Path to trained OCR model
            data_loader: Data loader instance
        """
        self.model = keras.models.load_model(model_path)
        self.data_loader = data_loader
        
    def decode_predictions(self, predictions: np.ndarray) -> List[str]:
        """
        Decode CTC predictions to text.
        
        Args:
            predictions: Model predictions
            
        Returns:
            List of decoded text strings
        """
        decoded_texts = []
        
        for pred in predictions:
            # Apply CTC decoding
            input_length = np.array([pred.shape[0]])
            decoded = keras.backend.ctc_decode(
                pred[np.newaxis, :, :], 
                input_length, 
                greedy=True
            )[0][0]
            
            # Convert to text
            decoded_text = ""
            for idx in decoded[0]:
                if idx < self.data_loader.num_classes:
                    decoded_text += self.data_loader.idx_to_char[idx]
            
            decoded_texts.append(decoded_text)
        
        return decoded_texts
    
    def calculate_character_accuracy(self, predicted: str, ground_truth: str) -> float:
        """Calculate character-level accuracy."""
        if len(ground_truth) == 0:
            return 1.0 if len(predicted) == 0 else 0.0
        
        correct_chars = sum(1 for p, g in zip(predicted, ground_truth) if p == g)
        return correct_chars / len(ground_truth)
    
    def calculate_word_accuracy(self, predicted: str, ground_truth: str) -> float:
        """Calculate word-level (exact match) accuracy."""
        return 1.0 if predicted.strip() == ground_truth.strip() else 0.0
    
    def calculate_edit_distance(self, predicted: str, ground_truth: str) -> int:
        """Calculate Levenshtein edit distance."""
        if len(predicted) == 0:
            return len(ground_truth)
        if len(ground_truth) == 0:
            return len(predicted)
        
        # Create matrix
        matrix = [[0] * (len(ground_truth) + 1) for _ in range(len(predicted) + 1)]
        
        # Initialize first row and column
        for i in range(len(predicted) + 1):
            matrix[i][0] = i
        for j in range(len(ground_truth) + 1):
            matrix[0][j] = j
        
        # Fill matrix
        for i in range(1, len(predicted) + 1):
            for j in range(1, len(ground_truth) + 1):
                if predicted[i-1] == ground_truth[j-1]:
                    matrix[i][j] = matrix[i-1][j-1]
                else:
                    matrix[i][j] = min(
                        matrix[i-1][j] + 1,      # deletion
                        matrix[i][j-1] + 1,      # insertion
                        matrix[i-1][j-1] + 1     # substitution
                    )
        
        return matrix[len(predicted)][len(ground_truth)]
    
    def evaluate_ocr(self, test_dataset) -> Dict:
        """
        Evaluate OCR model performance.
        
        Args:
            test_dataset: Test dataset
            
        Returns:
            Dictionary containing evaluation metrics
        """
        total_samples = 0
        total_char_accuracy = 0
        total_word_accuracy = 0
        total_edit_distance = 0
        
        all_predictions = []
        all_ground_truths = []
        
        print("Evaluating OCR model...")
        
        for batch_idx, (images, targets) in enumerate(test_dataset):
            # Get predictions
            predictions = self.model(images, training=False)
            
            # Decode predictions
            decoded_predictions = self.decode_predictions(predictions.numpy())
            
            # Extract ground truth texts
            batch_size = tf.shape(images)[0]
            for i in range(batch_size):
                # Decode ground truth
                gt_encoded = targets['text'][i].numpy()
                gt_text = self.data_loader.decode_text(gt_encoded)
                
                pred_text = decoded_predictions[i] if i < len(decoded_predictions) else ""
                
                # Calculate metrics
                char_acc = self.calculate_character_accuracy(pred_text, gt_text)
                word_acc = self.calculate_word_accuracy(pred_text, gt_text)
                edit_dist = self.calculate_edit_distance(pred_text, gt_text)
                
                total_char_accuracy += char_acc
                total_word_accuracy += word_acc
                total_edit_distance += edit_dist
                total_samples += 1
                
                all_predictions.append(pred_text)
                all_ground_truths.append(gt_text)
        
        # Calculate average metrics
        avg_char_accuracy = total_char_accuracy / total_samples if total_samples > 0 else 0
        avg_word_accuracy = total_word_accuracy / total_samples if total_samples > 0 else 0
        avg_edit_distance = total_edit_distance / total_samples if total_samples > 0 else 0
        
        # Calculate Character Error Rate (CER)
        total_chars = sum(len(gt) for gt in all_ground_truths)
        cer = total_edit_distance / total_chars if total_chars > 0 else 0
        
        metrics = {
            'character_accuracy': avg_char_accuracy,
            'word_accuracy': avg_word_accuracy,
            'average_edit_distance': avg_edit_distance,
            'character_error_rate': cer,
            'total_samples': total_samples,
            'predictions': all_predictions[:10],  # Sample predictions
            'ground_truths': all_ground_truths[:10]  # Sample ground truths
        }
        
        return metrics


class ALPREvaluator:
    """Main evaluator class for ALPR system."""
    
    def __init__(self, 
                 data_dir: str,
                 detection_model_path: str = None,
                 ocr_model_path: str = None):
        """
        Initialize ALPR evaluator.
        
        Args:
            data_dir: Directory containing test data
            detection_model_path: Path to detection model
            ocr_model_path: Path to OCR model
        """
        self.data_loader = ALPRDataLoader(data_dir)
        
        self.detection_evaluator = None
        self.ocr_evaluator = None
        
        if detection_model_path and os.path.exists(detection_model_path):
            self.detection_evaluator = DetectionEvaluator(detection_model_path, self.data_loader)
            
        if ocr_model_path and os.path.exists(ocr_model_path):
            self.ocr_evaluator = OCREvaluator(ocr_model_path, self.data_loader)
    
    def evaluate_all(self, save_results: bool = True) -> Dict:
        """
        Evaluate both detection and OCR models.
        
        Args:
            save_results: Whether to save results to file
            
        Returns:
            Dictionary containing all evaluation results
        """
        results = {}
        
        # Load test dataset
        test_detection_dataset = self.data_loader.create_detection_dataset('test', shuffle=False)
        test_ocr_dataset = self.data_loader.create_ocr_dataset('test', shuffle=False)
        
        # Evaluate detection model
        if self.detection_evaluator:
            print("Evaluating detection model...")
            detection_results = self.detection_evaluator.evaluate_detection(test_detection_dataset)
            results['detection'] = detection_results
            
            print("\nDetection Results:")
            print(f"Precision: {detection_results['precision']:.4f}")
            print(f"Recall: {detection_results['recall']:.4f}")
            print(f"F1-Score: {detection_results['f1_score']:.4f}")
            print(f"Mean IoU: {detection_results['mean_iou']:.4f}")
        
        # Evaluate OCR model
        if self.ocr_evaluator:
            print("\nEvaluating OCR model...")
            ocr_results = self.ocr_evaluator.evaluate_ocr(test_ocr_dataset)
            results['ocr'] = ocr_results
            
            print("\nOCR Results:")
            print(f"Character Accuracy: {ocr_results['character_accuracy']:.4f}")
            print(f"Word Accuracy: {ocr_results['word_accuracy']:.4f}")
            print(f"Character Error Rate: {ocr_results['character_error_rate']:.4f}")
            print(f"Average Edit Distance: {ocr_results['average_edit_distance']:.2f}")
        
        # Save results
        if save_results:
            self._save_results(results)
        
        return results
    
    def _save_results(self, results: Dict):
        """Save evaluation results to JSON file."""
        results_path = 'evaluation_results.json'
        
        # Convert numpy types to Python types for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                serializable_results[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        serializable_results[key][k] = v.tolist()
                    elif isinstance(v, (np.int64, np.int32, np.float64, np.float32)):
                        serializable_results[key][k] = v.item()
                    else:
                        serializable_results[key][k] = v
            else:
                serializable_results[key] = value
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nResults saved to: {results_path}")


def main():
    """Main evaluation function with CLI interface."""
    parser = argparse.ArgumentParser(description='Evaluate ALPR models')
    
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing test data')
    parser.add_argument('--detection_model', type=str,
                       help='Path to detection model')
    parser.add_argument('--ocr_model', type=str,
                       help='Path to OCR model')
    parser.add_argument('--save_results', action='store_true',
                       help='Save results to JSON file')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ALPREvaluator(
        data_dir=args.data_dir,
        detection_model_path=args.detection_model,
        ocr_model_path=args.ocr_model
    )
    
    # Run evaluation
    results = evaluator.evaluate_all(save_results=args.save_results)
    
    print("\nEvaluation completed!")


if __name__ == '__main__':
    main()
