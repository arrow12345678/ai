"""
Data loading and preprocessing module for ALPR project.
Handles image loading, label parsing, data augmentation, and TensorFlow data pipeline creation.
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import albumentations as A
from typing import Tuple, List, Dict, Optional
import xml.etree.ElementTree as ET
import glob
import json


class ALPRDataLoader:
    """Data loader for Automatic License Plate Recognition."""
    
    def __init__(self, 
                 data_dir: str,
                 img_size: Tuple[int, int] = (640, 640),
                 ocr_img_size: Tuple[int, int] = (128, 32),
                 batch_size: int = 16):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Root directory containing train/val/test folders
            img_size: Target image size for detection model (width, height)
            ocr_img_size: Target image size for OCR model (width, height)
            batch_size: Batch size for training
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.ocr_img_size = ocr_img_size
        self.batch_size = batch_size
        
        # Character set for OCR (numbers + uppercase letters)
        self.char_set = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.char_to_idx = {char: idx for idx, char in enumerate(self.char_set)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.char_set)}
        self.num_classes = len(self.char_set)
        
        # Data augmentation pipeline
        self.train_augmentation = A.Compose([
            A.HorizontalFlip(p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.MotionBlur(blur_limit=3, p=0.3),
            A.Rotate(limit=5, p=0.3),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        
        self.val_augmentation = A.Compose([
            # No augmentation for validation
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    def load_yolo_annotations(self, label_path: str) -> List[Dict]:
        """
        Load YOLO format annotations.
        
        Args:
            label_path: Path to YOLO .txt file
            
        Returns:
            List of annotation dictionaries
        """
        annotations = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Extract text if available (after bbox coordinates)
                        text = ' '.join(parts[5:]) if len(parts) > 5 else ""
                        
                        annotations.append({
                            'class_id': class_id,
                            'bbox': [x_center, y_center, width, height],
                            'text': text
                        })
        return annotations

    def load_pascal_voc_annotations(self, xml_path: str) -> List[Dict]:
        """
        Load PASCAL VOC format annotations.
        
        Args:
            xml_path: Path to XML annotation file
            
        Returns:
            List of annotation dictionaries
        """
        annotations = []
        if os.path.exists(xml_path):
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Get image dimensions
            size = root.find('size')
            img_width = int(size.find('width').text)
            img_height = int(size.find('height').text)
            
            # Extract objects
            for obj in root.findall('object'):
                name = obj.find('name').text
                bbox = obj.find('bndbox')
                
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                
                # Convert to YOLO format (normalized center coordinates)
                x_center = (xmin + xmax) / 2.0 / img_width
                y_center = (ymin + ymax) / 2.0 / img_height
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height
                
                annotations.append({
                    'class_id': 0,  # Assuming single class (license plate)
                    'bbox': [x_center, y_center, width, height],
                    'text': name if name != 'license_plate' else ""
                })
                
        return annotations

    def preprocess_image(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Preprocess image for model input.
        
        Args:
            image: Input image
            target_size: Target size (width, height)
            
        Returns:
            Preprocessed image
        """
        # Resize image
        image = cv2.resize(image, target_size)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        return image

    def preprocess_ocr_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image specifically for OCR.
        
        Args:
            image: Input license plate image
            
        Returns:
            Preprocessed image for OCR
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Resize to OCR input size
        resized = cv2.resize(thresh, self.ocr_img_size)
        
        # Normalize and add channel dimension
        normalized = resized.astype(np.float32) / 255.0
        return np.expand_dims(normalized, axis=-1)

    def encode_text(self, text: str, max_length: int = 10) -> np.ndarray:
        """
        Encode text to numerical representation for CTC loss.
        
        Args:
            text: Input text
            max_length: Maximum sequence length
            
        Returns:
            Encoded text sequence
        """
        # Convert to uppercase and filter valid characters
        text = text.upper()
        encoded = []
        
        for char in text:
            if char in self.char_to_idx:
                encoded.append(self.char_to_idx[char])
                
        # Pad or truncate to max_length
        if len(encoded) > max_length:
            encoded = encoded[:max_length]
        else:
            encoded.extend([self.num_classes] * (max_length - len(encoded)))  # Use num_classes as blank token
            
        return np.array(encoded, dtype=np.int32)

    def decode_text(self, encoded: np.ndarray) -> str:
        """
        Decode numerical representation back to text.
        
        Args:
            encoded: Encoded text sequence
            
        Returns:
            Decoded text string
        """
        decoded = []
        for idx in encoded:
            if idx < self.num_classes:
                decoded.append(self.idx_to_char[idx])
        return ''.join(decoded)

    def load_dataset_split(self, split: str) -> Tuple[List[str], List[List[Dict]]]:
        """
        Load a dataset split (train/val/test).
        
        Args:
            split: Dataset split name ('train', 'val', 'test')
            
        Returns:
            Tuple of (image_paths, annotations_list)
        """
        split_dir = os.path.join(self.data_dir, split)
        images_dir = os.path.join(split_dir, 'images')
        labels_dir = os.path.join(split_dir, 'labels')
        
        image_paths = []
        annotations_list = []
        
        # Get all image files
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_paths.extend(glob.glob(os.path.join(images_dir, ext)))
            
        for img_path in image_paths:
            # Get corresponding label file
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            
            # Try YOLO format first
            yolo_label_path = os.path.join(labels_dir, f"{img_name}.txt")
            xml_label_path = os.path.join(labels_dir, f"{img_name}.xml")
            
            if os.path.exists(yolo_label_path):
                annotations = self.load_yolo_annotations(yolo_label_path)
            elif os.path.exists(xml_label_path):
                annotations = self.load_pascal_voc_annotations(xml_label_path)
            else:
                annotations = []  # No annotations found
                
            annotations_list.append(annotations)
            
        return image_paths, annotations_list
