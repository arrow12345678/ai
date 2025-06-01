"""
Example usage script for ALPR project.
Demonstrates how to use the ALPR system components.
"""

import os
import numpy as np
from src.data_loader import ALPRDataLoader
from src.model_builder import ALPRModelBuilder
from src.train import ALPRTrainer
from src.evaluate import ALPREvaluator
from src.predict import ALPRPredictor


def example_data_loading():
    """Example of data loading and preprocessing."""
    print("=== Data Loading Example ===")
    
    # Initialize data loader
    data_loader = ALPRDataLoader(
        data_dir='data',
        img_size=(640, 640),
        ocr_img_size=(128, 32),
        batch_size=16
    )
    
    # Load dataset splits
    print("Loading dataset splits...")
    train_images, train_annotations = data_loader.load_dataset_split('train')
    print(f"Training set: {len(train_images)} images")
    
    # Create TensorFlow datasets
    print("Creating TensorFlow datasets...")
    train_detection_dataset = data_loader.create_detection_dataset('train')
    train_ocr_dataset = data_loader.create_ocr_dataset('train')
    
    print("Data loading completed!")
    return data_loader


def example_model_building():
    """Example of model building and compilation."""
    print("\n=== Model Building Example ===")
    
    # Initialize model builder
    model_builder = ALPRModelBuilder()
    
    # Build detection model
    print("Building detection model...")
    detection_model = model_builder.build_detection_model(
        input_shape=(640, 640, 3),
        num_classes=1
    )
    model_builder.compile_detection_model(learning_rate=0.001)
    
    print(f"Detection model built with {detection_model.count_params()} parameters")
    
    # Build OCR model
    print("Building OCR model...")
    ocr_model = model_builder.build_ocr_model(
        input_shape=(32, 128, 1),
        num_classes=37  # 36 chars + blank
    )
    model_builder.compile_ocr_model(learning_rate=0.001)
    
    print(f"OCR model built with {ocr_model.count_params()} parameters")
    
    return model_builder


def example_training():
    """Example of model training."""
    print("\n=== Training Example ===")
    
    # Check if data directory exists
    if not os.path.exists('data/train'):
        print("Training data not found. Please prepare your dataset first.")
        return
    
    # Initialize trainer
    trainer = ALPRTrainer(
        data_dir='data',
        model_save_dir='models',
        log_dir='logs'
    )
    
    # Train detection model (small example)
    print("Training detection model...")
    try:
        detection_model, detection_history = trainer.train_detection_model(
            epochs=5,  # Small number for example
            batch_size=8,
            learning_rate=0.001,
            img_size=(640, 640)
        )
        print("Detection model training completed!")
    except Exception as e:
        print(f"Detection training failed: {e}")
    
    # Train OCR model (small example)
    print("Training OCR model...")
    try:
        ocr_model, ocr_history = trainer.train_ocr_model(
            epochs=5,  # Small number for example
            batch_size=16,
            learning_rate=0.001,
            img_size=(128, 32)
        )
        print("OCR model training completed!")
    except Exception as e:
        print(f"OCR training failed: {e}")


def example_evaluation():
    """Example of model evaluation."""
    print("\n=== Evaluation Example ===")
    
    # Check if models exist
    detection_model_path = 'models/detection_model_best.h5'
    ocr_model_path = 'models/ocr_model_best.h5'
    
    if not (os.path.exists(detection_model_path) and os.path.exists(ocr_model_path)):
        print("Trained models not found. Please train models first.")
        return
    
    # Initialize evaluator
    evaluator = ALPREvaluator(
        data_dir='data',
        detection_model_path=detection_model_path,
        ocr_model_path=ocr_model_path
    )
    
    # Run evaluation
    try:
        results = evaluator.evaluate_all(save_results=True)
        
        # Print results summary
        if 'detection' in results:
            det_results = results['detection']
            print(f"Detection Results:")
            print(f"  Precision: {det_results['precision']:.4f}")
            print(f"  Recall: {det_results['recall']:.4f}")
            print(f"  F1-Score: {det_results['f1_score']:.4f}")
        
        if 'ocr' in results:
            ocr_results = results['ocr']
            print(f"OCR Results:")
            print(f"  Character Accuracy: {ocr_results['character_accuracy']:.4f}")
            print(f"  Word Accuracy: {ocr_results['word_accuracy']:.4f}")
            
    except Exception as e:
        print(f"Evaluation failed: {e}")


def example_prediction():
    """Example of prediction on new images."""
    print("\n=== Prediction Example ===")
    
    # Check if models exist
    detection_model_path = 'models/detection_model_best.h5'
    ocr_model_path = 'models/ocr_model_best.h5'
    
    if not (os.path.exists(detection_model_path) and os.path.exists(ocr_model_path)):
        print("Trained models not found. Please train models first.")
        return
    
    # Initialize predictor
    try:
        predictor = ALPRPredictor(
            detection_model_path=detection_model_path,
            ocr_model_path=ocr_model_path,
            confidence_threshold=0.5
        )
        
        # Example prediction on test images
        test_images_dir = 'data/test/images'
        if os.path.exists(test_images_dir):
            import glob
            test_images = glob.glob(os.path.join(test_images_dir, '*.jpg'))[:5]  # First 5 images
            
            if test_images:
                print(f"Running prediction on {len(test_images)} test images...")
                results = predictor.predict_batch(test_images)
                
                # Print results
                for img_path, detections in results.items():
                    print(f"\nImage: {os.path.basename(img_path)}")
                    if detections:
                        for i, detection in enumerate(detections):
                            print(f"  Plate {i+1}: '{detection['text']}' (confidence: {detection['confidence']:.3f})")
                    else:
                        print("  No license plates detected")
            else:
                print("No test images found")
        else:
            print("Test images directory not found")
            
    except Exception as e:
        print(f"Prediction failed: {e}")


def example_data_splitting():
    """Example of splitting raw dataset."""
    print("\n=== Data Splitting Example ===")
    
    # This example shows how to split a raw dataset
    raw_data_dir = 'raw_data'  # Replace with your raw data directory
    
    if os.path.exists(raw_data_dir):
        data_loader = ALPRDataLoader('data')
        
        try:
            data_loader.split_dataset(
                source_dir=raw_data_dir,
                train_ratio=0.8,
                val_ratio=0.1
                # test_ratio will be 0.1 (remaining)
            )
            print("Dataset splitting completed!")
        except Exception as e:
            print(f"Dataset splitting failed: {e}")
    else:
        print(f"Raw data directory '{raw_data_dir}' not found")
        print("To use this feature, place your images and labels in:")
        print(f"  {raw_data_dir}/images/")
        print(f"  {raw_data_dir}/labels/")


def main():
    """Run all examples."""
    print("🚗 ALPR Project - Example Usage")
    print("=" * 50)
    
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Run examples
    try:
        # Data loading example
        data_loader = example_data_loading()
        
        # Model building example
        model_builder = example_model_building()
        
        # Data splitting example (if raw data exists)
        example_data_splitting()
        
        # Training example (if data exists)
        example_training()
        
        # Evaluation example (if models exist)
        example_evaluation()
        
        # Prediction example (if models exist)
        example_prediction()
        
    except Exception as e:
        print(f"Example execution failed: {e}")
    
    print("\n" + "=" * 50)
    print("Example usage completed!")
    print("\nNext steps:")
    print("1. Prepare your dataset in the 'data' directory")
    print("2. Run training: python src/train.py --data_dir data --model both")
    print("3. Evaluate models: python src/evaluate.py --data_dir data --detection_model models/detection_model_best.h5 --ocr_model models/ocr_model_best.h5")
    print("4. Make predictions: python src/predict.py --detection_model models/detection_model_best.h5 --ocr_model models/ocr_model_best.h5 --input path/to/image.jpg")


if __name__ == '__main__':
    main()
