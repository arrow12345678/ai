"""
Training module for ALPR project.
Handles training of both detection and OCR models.
"""

import os
import argparse
import tensorflow as tf
from tensorflow import keras
import numpy as np
from datetime import datetime
import json

from data_loader import ALPRDataLoader
from model_builder import ALPRModelBuilder


class ALPRTrainer:
    """Main trainer class for ALPR models."""
    
    def __init__(self, 
                 data_dir: str,
                 model_save_dir: str = 'models',
                 log_dir: str = 'logs'):
        """
        Initialize trainer.
        
        Args:
            data_dir: Directory containing training data
            model_save_dir: Directory to save trained models
            log_dir: Directory for TensorBoard logs
        """
        self.data_dir = data_dir
        self.model_save_dir = model_save_dir
        self.log_dir = log_dir
        
        # Create directories
        os.makedirs(model_save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize components
        self.data_loader = ALPRDataLoader(data_dir)
        self.model_builder = ALPRModelBuilder()
        
        # Training history
        self.training_history = {}
        
    def create_callbacks(self, model_name: str, monitor: str = 'val_loss'):
        """Create training callbacks."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        callbacks = [
            # Model checkpoint
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.model_save_dir, f'{model_name}_best.h5'),
                monitor=monitor,
                save_best_only=True,
                save_weights_only=False,
                mode='min' if 'loss' in monitor else 'max',
                verbose=1
            ),
            
            # Early stopping
            keras.callbacks.EarlyStopping(
                monitor=monitor,
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            keras.callbacks.ReduceLROnPlateau(
                monitor=monitor,
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # TensorBoard logging
            keras.callbacks.TensorBoard(
                log_dir=os.path.join(self.log_dir, f'{model_name}_{timestamp}'),
                histogram_freq=1,
                write_graph=True,
                write_images=True
            ),
            
            # CSV logger
            keras.callbacks.CSVLogger(
                filename=os.path.join(self.log_dir, f'{model_name}_training.csv'),
                append=True
            )
        ]
        
        return callbacks
    
    def train_detection_model(self,
                            epochs: int = 100,
                            batch_size: int = 16,
                            learning_rate: float = 0.001,
                            img_size: tuple = (640, 640),
                            resume_from: str = None):
        """
        Train the detection model.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            img_size: Input image size (width, height)
            resume_from: Path to model to resume training from
        """
        print("Starting detection model training...")
        
        # Update data loader settings
        self.data_loader.img_size = img_size
        self.data_loader.batch_size = batch_size
        
        # Create datasets
        print("Loading datasets...")
        train_dataset = self.data_loader.create_detection_dataset('train', shuffle=True)
        val_dataset = self.data_loader.create_detection_dataset('val', shuffle=False)
        
        # Build or load model
        if resume_from and os.path.exists(resume_from):
            print(f"Resuming training from {resume_from}")
            model = keras.models.load_model(resume_from)
        else:
            print("Building new detection model...")
            model = self.model_builder.build_detection_model(
                input_shape=(*img_size, 3),
                num_classes=1
            )
            self.model_builder.compile_detection_model(learning_rate=learning_rate)
            model = self.model_builder.detection_model
        
        # Print model summary
        print("\nDetection Model Summary:")
        model.summary()
        
        # Create callbacks
        callbacks = self.create_callbacks('detection_model', monitor='val_loss')
        
        # Train model
        print(f"\nStarting training for {epochs} epochs...")
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        final_model_path = os.path.join(self.model_save_dir, 'detection_model_final.h5')
        model.save(final_model_path)
        print(f"Final model saved to: {final_model_path}")
        
        # Save training history
        self.training_history['detection'] = history.history
        self._save_training_history()
        
        return model, history
    
    def train_ocr_model(self,
                       epochs: int = 100,
                       batch_size: int = 32,
                       learning_rate: float = 0.001,
                       img_size: tuple = (128, 32),
                       resume_from: str = None):
        """
        Train the OCR model.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            img_size: Input image size (width, height)
            resume_from: Path to model to resume training from
        """
        print("Starting OCR model training...")
        
        # Update data loader settings
        self.data_loader.ocr_img_size = img_size
        self.data_loader.batch_size = batch_size
        
        # Create datasets
        print("Loading datasets...")
        train_dataset = self.data_loader.create_ocr_dataset('train', shuffle=True)
        val_dataset = self.data_loader.create_ocr_dataset('val', shuffle=False)
        
        # Build or load model
        if resume_from and os.path.exists(resume_from):
            print(f"Resuming training from {resume_from}")
            model = keras.models.load_model(resume_from)
        else:
            print("Building new OCR model...")
            model = self.model_builder.build_ocr_model(
                input_shape=(*img_size, 1),
                num_classes=self.data_loader.num_classes + 1  # +1 for CTC blank
            )
            self.model_builder.compile_ocr_model(learning_rate=learning_rate)
            model = self.model_builder.ocr_model
        
        # Print model summary
        print("\nOCR Model Summary:")
        model.summary()
        
        # Create callbacks
        callbacks = self.create_callbacks('ocr_model', monitor='val_loss')
        
        # Train model
        print(f"\nStarting training for {epochs} epochs...")
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        final_model_path = os.path.join(self.model_save_dir, 'ocr_model_final.h5')
        model.save(final_model_path)
        print(f"Final model saved to: {final_model_path}")
        
        # Save training history
        self.training_history['ocr'] = history.history
        self._save_training_history()
        
        return model, history
    
    def _save_training_history(self):
        """Save training history to JSON file."""
        history_path = os.path.join(self.log_dir, 'training_history.json')
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_history = {}
        for model_name, history in self.training_history.items():
            serializable_history[model_name] = {}
            for key, values in history.items():
                if isinstance(values, np.ndarray):
                    serializable_history[model_name][key] = values.tolist()
                elif isinstance(values, list):
                    serializable_history[model_name][key] = values
                else:
                    serializable_history[model_name][key] = str(values)
        
        with open(history_path, 'w') as f:
            json.dump(serializable_history, f, indent=2)
        
        print(f"Training history saved to: {history_path}")


def main():
    """Main training function with CLI interface."""
    parser = argparse.ArgumentParser(description='Train ALPR models')
    
    # General arguments
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing training data')
    parser.add_argument('--model_save_dir', type=str, default='models',
                       help='Directory to save trained models')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Directory for training logs')
    
    # Model selection
    parser.add_argument('--model', type=str, choices=['detection', 'ocr', 'both'],
                       default='both', help='Which model to train')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    
    # Model-specific parameters
    parser.add_argument('--detection_img_size', type=int, nargs=2, default=[640, 640],
                       help='Input image size for detection model (width height)')
    parser.add_argument('--ocr_img_size', type=int, nargs=2, default=[128, 32],
                       help='Input image size for OCR model (width height)')
    
    # Resume training
    parser.add_argument('--resume_detection', type=str, default=None,
                       help='Path to detection model to resume training from')
    parser.add_argument('--resume_ocr', type=str, default=None,
                       help='Path to OCR model to resume training from')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ALPRTrainer(
        data_dir=args.data_dir,
        model_save_dir=args.model_save_dir,
        log_dir=args.log_dir
    )
    
    # Train models based on selection
    if args.model in ['detection', 'both']:
        print("=" * 50)
        print("TRAINING DETECTION MODEL")
        print("=" * 50)
        
        detection_model, detection_history = trainer.train_detection_model(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            img_size=tuple(args.detection_img_size),
            resume_from=args.resume_detection
        )
    
    if args.model in ['ocr', 'both']:
        print("=" * 50)
        print("TRAINING OCR MODEL")
        print("=" * 50)
        
        ocr_model, ocr_history = trainer.train_ocr_model(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            img_size=tuple(args.ocr_img_size),
            resume_from=args.resume_ocr
        )
    
    print("Training completed!")


if __name__ == '__main__':
    main()
