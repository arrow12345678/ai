"""
Configuration utilities for ALPR project.
Handles loading and managing configuration parameters.
"""

import yaml
import os
from typing import Dict, Any, Optional
import json


class ALPRConfig:
    """Configuration manager for ALPR project."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config_path = config_path
        self.config = self.load_config()
        
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            Configuration dictionary
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        return config
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to configuration value (e.g., 'data.batch_size')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any):
        """
        Set configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to configuration value
            value: Value to set
        """
        keys = key_path.split('.')
        config_ref = self.config
        
        # Navigate to parent dictionary
        for key in keys[:-1]:
            if key not in config_ref:
                config_ref[key] = {}
            config_ref = config_ref[key]
        
        # Set the value
        config_ref[keys[-1]] = value
    
    def save_config(self, output_path: Optional[str] = None):
        """
        Save configuration to YAML file.
        
        Args:
            output_path: Output path (defaults to original config path)
        """
        if output_path is None:
            output_path = self.config_path
            
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data-related configuration."""
        return self.get('data', {})
    
    def get_model_config(self, model_type: str = None) -> Dict[str, Any]:
        """
        Get model configuration.
        
        Args:
            model_type: Specific model type ('detection' or 'ocr')
            
        Returns:
            Model configuration
        """
        if model_type:
            return self.get(f'models.{model_type}', {})
        return self.get('models', {})
    
    def get_training_config(self, model_type: str = None) -> Dict[str, Any]:
        """
        Get training configuration.
        
        Args:
            model_type: Specific model type ('detection' or 'ocr')
            
        Returns:
            Training configuration
        """
        if model_type:
            # Get model-specific config and merge with general config
            general_config = self.get('training', {})
            specific_config = self.get(f'training.{model_type}', {})
            
            # Merge configurations (specific overrides general)
            merged_config = general_config.copy()
            merged_config.update(specific_config)
            return merged_config
        
        return self.get('training', {})
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration."""
        return self.get('evaluation', {})
    
    def get_prediction_config(self) -> Dict[str, Any]:
        """Get prediction configuration."""
        return self.get('prediction', {})
    
    def get_directories(self) -> Dict[str, str]:
        """Get directory configuration."""
        return self.get('directories', {})
    
    def create_directories(self):
        """Create all configured directories."""
        directories = self.get_directories()
        
        for dir_name, dir_path in directories.items():
            os.makedirs(dir_path, exist_ok=True)
            print(f"Created directory: {dir_path}")
    
    def get_image_processing_config(self) -> Dict[str, Any]:
        """Get image processing configuration."""
        return self.get('image_processing', {})
    
    def get_augmentation_config(self) -> Dict[str, Any]:
        """Get data augmentation configuration."""
        return self.get('image_processing.augmentation', {})
    
    def get_hardware_config(self) -> Dict[str, Any]:
        """Get hardware configuration."""
        return self.get('hardware', {})
    
    def get_visualization_config(self) -> Dict[str, Any]:
        """Get visualization configuration."""
        return self.get('visualization', {})
    
    def print_config(self, section: str = None):
        """
        Print configuration in a readable format.
        
        Args:
            section: Specific section to print (optional)
        """
        if section:
            config_to_print = self.get(section, {})
            print(f"=== {section.upper()} CONFIGURATION ===")
        else:
            config_to_print = self.config
            print("=== FULL CONFIGURATION ===")
        
        print(yaml.dump(config_to_print, default_flow_style=False, indent=2))
    
    def validate_config(self) -> bool:
        """
        Validate configuration for required fields and reasonable values.
        
        Returns:
            True if configuration is valid
        """
        errors = []
        
        # Check required sections
        required_sections = ['data', 'models', 'training', 'directories']
        for section in required_sections:
            if section not in self.config:
                errors.append(f"Missing required section: {section}")
        
        # Validate data configuration
        data_config = self.get_data_config()
        if 'data_dir' not in data_config:
            errors.append("Missing 'data.data_dir' in configuration")
        
        # Validate model configuration
        models_config = self.get_model_config()
        if 'detection' not in models_config or 'ocr' not in models_config:
            errors.append("Missing detection or OCR model configuration")
        
        # Validate training configuration
        training_config = self.get_training_config()
        if 'epochs' not in training_config:
            errors.append("Missing 'training.epochs' in configuration")
        
        # Check for reasonable values
        epochs = self.get('training.epochs', 0)
        if epochs <= 0:
            errors.append("Training epochs must be positive")
        
        batch_size = self.get('training.batch_size', 0)
        if batch_size <= 0:
            errors.append("Batch size must be positive")
        
        learning_rate = self.get('training.learning_rate', 0)
        if learning_rate <= 0 or learning_rate > 1:
            errors.append("Learning rate must be between 0 and 1")
        
        # Print errors if any
        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        print("Configuration validation passed!")
        return True
    
    def export_to_json(self, output_path: str):
        """
        Export configuration to JSON format.
        
        Args:
            output_path: Output JSON file path
        """
        with open(output_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"Configuration exported to: {output_path}")


def load_config(config_path: str = "config.yaml") -> ALPRConfig:
    """
    Load ALPR configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        ALPRConfig instance
    """
    return ALPRConfig(config_path)


def create_default_config(output_path: str = "config.yaml"):
    """
    Create a default configuration file.
    
    Args:
        output_path: Output path for configuration file
    """
    default_config = {
        'data': {
            'data_dir': 'data',
            'train_ratio': 0.8,
            'val_ratio': 0.1,
            'test_ratio': 0.1
        },
        'image_processing': {
            'detection_img_size': [640, 640],
            'ocr_img_size': [128, 32]
        },
        'models': {
            'detection': {
                'input_shape': [640, 640, 3],
                'num_classes': 1,
                'num_anchors': 3
            },
            'ocr': {
                'input_shape': [32, 128, 1],
                'num_classes': 37,
                'rnn_units': 128
            }
        },
        'training': {
            'epochs': 100,
            'batch_size': 16,
            'learning_rate': 0.001
        },
        'directories': {
            'models': 'models',
            'logs': 'logs',
            'predictions': 'predictions'
        }
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False, indent=2)
    
    print(f"Default configuration created: {output_path}")


if __name__ == '__main__':
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='ALPR Configuration Utilities')
    parser.add_argument('--create_default', action='store_true',
                       help='Create default configuration file')
    parser.add_argument('--validate', action='store_true',
                       help='Validate existing configuration')
    parser.add_argument('--print_config', type=str, default=None,
                       help='Print configuration (optionally specify section)')
    parser.add_argument('--config_path', type=str, default='config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    if args.create_default:
        create_default_config(args.config_path)
    
    if args.validate or args.print_config is not None:
        config = load_config(args.config_path)
        
        if args.validate:
            config.validate_config()
        
        if args.print_config is not None:
            if args.print_config == 'all':
                config.print_config()
            else:
                config.print_config(args.print_config)
