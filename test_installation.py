"""
Installation test script for ALPR project.
Verifies that all dependencies are installed correctly and modules can be imported.
"""

import sys
import importlib
import traceback
from typing import List, Tuple


def test_import(module_name: str, package_name: str = None) -> Tuple[bool, str]:
    """
    Test if a module can be imported.
    
    Args:
        module_name: Name of the module to import
        package_name: Display name for the package (optional)
        
    Returns:
        Tuple of (success, error_message)
    """
    try:
        importlib.import_module(module_name)
        return True, ""
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


def test_tensorflow_gpu():
    """Test TensorFlow GPU availability."""
    try:
        import tensorflow as tf
        
        print(f"TensorFlow version: {tf.__version__}")
        
        # Check GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"GPU devices found: {len(gpus)}")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu}")
            
            # Test GPU memory growth
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("GPU memory growth enabled successfully")
            except Exception as e:
                print(f"Warning: Could not enable GPU memory growth: {e}")
        else:
            print("No GPU devices found - using CPU")
        
        return True
    except Exception as e:
        print(f"TensorFlow GPU test failed: {e}")
        return False


def test_opencv():
    """Test OpenCV functionality."""
    try:
        import cv2
        import numpy as np
        
        print(f"OpenCV version: {cv2.__version__}")
        
        # Test basic OpenCV operations
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        resized = cv2.resize(test_image, (50, 50))
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        
        print("OpenCV basic operations test passed")
        return True
    except Exception as e:
        print(f"OpenCV test failed: {e}")
        return False


def test_project_modules():
    """Test project-specific modules."""
    project_modules = [
        ('src.data_loader', 'Data Loader'),
        ('src.model_builder', 'Model Builder'),
        ('src.train', 'Training Module'),
        ('src.evaluate', 'Evaluation Module'),
        ('src.predict', 'Prediction Module'),
        ('src.config_utils', 'Configuration Utils')
    ]
    
    print("\nTesting project modules:")
    all_passed = True
    
    for module_name, display_name in project_modules:
        success, error = test_import(module_name, display_name)
        if success:
            print(f"  ✓ {display_name}")
        else:
            print(f"  ✗ {display_name}: {error}")
            all_passed = False
    
    return all_passed


def test_project_classes():
    """Test instantiation of main project classes."""
    try:
        print("\nTesting project class instantiation:")
        
        # Test data loader
        from src.data_loader import ALPRDataLoader
        data_loader = ALPRDataLoader('.')
        print("  ✓ ALPRDataLoader")
        
        # Test model builder
        from src.model_builder import ALPRModelBuilder
        model_builder = ALPRModelBuilder()
        print("  ✓ ALPRModelBuilder")
        
        # Test config utils
        from src.config_utils import ALPRConfig
        # Don't load actual config file in test
        print("  ✓ ALPRConfig")
        
        return True
    except Exception as e:
        print(f"  ✗ Class instantiation failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all installation tests."""
    print("🚗 ALPR Project - Installation Test")
    print("=" * 50)
    
    # Required packages
    required_packages = [
        ('tensorflow', 'TensorFlow'),
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('matplotlib', 'Matplotlib'),
        ('sklearn', 'Scikit-learn'),
        ('PIL', 'Pillow'),
        ('pandas', 'Pandas'),
        ('tqdm', 'TQDM'),
        ('albumentations', 'Albumentations'),
        ('seaborn', 'Seaborn'),
        ('yaml', 'PyYAML'),
        ('click', 'Click')
    ]
    
    print("Testing required packages:")
    failed_packages = []
    
    for module_name, package_name in required_packages:
        success, error = test_import(module_name, package_name)
        if success:
            print(f"  ✓ {package_name}")
        else:
            print(f"  ✗ {package_name}: {error}")
            failed_packages.append(package_name)
    
    # Test TensorFlow GPU
    print("\nTesting TensorFlow GPU support:")
    tf_gpu_success = test_tensorflow_gpu()
    
    # Test OpenCV
    print("\nTesting OpenCV functionality:")
    cv_success = test_opencv()
    
    # Test project modules
    modules_success = test_project_modules()
    
    # Test project classes
    classes_success = test_project_classes()
    
    # Test configuration
    print("\nTesting configuration:")
    try:
        import os
        if os.path.exists('config.yaml'):
            from src.config_utils import load_config
            config = load_config()
            config.validate_config()
            print("  ✓ Configuration file loaded and validated")
            config_success = True
        else:
            print("  ! Configuration file not found (this is optional)")
            config_success = True
    except Exception as e:
        print(f"  ✗ Configuration test failed: {e}")
        config_success = False
    
    # Summary
    print("\n" + "=" * 50)
    print("INSTALLATION TEST SUMMARY")
    print("=" * 50)
    
    if failed_packages:
        print(f"❌ Failed packages ({len(failed_packages)}):")
        for package in failed_packages:
            print(f"   - {package}")
        print("\nTo install missing packages, run:")
        print("   pip install -r requirements.txt")
    else:
        print("✅ All required packages installed successfully")
    
    print(f"\n📊 Test Results:")
    print(f"   TensorFlow GPU: {'✅' if tf_gpu_success else '❌'}")
    print(f"   OpenCV: {'✅' if cv_success else '❌'}")
    print(f"   Project Modules: {'✅' if modules_success else '❌'}")
    print(f"   Class Instantiation: {'✅' if classes_success else '❌'}")
    print(f"   Configuration: {'✅' if config_success else '❌'}")
    
    overall_success = (
        len(failed_packages) == 0 and
        tf_gpu_success and
        cv_success and
        modules_success and
        classes_success and
        config_success
    )
    
    if overall_success:
        print("\n🎉 Installation test PASSED! Your ALPR project is ready to use.")
        print("\nNext steps:")
        print("1. Prepare your dataset in the 'data' directory")
        print("2. Run: python src/train.py --data_dir data --model both")
        print("3. For help: python example_usage.py")
    else:
        print("\n⚠️  Installation test FAILED. Please fix the issues above.")
        print("\nCommon solutions:")
        print("- Install missing packages: pip install -r requirements.txt")
        print("- Check Python version (3.8+ recommended)")
        print("- Verify CUDA installation for GPU support")
    
    return overall_success


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
