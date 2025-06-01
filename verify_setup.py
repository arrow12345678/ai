#!/usr/bin/env python3
"""
Verification script to ensure all imports and setup are working correctly.
"""

import sys
import os

def test_basic_imports():
    """Test basic Python imports."""
    print("🔍 Testing basic imports...")
    
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False
    
    return True

def test_tensorflow_imports():
    """Test TensorFlow and Keras imports."""
    print("\n🔍 Testing TensorFlow imports...")
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow: {tf.__version__}")
    except ImportError as e:
        print(f"❌ TensorFlow import failed: {e}")
        return False
    
    try:
        from tensorflow import keras
        print(f"✅ Keras: {keras.__version__}")
    except ImportError as e:
        print(f"❌ Keras import failed: {e}")
        return False
    
    try:
        from tensorflow.keras import layers
        print("✅ Keras layers imported successfully")
    except ImportError as e:
        print(f"❌ Keras layers import failed: {e}")
        return False
    
    return True

def test_ml_libraries():
    """Test machine learning libraries."""
    print("\n🔍 Testing ML libraries...")
    
    try:
        import sklearn
        print(f"✅ Scikit-learn: {sklearn.__version__}")
    except ImportError as e:
        print(f"❌ Scikit-learn import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✅ Pandas: {pd.__version__}")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        import matplotlib
        print(f"✅ Matplotlib: {matplotlib.__version__}")
    except ImportError as e:
        print(f"❌ Matplotlib import failed: {e}")
        return False
    
    return True

def test_project_imports():
    """Test project-specific imports."""
    print("\n🔍 Testing project imports...")
    
    # Add src to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    
    try:
        from model_builder import ALPRModelBuilder
        print("✅ ALPRModelBuilder imported successfully")
    except ImportError as e:
        print(f"❌ ALPRModelBuilder import failed: {e}")
        return False
    
    try:
        from data_loader import ALPRDataLoader
        print("✅ ALPRDataLoader imported successfully")
    except ImportError as e:
        print(f"❌ ALPRDataLoader import failed: {e}")
        return False
    
    return True

def test_model_creation():
    """Test creating models."""
    print("\n🔍 Testing model creation...")
    
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        from model_builder import ALPRModelBuilder
        
        builder = ALPRModelBuilder()
        
        # Test detection model
        detection_model = builder.build_detection_model(
            input_shape=(640, 640, 3),
            num_classes=1
        )
        print("✅ Detection model created successfully")
        
        # Test OCR model
        ocr_model = builder.build_ocr_model(
            input_shape=(32, 128, 1),
            num_classes=37
        )
        print("✅ OCR model created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def main():
    """Run all verification tests."""
    print("🚀 Starting ALPR Project Setup Verification")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_tensorflow_imports,
        test_ml_libraries,
        test_project_imports,
        test_model_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print("❌ Test failed")
        except Exception as e:
            print(f"❌ Test error: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your setup is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
