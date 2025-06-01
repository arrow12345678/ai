#!/usr/bin/env python3
"""
Test script to verify TensorFlow imports are working correctly.
"""

try:
    import tensorflow as tf
    print(f"✅ TensorFlow imported successfully: {tf.__version__}")
    
    from tensorflow import keras
    print(f"✅ Keras imported successfully: {keras.__version__}")
    
    from tensorflow.keras import layers
    print("✅ Keras layers imported successfully")
    
    # Test creating a simple model
    model = keras.Sequential([
        layers.Dense(10, activation='relu', input_shape=(5,)),
        layers.Dense(1, activation='sigmoid')
    ])
    print("✅ Simple model created successfully")
    
    print("\n🎉 All TensorFlow imports are working correctly!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
