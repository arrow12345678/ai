#!/usr/bin/env python3
"""
Simple test to check TensorFlow imports
"""

# Test basic TensorFlow import
try:
    import tensorflow as tf
    print(f"✅ TensorFlow imported: {tf.__version__}")
except ImportError as e:
    print(f"❌ TensorFlow import failed: {e}")
    exit(1)

# Test Keras import from TensorFlow
try:
    import tensorflow.keras as keras
    print(f"✅ TensorFlow.Keras imported: {keras.__version__}")
except ImportError as e:
    print(f"❌ TensorFlow.Keras import failed: {e}")
    exit(1)

# Test layers import
try:
    from tensorflow.keras import layers
    print("✅ TensorFlow.Keras.layers imported successfully")
except ImportError as e:
    print(f"❌ TensorFlow.Keras.layers import failed: {e}")
    exit(1)

print("🎉 All TensorFlow imports working correctly!")
