"""
Model building module for ALPR project.
Contains detection and OCR model architectures.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, Optional


class YOLODetectionModel:
    """YOLO-based license plate detection model."""
    
    def __init__(self, 
                 input_shape: Tuple[int, int, int] = (640, 640, 3),
                 num_classes: int = 1,
                 num_anchors: int = 3):
        """
        Initialize YOLO detection model.
        
        Args:
            input_shape: Input image shape (height, width, channels)
            num_classes: Number of object classes (1 for license plates)
            num_anchors: Number of anchor boxes per grid cell
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
    def create_backbone(self, input_tensor):
        """Create backbone network (modified MobileNetV2)."""
        # Use MobileNetV2 as backbone
        backbone = tf.keras.applications.MobileNetV2(
            input_tensor=input_tensor,
            weights='imagenet',
            include_top=False,
            alpha=1.0
        )
        
        # Extract features from different layers for FPN
        c3 = backbone.get_layer('block_6_expand_relu').output  # 80x80
        c4 = backbone.get_layer('block_13_expand_relu').output  # 40x40
        c5 = backbone.get_layer('out_relu').output  # 20x20
        
        return c3, c4, c5
    
    def create_fpn(self, c3, c4, c5):
        """Create Feature Pyramid Network."""
        # Top-down pathway
        p5 = layers.Conv2D(256, 1, padding='same', name='p5_conv')(c5)
        
        p4 = layers.Add(name='p4_add')([
            layers.UpSampling2D(2, name='p5_upsampled')(p5),
            layers.Conv2D(256, 1, padding='same', name='p4_conv')(c4)
        ])
        
        p3 = layers.Add(name='p3_add')([
            layers.UpSampling2D(2, name='p4_upsampled')(p4),
            layers.Conv2D(256, 1, padding='same', name='p3_conv')(c3)
        ])
        
        # Apply 3x3 convolution to reduce aliasing
        p3 = layers.Conv2D(256, 3, padding='same', name='p3_out')(p3)
        p4 = layers.Conv2D(256, 3, padding='same', name='p4_out')(p4)
        p5 = layers.Conv2D(256, 3, padding='same', name='p5_out')(p5)
        
        return p3, p4, p5
    
    def create_detection_head(self, feature_map, name_prefix):
        """Create detection head for a feature map."""
        # Classification head
        cls_conv1 = layers.Conv2D(256, 3, padding='same', activation='relu', 
                                 name=f'{name_prefix}_cls_conv1')(feature_map)
        cls_conv2 = layers.Conv2D(256, 3, padding='same', activation='relu',
                                 name=f'{name_prefix}_cls_conv2')(cls_conv1)
        cls_output = layers.Conv2D(self.num_anchors * self.num_classes, 3, 
                                  padding='same', name=f'{name_prefix}_cls_output')(cls_conv2)
        
        # Regression head
        reg_conv1 = layers.Conv2D(256, 3, padding='same', activation='relu',
                                 name=f'{name_prefix}_reg_conv1')(feature_map)
        reg_conv2 = layers.Conv2D(256, 3, padding='same', activation='relu',
                                 name=f'{name_prefix}_reg_conv2')(reg_conv1)
        reg_output = layers.Conv2D(self.num_anchors * 4, 3, padding='same',
                                  name=f'{name_prefix}_reg_output')(reg_conv2)
        
        # Objectness head
        obj_conv1 = layers.Conv2D(256, 3, padding='same', activation='relu',
                                 name=f'{name_prefix}_obj_conv1')(feature_map)
        obj_conv2 = layers.Conv2D(256, 3, padding='same', activation='relu',
                                 name=f'{name_prefix}_obj_conv2')(obj_conv1)
        obj_output = layers.Conv2D(self.num_anchors, 3, padding='same',
                                  name=f'{name_prefix}_obj_output')(obj_conv2)
        
        return cls_output, reg_output, obj_output
    
    def build_model(self) -> keras.Model:
        """Build complete YOLO detection model."""
        inputs = layers.Input(shape=self.input_shape, name='image_input')
        
        # Backbone
        c3, c4, c5 = self.create_backbone(inputs)
        
        # FPN
        p3, p4, p5 = self.create_fpn(c3, c4, c5)
        
        # Detection heads
        p3_cls, p3_reg, p3_obj = self.create_detection_head(p3, 'p3')
        p4_cls, p4_reg, p4_obj = self.create_detection_head(p4, 'p4')
        p5_cls, p5_reg, p5_obj = self.create_detection_head(p5, 'p5')
        
        # Combine outputs
        outputs = {
            'p3_classification': p3_cls,
            'p3_regression': p3_reg,
            'p3_objectness': p3_obj,
            'p4_classification': p4_cls,
            'p4_regression': p4_reg,
            'p4_objectness': p4_obj,
            'p5_classification': p5_cls,
            'p5_regression': p5_reg,
            'p5_objectness': p5_obj,
        }
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='yolo_detection')
        return model


class CRNNOCRModel:
    """CRNN model for license plate text recognition."""
    
    def __init__(self, 
                 input_shape: Tuple[int, int, int] = (32, 128, 1),
                 num_classes: int = 37,  # 36 chars + blank
                 rnn_units: int = 128):
        """
        Initialize CRNN OCR model.
        
        Args:
            input_shape: Input image shape (height, width, channels)
            num_classes: Number of character classes (including blank for CTC)
            rnn_units: Number of RNN units
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.rnn_units = rnn_units
        
    def create_cnn_backbone(self, inputs):
        """Create CNN backbone for feature extraction."""
        # First conv block
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='conv1')(inputs)
        x = layers.MaxPooling2D((2, 2), name='pool1')(x)
        
        # Second conv block
        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='conv2')(x)
        x = layers.MaxPooling2D((2, 2), name='pool2')(x)
        
        # Third conv block
        x = layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='conv3')(x)
        x = layers.BatchNormalization(name='bn3')(x)
        
        # Fourth conv block
        x = layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='conv4')(x)
        x = layers.MaxPooling2D((2, 1), name='pool4')(x)  # Only pool height
        
        # Fifth conv block
        x = layers.Conv2D(512, (3, 3), padding='same', activation='relu', name='conv5')(x)
        x = layers.BatchNormalization(name='bn5')(x)
        
        # Sixth conv block
        x = layers.Conv2D(512, (3, 3), padding='same', activation='relu', name='conv6')(x)
        x = layers.MaxPooling2D((2, 1), name='pool6')(x)  # Only pool height
        
        # Seventh conv block (reduce to 1D)
        x = layers.Conv2D(512, (2, 2), activation='relu', name='conv7')(x)
        
        return x
    
    def create_rnn_layers(self, cnn_output):
        """Create RNN layers for sequence modeling."""
        # Reshape CNN output for RNN input
        # CNN output shape: (batch, height, width, channels)
        # RNN input shape: (batch, time_steps, features)
        
        # Get the shape
        shape = tf.shape(cnn_output)
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]
        channels = shape[3]
        
        # Reshape: (batch, width, height * channels)
        x = tf.transpose(cnn_output, [0, 2, 1, 3])  # (batch, width, height, channels)
        x = tf.reshape(x, [batch_size, width, height * channels])
        
        # Bidirectional LSTM layers
        x = layers.Bidirectional(
            layers.LSTM(self.rnn_units, return_sequences=True, dropout=0.25),
            name='bilstm1'
        )(x)
        
        x = layers.Bidirectional(
            layers.LSTM(self.rnn_units, return_sequences=True, dropout=0.25),
            name='bilstm2'
        )(x)
        
        return x
    
    def build_model(self) -> keras.Model:
        """Build complete CRNN OCR model."""
        inputs = layers.Input(shape=self.input_shape, name='image_input')
        
        # CNN backbone
        cnn_output = self.create_cnn_backbone(inputs)
        
        # RNN layers
        rnn_output = self.create_rnn_layers(cnn_output)
        
        # Dense layer for character prediction
        dense_output = layers.Dense(self.num_classes, activation='softmax', name='dense_output')(rnn_output)
        
        model = keras.Model(inputs=inputs, outputs=dense_output, name='crnn_ocr')
        return model


class ALPRModelBuilder:
    """Main class for building ALPR models."""
    
    def __init__(self):
        self.detection_model = None
        self.ocr_model = None
        
    def build_detection_model(self, 
                            input_shape: Tuple[int, int, int] = (640, 640, 3),
                            num_classes: int = 1) -> keras.Model:
        """Build detection model."""
        builder = YOLODetectionModel(input_shape, num_classes)
        self.detection_model = builder.build_model()
        return self.detection_model
    
    def build_ocr_model(self, 
                       input_shape: Tuple[int, int, int] = (32, 128, 1),
                       num_classes: int = 37) -> keras.Model:
        """Build OCR model."""
        builder = CRNNOCRModel(input_shape, num_classes)
        self.ocr_model = builder.build_model()
        return self.ocr_model
    
    def compile_detection_model(self, 
                              learning_rate: float = 0.001,
                              **kwargs):
        """Compile detection model with appropriate loss functions."""
        if self.detection_model is None:
            raise ValueError("Detection model not built yet. Call build_detection_model() first.")
            
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Define loss functions for different outputs
        losses = {
            'p3_classification': 'sparse_categorical_crossentropy',
            'p3_regression': 'mse',
            'p3_objectness': 'binary_crossentropy',
            'p4_classification': 'sparse_categorical_crossentropy',
            'p4_regression': 'mse',
            'p4_objectness': 'binary_crossentropy',
            'p5_classification': 'sparse_categorical_crossentropy',
            'p5_regression': 'mse',
            'p5_objectness': 'binary_crossentropy',
        }
        
        # Loss weights
        loss_weights = {
            'p3_classification': 1.0,
            'p3_regression': 1.0,
            'p3_objectness': 1.0,
            'p4_classification': 1.0,
            'p4_regression': 1.0,
            'p4_objectness': 1.0,
            'p5_classification': 1.0,
            'p5_regression': 1.0,
            'p5_objectness': 1.0,
        }
        
        self.detection_model.compile(
            optimizer=optimizer,
            loss=losses,
            loss_weights=loss_weights,
            metrics=['accuracy']
        )
    
    def compile_ocr_model(self, 
                         learning_rate: float = 0.001,
                         **kwargs):
        """Compile OCR model with CTC loss."""
        if self.ocr_model is None:
            raise ValueError("OCR model not built yet. Call build_ocr_model() first.")
            
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        # CTC loss function
        def ctc_loss(y_true, y_pred):
            # y_true shape: (batch_size, max_length)
            # y_pred shape: (batch_size, time_steps, num_classes)
            
            batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
            input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
            label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
            
            input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
            label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
            
            loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
            return loss
        
        self.ocr_model.compile(
            optimizer=optimizer,
            loss=ctc_loss,
            metrics=['accuracy']
        )
