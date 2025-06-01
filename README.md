# 🚗 Automatic License Plate Recognition (ALPR) Project

A comprehensive Python-based Automatic License Plate Recognition system using TensorFlow/Keras, OpenCV, and modern deep learning techniques.

## 📋 Project Overview

This project implements an end-to-end ALPR system that can:
- **Detect** license plates in vehicle images using YOLO-based object detection
- **Recognize** text from detected license plates using CRNN with CTC loss
- **Evaluate** model performance with comprehensive metrics
- **Predict** on new images with visualization capabilities

## 🏗️ Project Structure

```
alpr_project/
├── data/                    # Dataset directory
│   ├── train/
│   │   ├── images/         # Training images
│   │   └── labels/         # Training labels (YOLO/PASCAL VOC format)
│   ├── val/
│   │   ├── images/         # Validation images
│   │   └── labels/         # Validation labels
│   └── test/
│       ├── images/         # Test images
│       └── labels/         # Test labels
├── models/                  # Saved trained models
├── logs/                   # Training logs and TensorBoard files
├── src/                    # Source code
│   ├── data_loader.py      # Data loading and preprocessing
│   ├── model_builder.py    # Model architectures (YOLO + CRNN)
│   ├── train.py           # Training pipeline
│   ├── evaluate.py        # Model evaluation
│   └── predict.py         # Inference pipeline
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd alpr_project

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

#### Option A: Use Pre-split Data
Place your data in the following structure:
```
data/
├── train/images/ + train/labels/
├── val/images/ + val/labels/
└── test/images/ + test/labels/
```

#### Option B: Auto-split Dataset
If you have unsplit data:
```python
from src.data_loader import ALPRDataLoader

loader = ALPRDataLoader('data')
loader.split_dataset('path/to/your/raw/data')
```

#### Supported Label Formats:
- **YOLO format (.txt)**: `class_id x_center y_center width height [text]`
- **PASCAL VOC format (.xml)**: Standard XML annotation format

### 3. Training

#### Train Both Models (Recommended)
```bash
python src/train.py \
    --data_dir data \
    --model both \
    --epochs 100 \
    --batch_size 16 \
    --learning_rate 0.001
```

#### Train Detection Model Only
```bash
python src/train.py \
    --data_dir data \
    --model detection \
    --epochs 100 \
    --detection_img_size 640 640
```

#### Train OCR Model Only
```bash
python src/train.py \
    --data_dir data \
    --model ocr \
    --epochs 100 \
    --ocr_img_size 128 32
```

#### Resume Training
```bash
python src/train.py \
    --data_dir data \
    --model both \
    --resume_detection models/detection_model_best.h5 \
    --resume_ocr models/ocr_model_best.h5
```

### 4. Evaluation

```bash
python src/evaluate.py \
    --data_dir data \
    --detection_model models/detection_model_best.h5 \
    --ocr_model models/ocr_model_best.h5 \
    --save_results
```

### 5. Prediction/Inference

#### Single Image
```bash
python src/predict.py \
    --detection_model models/detection_model_best.h5 \
    --ocr_model models/ocr_model_best.h5 \
    --input path/to/image.jpg \
    --visualize \
    --save_json
```

#### Batch Processing
```bash
python src/predict.py \
    --detection_model models/detection_model_best.h5 \
    --ocr_model models/ocr_model_best.h5 \
    --input path/to/image/directory/ \
    --output_dir results \
    --visualize
```

## 🧠 Model Architectures

### Detection Model (YOLO-based)
- **Backbone**: MobileNetV2 (pre-trained on ImageNet)
- **Neck**: Feature Pyramid Network (FPN)
- **Head**: Multi-scale detection heads for classification, regression, and objectness
- **Input**: 640×640×3 RGB images
- **Output**: Bounding boxes with confidence scores

### OCR Model (CRNN)
- **CNN Backbone**: Custom convolutional layers for feature extraction
- **RNN**: Bidirectional LSTM layers for sequence modeling
- **CTC Loss**: Connectionist Temporal Classification for sequence alignment
- **Input**: 128×32×1 grayscale license plate images
- **Output**: Character sequences (0-9, A-Z)

## 📊 Evaluation Metrics

### Detection Metrics
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **mAP**: Mean Average Precision
- **IoU**: Intersection over Union

### OCR Metrics
- **Character Accuracy**: Percentage of correctly recognized characters
- **Word Accuracy**: Percentage of completely correct license plates
- **Character Error Rate (CER)**: Edit distance / Total characters
- **Average Edit Distance**: Levenshtein distance between predictions and ground truth

## 🔧 Configuration Options

### Data Augmentation
The system includes comprehensive data augmentation:
- Horizontal flipping
- Random brightness/contrast adjustment
- Gaussian noise
- Motion blur
- Small rotations

### Training Parameters
- **Learning Rate**: Adaptive with ReduceLROnPlateau
- **Batch Size**: Configurable (default: 16 for detection, 32 for OCR)
- **Epochs**: Configurable with early stopping
- **Optimizers**: Adam optimizer with configurable learning rates

## 📈 Monitoring Training

### TensorBoard
```bash
tensorboard --logdir logs
```

### Training Logs
- CSV logs saved in `logs/` directory
- Model checkpoints saved in `models/` directory
- Training history saved as JSON

## 🛠️ Advanced Usage

### Custom Character Set
Modify the character set in `data_loader.py`:
```python
self.char_set = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'  # Default
# Add custom characters as needed
```

### Model Customization
- Modify architectures in `model_builder.py`
- Adjust input sizes, layer configurations, etc.
- Implement custom loss functions

### Data Pipeline Optimization
- Adjust batch sizes based on GPU memory
- Modify augmentation strategies
- Implement custom preprocessing steps

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce batch size
   - Use smaller input image sizes
   - Enable mixed precision training

2. **Poor Detection Performance**
   - Increase training epochs
   - Adjust learning rate
   - Check data quality and annotations

3. **Poor OCR Performance**
   - Ensure license plate images are clear
   - Adjust OCR preprocessing parameters
   - Increase OCR training data

### Performance Tips
- Use GPU acceleration for training
- Implement data pipeline optimization
- Use mixed precision training for faster training
- Monitor training with TensorBoard

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📚 References

- YOLO: You Only Look Once object detection
- CRNN: Convolutional Recurrent Neural Network
- CTC Loss: Connectionist Temporal Classification
- TensorFlow/Keras documentation
- OpenCV documentation

## 📞 Support

For questions, issues, or contributions, please:
1. Check the troubleshooting section
2. Search existing issues
3. Create a new issue with detailed information
4. Provide sample data and error logs when possible
