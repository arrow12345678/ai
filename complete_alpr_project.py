#!/usr/bin/env python3
"""
مشروع التعرف على لوحات السيارات - تطبيق عملي كامل
Automatic License Plate Recognition (ALPR) - Complete Implementation

المراحل:
1. تحضير قاعدة البيانات
2. تقسيم البيانات (80% train, 10% val, 10% test)
3. بناء نموذج CNN للكشف والتعرف
4. تدريب النموذج
5. تقييم الأداء
6. عرض النتائج
"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# إعداد المسارات
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# إنشاء المجلدات المطلوبة
for dir_path in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    dir_path.mkdir(exist_ok=True)

class ALPRDatasetManager:
    """إدارة قاعدة بيانات لوحات السيارات"""
    
    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = Path(data_dir)
        self.setup_directories()
        
    def setup_directories(self):
        """إنشاء هيكل المجلدات"""
        dirs = [
            "raw",           # البيانات الخام
            "processed",     # البيانات المعالجة
            "train/images", "train/labels",
            "val/images", "val/labels", 
            "test/images", "test/labels"
        ]
        
        for dir_name in dirs:
            (self.data_dir / dir_name).mkdir(parents=True, exist_ok=True)
            
        print("✅ تم إنشاء هيكل المجلدات:")
        for dir_name in dirs:
            print(f"   📁 {self.data_dir / dir_name}")
    
    def download_sample_dataset(self):
        """تحميل عينة من البيانات للتجربة"""
        print("📥 تحميل عينة من البيانات...")
        
        # إنشاء بيانات تجريبية
        sample_data = {
            "images": [],
            "plates": [],
            "labels": []
        }
        
        # محاكاة بيانات لوحات السيارات
        plate_numbers = [
            "ABC123", "XYZ789", "DEF456", "GHI012", "JKL345",
            "MNO678", "PQR901", "STU234", "VWX567", "YZA890"
        ]
        
        for i, plate in enumerate(plate_numbers):
            # إنشاء صورة تجريبية
            img = np.random.randint(0, 255, (100, 300, 3), dtype=np.uint8)
            
            # إضافة نص اللوحة على الصورة
            cv2.putText(img, plate, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (255, 255, 255), 2)
            
            sample_data["images"].append(img)
            sample_data["plates"].append(plate)
            sample_data["labels"].append(i)
        
        # حفظ البيانات
        raw_dir = self.data_dir / "raw"
        for i, (img, plate, label) in enumerate(zip(
            sample_data["images"], 
            sample_data["plates"], 
            sample_data["labels"]
        )):
            # حفظ الصورة
            img_path = raw_dir / f"plate_{i:03d}.jpg"
            cv2.imwrite(str(img_path), img)
            
            # حفظ التسمية
            label_path = raw_dir / f"plate_{i:03d}.txt"
            with open(label_path, 'w') as f:
                f.write(f"{plate}\n")
        
        print(f"✅ تم إنشاء {len(sample_data['images'])} صورة تجريبية")
        return sample_data
    
    def split_dataset(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        """تقسيم قاعدة البيانات"""
        print("📊 تقسيم قاعدة البيانات...")
        
        # التحقق من النسب
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "مجموع النسب يجب أن يساوي 1.0"
        
        # جلب جميع الصور
        raw_dir = self.data_dir / "raw"
        image_files = list(raw_dir.glob("*.jpg"))
        
        if not image_files:
            print("⚠️ لا توجد صور في المجلد الخام. سيتم إنشاء بيانات تجريبية...")
            self.download_sample_dataset()
            image_files = list(raw_dir.glob("*.jpg"))
        
        print(f"📁 عدد الصور الإجمالي: {len(image_files)}")
        
        # تقسيم البيانات
        train_files, temp_files = train_test_split(
            image_files, train_size=train_ratio, random_state=42
        )
        
        val_files, test_files = train_test_split(
            temp_files, 
            train_size=val_ratio/(val_ratio + test_ratio), 
            random_state=42
        )
        
        # نسخ الملفات إلى المجلدات المناسبة
        splits = {
            'train': train_files,
            'val': val_files, 
            'test': test_files
        }
        
        for split_name, files in splits.items():
            print(f"📂 {split_name}: {len(files)} صورة")
            
            for img_file in files:
                # نسخ الصورة
                dst_img = self.data_dir / split_name / "images" / img_file.name
                import shutil
                shutil.copy2(img_file, dst_img)
                
                # نسخ ملف التسمية
                label_file = img_file.with_suffix('.txt')
                if label_file.exists():
                    dst_label = self.data_dir / split_name / "labels" / label_file.name
                    shutil.copy2(label_file, dst_label)
        
        print("✅ تم تقسيم البيانات بنجاح!")
        return splits
    
    def visualize_dataset(self, split='train', num_samples=6):
        """عرض عينات من قاعدة البيانات"""
        print(f"🖼️ عرض عينات من {split} dataset...")
        
        images_dir = self.data_dir / split / "images"
        labels_dir = self.data_dir / split / "labels"
        
        image_files = list(images_dir.glob("*.jpg"))[:num_samples]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, img_file in enumerate(image_files):
            if i >= num_samples:
                break
                
            # قراءة الصورة
            img = cv2.imread(str(img_file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # قراءة التسمية
            label_file = labels_dir / img_file.with_suffix('.txt').name
            label = "Unknown"
            if label_file.exists():
                with open(label_file, 'r') as f:
                    label = f.read().strip()
            
            # عرض الصورة
            axes[i].imshow(img)
            axes[i].set_title(f"اللوحة: {label}", fontsize=12)
            axes[i].axis('off')
        
        # إخفاء المحاور الفارغة
        for i in range(len(image_files), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"{split}_samples.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ تم حفظ العينات في: {RESULTS_DIR / f'{split}_samples.png'}")

class ALPRModel:
    """نموذج التعرف على لوحات السيارات"""
    
    def __init__(self, input_shape=(128, 384, 3), num_classes=36):
        self.input_shape = input_shape
        self.num_classes = num_classes  # 26 حرف + 10 أرقام
        self.model = None
        self.history = None
        
    def build_cnn_model(self):
        """بناء نموذج CNN للتعرف على الأحرف والأرقام"""
        print("🏗️ بناء نموذج CNN...")
        
        model = keras.Sequential([
            # طبقة الدخل
            layers.Input(shape=self.input_shape),
            
            # Block 1
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 2  
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 4
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Global Average Pooling
            layers.GlobalAveragePooling2D(),
            
            # Dense layers
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        
        # تجميع النموذج
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ تم بناء النموذج بنجاح!")
        print(f"📊 عدد المعاملات: {self.model.count_params():,}")
        
        return self.model
    
    def train_model(self, train_data, val_data, epochs=50):
        """تدريب النموذج"""
        print("🚀 بدء تدريب النموذج...")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                MODELS_DIR / "best_model.h5",
                monitor='val_accuracy',
                save_best_only=True
            )
        ]
        
        # تدريب النموذج
        self.history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ تم الانتهاء من التدريب!")
        return self.history
    
    def evaluate_model(self, test_data):
        """تقييم أداء النموذج"""
        print("📊 تقييم أداء النموذج...")
        
        # تقييم على بيانات الاختبار
        test_loss, test_accuracy = self.model.evaluate(test_data, verbose=0)
        
        print(f"📈 دقة النموذج على بيانات الاختبار: {test_accuracy:.4f}")
        print(f"📉 خسارة النموذج على بيانات الاختبار: {test_loss:.4f}")
        
        return test_loss, test_accuracy
    
    def plot_training_history(self):
        """رسم منحنيات التدريب"""
        if self.history is None:
            print("⚠️ لا توجد بيانات تدريب لعرضها")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # منحنى الدقة
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # منحنى الخسارة
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "training_history.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ تم حفظ منحنيات التدريب في: {RESULTS_DIR / 'training_history.png'}")

def main():
    """الدالة الرئيسية لتشغيل المشروع"""
    print("🚗 مشروع التعرف على لوحات السيارات")
    print("=" * 50)
    
    # 1. إعداد قاعدة البيانات
    print("\n1️⃣ إعداد قاعدة البيانات...")
    dataset_manager = ALPRDatasetManager()
    
    # 2. تقسيم البيانات
    print("\n2️⃣ تقسيم البيانات...")
    splits = dataset_manager.split_dataset(
        train_ratio=0.8,
        val_ratio=0.1, 
        test_ratio=0.1
    )
    
    # 3. عرض عينات من البيانات
    print("\n3️⃣ عرض عينات من البيانات...")
    dataset_manager.visualize_dataset('train', num_samples=6)
    
    # 4. بناء النموذج
    print("\n4️⃣ بناء النموذج...")
    model = ALPRModel()
    cnn_model = model.build_cnn_model()
    
    # عرض هيكل النموذج
    print("\n📋 هيكل النموذج:")
    cnn_model.summary()
    
    print("\n✅ تم إعداد المشروع بنجاح!")
    print("\n📝 الخطوات التالية:")
    print("   1. إضافة بيانات حقيقية إلى مجلد data/raw/")
    print("   2. تشغيل التدريب باستخدام model.train_model()")
    print("   3. تقييم النموذج باستخدام model.evaluate_model()")
    
    return dataset_manager, model

if __name__ == "__main__":
    dataset_manager, model = main()
