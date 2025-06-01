#!/usr/bin/env python3
"""
تدريب وتقييم نموذج التعرف على لوحات السيارات
ALPR Model Training and Evaluation

المراحل:
1. تحضير البيانات
2. تدريب النموذج
3. تقييم الأداء
4. حفظ النتائج
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import json
from datetime import datetime
import cv2
from tqdm import tqdm

# إضافة المجلد الحالي للمسار
sys.path.append(str(Path(__file__).parent))

from complete_alpr_project import ALPRDatasetManager, ALPRModel

class ALPRTrainer:
    """مدرب نموذج التعرف على لوحات السيارات"""
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.results_dir = Path("results")
        self.models_dir = Path("models")
        
        # إنشاء المجلدات
        self.results_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        
        # إعداد مدير البيانات
        self.dataset_manager = ALPRDatasetManager(data_dir)
        
        # إعداد النموذج
        self.model = ALPRModel()
        
        # معلومات التدريب
        self.training_info = {
            "start_time": None,
            "end_time": None,
            "epochs": 0,
            "best_accuracy": 0.0,
            "final_loss": 0.0
        }
    
    def prepare_data_generators(self, batch_size=32, img_size=(128, 384)):
        """تحضير مولدات البيانات"""
        print("📊 تحضير مولدات البيانات...")
        
        # إعداد معالجة الصور
        train_datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=5,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=False,  # لا نقلب لوحات السيارات أفقياً
            fill_mode='nearest'
        )
        
        val_test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        
        # إنشاء مولدات البيانات
        train_generator = train_datagen.flow_from_directory(
            self.data_dir / "train",
            target_size=img_size,
            batch_size=batch_size,
            class_mode='sparse',
            shuffle=True
        )
        
        val_generator = val_test_datagen.flow_from_directory(
            self.data_dir / "val",
            target_size=img_size,
            batch_size=batch_size,
            class_mode='sparse',
            shuffle=False
        )
        
        test_generator = val_test_datagen.flow_from_directory(
            self.data_dir / "test",
            target_size=img_size,
            batch_size=batch_size,
            class_mode='sparse',
            shuffle=False
        )
        
        print(f"✅ تم إعداد المولدات:")
        print(f"   📈 التدريب: {train_generator.samples} عينة")
        print(f"   📊 التحقق: {val_generator.samples} عينة")
        print(f"   🧪 الاختبار: {test_generator.samples} عينة")
        
        return train_generator, val_generator, test_generator
    
    def create_character_dataset(self):
        """إنشاء قاعدة بيانات للأحرف والأرقام"""
        print("🔤 إنشاء قاعدة بيانات الأحرف والأرقام...")
        
        # الأحرف والأرقام المدعومة
        characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        
        # إنشاء مجلدات للأحرف
        for split in ['train', 'val', 'test']:
            for char in characters:
                char_dir = self.data_dir / split / char
                char_dir.mkdir(parents=True, exist_ok=True)
        
        # قراءة البيانات الموجودة وتصنيفها
        for split in ['train', 'val', 'test']:
            images_dir = self.data_dir / split / "images"
            labels_dir = self.data_dir / split / "labels"
            
            if not images_dir.exists():
                continue
            
            for img_file in images_dir.glob("*.jpg"):
                label_file = labels_dir / img_file.with_suffix('.txt').name
                
                if label_file.exists():
                    with open(label_file, 'r') as f:
                        plate_text = f.read().strip()
                    
                    # قراءة الصورة
                    img = cv2.imread(str(img_file))
                    if img is None:
                        continue
                    
                    # تقسيم النص إلى أحرف منفردة
                    for i, char in enumerate(plate_text):
                        if char in characters:
                            # استخراج جزء من الصورة للحرف
                            h, w = img.shape[:2]
                            char_width = w // len(plate_text)
                            x1 = i * char_width
                            x2 = (i + 1) * char_width
                            
                            char_img = img[:, x1:x2]
                            char_img = cv2.resize(char_img, (64, 64))
                            
                            # حفظ صورة الحرف
                            char_path = self.data_dir / split / char / f"{img_file.stem}_{i}.jpg"
                            cv2.imwrite(str(char_path), char_img)
        
        print("✅ تم إنشاء قاعدة بيانات الأحرف")
    
    def train_model(self, epochs=50, batch_size=32):
        """تدريب النموذج"""
        print("🚀 بدء تدريب النموذج...")
        
        self.training_info["start_time"] = datetime.now()
        self.training_info["epochs"] = epochs
        
        # إنشاء قاعدة بيانات الأحرف
        self.create_character_dataset()
        
        # تحضير البيانات
        train_gen, val_gen, test_gen = self.prepare_data_generators(batch_size)
        
        # بناء النموذج
        model = self.model.build_cnn_model()
        
        # إعداد callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                self.models_dir / "best_model.h5",
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            keras.callbacks.CSVLogger(
                self.results_dir / "training_log.csv"
            )
        ]
        
        # تدريب النموذج
        history = self.model.train_model(
            train_gen, 
            val_gen, 
            epochs=epochs
        )
        
        self.training_info["end_time"] = datetime.now()
        self.training_info["best_accuracy"] = max(history.history['val_accuracy'])
        self.training_info["final_loss"] = history.history['val_loss'][-1]
        
        # حفظ النموذج النهائي
        model.save(self.models_dir / "final_model.h5")
        
        print("✅ تم الانتهاء من التدريب!")
        return history, test_gen
    
    def evaluate_model(self, test_generator):
        """تقييم شامل للنموذج"""
        print("📊 تقييم شامل للنموذج...")
        
        # تحميل أفضل نموذج
        best_model = keras.models.load_model(self.models_dir / "best_model.h5")
        
        # تقييم على بيانات الاختبار
        test_loss, test_accuracy = best_model.evaluate(test_generator, verbose=0)
        
        # التنبؤ على بيانات الاختبار
        predictions = best_model.predict(test_generator, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # الحصول على التسميات الحقيقية
        true_classes = test_generator.classes
        class_labels = list(test_generator.class_indices.keys())
        
        # تقرير التصنيف
        report = classification_report(
            true_classes, 
            predicted_classes, 
            target_names=class_labels,
            output_dict=True
        )
        
        # مصفوفة الخلط
        cm = confusion_matrix(true_classes, predicted_classes)
        
        # حفظ النتائج
        results = {
            "test_accuracy": float(test_accuracy),
            "test_loss": float(test_loss),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "class_labels": class_labels,
            "training_info": self.training_info
        }
        
        # حفظ النتائج في ملف JSON
        with open(self.results_dir / "evaluation_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"📈 دقة النموذج: {test_accuracy:.4f}")
        print(f"📉 خسارة النموذج: {test_loss:.4f}")
        
        return results
    
    def plot_results(self, history, results):
        """رسم النتائج والتحليلات"""
        print("📊 إنشاء الرسوم البيانية...")
        
        # إعداد الرسم
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 15))
        
        # 1. منحنيات التدريب
        ax1 = plt.subplot(2, 3, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        plt.title('Model Accuracy', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        ax2 = plt.subplot(2, 3, 2)
        plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
        plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        plt.title('Model Loss', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. مصفوفة الخلط
        ax3 = plt.subplot(2, 3, 3)
        cm = np.array(results['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=results['class_labels'][:10],  # أول 10 فئات فقط
                   yticklabels=results['class_labels'][:10])
        plt.title('Confusion Matrix (Top 10 Classes)', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # 3. توزيع الدقة حسب الفئة
        ax4 = plt.subplot(2, 3, 4)
        class_accuracies = []
        class_names = []
        for class_name, metrics in results['classification_report'].items():
            if isinstance(metrics, dict) and 'precision' in metrics:
                class_accuracies.append(metrics['f1-score'])
                class_names.append(class_name)
        
        if class_accuracies:
            plt.bar(range(len(class_accuracies)), class_accuracies)
            plt.title('F1-Score by Class', fontsize=14, fontweight='bold')
            plt.xlabel('Class')
            plt.ylabel('F1-Score')
            plt.xticks(range(len(class_names)), class_names, rotation=45)
            plt.grid(True, alpha=0.3)
        
        # 4. ملخص الأداء
        ax5 = plt.subplot(2, 3, 5)
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [
            results['test_accuracy'],
            results['classification_report']['macro avg']['precision'],
            results['classification_report']['macro avg']['recall'],
            results['classification_report']['macro avg']['f1-score']
        ]
        
        bars = plt.bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        plt.title('Overall Performance Metrics', fontsize=14, fontweight='bold')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        
        # إضافة قيم على الأعمدة
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.grid(True, alpha=0.3)
        
        # 5. معلومات التدريب
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        training_duration = self.training_info["end_time"] - self.training_info["start_time"]
        
        info_text = f"""
        Training Information:
        
        • Duration: {training_duration}
        • Epochs: {self.training_info['epochs']}
        • Best Validation Accuracy: {self.training_info['best_accuracy']:.4f}
        • Final Validation Loss: {self.training_info['final_loss']:.4f}
        • Test Accuracy: {results['test_accuracy']:.4f}
        • Test Loss: {results['test_loss']:.4f}
        
        Model Architecture:
        • Type: Convolutional Neural Network
        • Input Shape: (128, 384, 3)
        • Output Classes: {len(results['class_labels'])}
        """
        
        ax6.text(0.1, 0.9, info_text, transform=ax6.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "complete_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ تم حفظ التحليل الكامل في: {self.results_dir / 'complete_analysis.png'}")
    
    def generate_report(self, results):
        """إنشاء تقرير شامل"""
        print("📄 إنشاء التقرير الشامل...")
        
        report_content = f"""
# تقرير مشروع التعرف على لوحات السيارات
## Automatic License Plate Recognition (ALPR) Project Report

### معلومات المشروع
- **تاريخ التدريب**: {self.training_info['start_time'].strftime('%Y-%m-%d %H:%M:%S')}
- **مدة التدريب**: {self.training_info['end_time'] - self.training_info['start_time']}
- **عدد العصور**: {self.training_info['epochs']}

### نتائج الأداء
- **دقة الاختبار**: {results['test_accuracy']:.4f} ({results['test_accuracy']*100:.2f}%)
- **خسارة الاختبار**: {results['test_loss']:.4f}
- **أفضل دقة تحقق**: {self.training_info['best_accuracy']:.4f}

### تفاصيل التصنيف
"""
        
        # إضافة تفاصيل كل فئة
        for class_name, metrics in results['classification_report'].items():
            if isinstance(metrics, dict) and 'precision' in metrics:
                report_content += f"""
#### الفئة: {class_name}
- **الدقة (Precision)**: {metrics['precision']:.3f}
- **الاستدعاء (Recall)**: {metrics['recall']:.3f}
- **F1-Score**: {metrics['f1-score']:.3f}
- **عدد العينات**: {metrics['support']}
"""
        
        report_content += f"""

### الخلاصة
تم تدريب نموذج التعرف على لوحات السيارات بنجاح بدقة {results['test_accuracy']*100:.2f}%.
النموذج قادر على التعرف على {len(results['class_labels'])} فئة مختلفة من الأحرف والأرقام.

### الملفات المُنتجة
- `best_model.h5`: أفضل نموذج مدرب
- `final_model.h5`: النموذج النهائي
- `evaluation_results.json`: نتائج التقييم التفصيلية
- `complete_analysis.png`: التحليل البصري الشامل
- `training_log.csv`: سجل التدريب

### التوصيات
1. يمكن تحسين الأداء بإضافة المزيد من البيانات
2. تجربة تقنيات تعزيز البيانات المختلفة
3. ضبط معاملات النموذج (hyperparameter tuning)
"""
        
        # حفظ التقرير
        with open(self.results_dir / "project_report.md", 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"✅ تم حفظ التقرير في: {self.results_dir / 'project_report.md'}")

def main():
    """تشغيل التدريب والتقييم الكامل"""
    print("🚗 تدريب وتقييم نموذج التعرف على لوحات السيارات")
    print("=" * 60)
    
    # إنشاء المدرب
    trainer = ALPRTrainer()
    
    # تدريب النموذج
    history, test_gen = trainer.train_model(epochs=30, batch_size=32)
    
    # تقييم النموذج
    results = trainer.evaluate_model(test_gen)
    
    # رسم النتائج
    trainer.plot_results(history, results)
    
    # إنشاء التقرير
    trainer.generate_report(results)
    
    print("\n🎉 تم الانتهاء من المشروع بنجاح!")
    print("📁 تحقق من مجلد 'results' للاطلاع على النتائج")

if __name__ == "__main__":
    main()
