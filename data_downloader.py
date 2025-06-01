#!/usr/bin/env python3
"""
تحميل قواعد بيانات لوحات السيارات من مصادر مختلفة
License Plate Dataset Downloader

المصادر المدعومة:
1. Kaggle Datasets
2. GitHub Repositories  
3. OpenImages Dataset
4. Custom URLs
"""

import os
import requests
import zipfile
import tarfile
from pathlib import Path
import json
from tqdm import tqdm
import cv2
import numpy as np

class DatasetDownloader:
    """تحميل قواعد البيانات من مصادر مختلفة"""
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # قائمة بمصادر البيانات المتاحة
        self.available_datasets = {
            "sample_plates": {
                "description": "عينة تجريبية من لوحات السيارات",
                "size": "صغير (< 1MB)",
                "type": "تجريبي",
                "url": None
            },
            "kaggle_license_plates": {
                "description": "Kaggle License Plate Dataset",
                "size": "متوسط (~100MB)",
                "type": "Kaggle",
                "url": "https://www.kaggle.com/datasets/andrewmvd/car-plate-detection"
            },
            "github_alpr": {
                "description": "GitHub ALPR Dataset",
                "size": "كبير (~500MB)",
                "type": "GitHub",
                "url": "https://github.com/detectRecog/CCPD"
            },
            "openimages_vehicles": {
                "description": "OpenImages Vehicle Dataset",
                "size": "كبير جداً (~2GB)",
                "type": "OpenImages",
                "url": "https://storage.googleapis.com/openimages/web/index.html"
            }
        }
    
    def list_available_datasets(self):
        """عرض قائمة بقواعد البيانات المتاحة"""
        print("📊 قواعد البيانات المتاحة:")
        print("=" * 60)
        
        for i, (key, info) in enumerate(self.available_datasets.items(), 1):
            print(f"{i}. {key}")
            print(f"   📝 الوصف: {info['description']}")
            print(f"   📏 الحجم: {info['size']}")
            print(f"   🔗 المصدر: {info['type']}")
            print(f"   🌐 الرابط: {info['url'] or 'محلي'}")
            print()
    
    def create_sample_dataset(self, num_samples=50):
        """إنشاء قاعدة بيانات تجريبية"""
        print(f"🎨 إنشاء قاعدة بيانات تجريبية ({num_samples} عينة)...")
        
        # إنشاء مجلد البيانات الخام
        raw_dir = self.data_dir / "raw"
        raw_dir.mkdir(exist_ok=True)
        
        # قوائم الأحرف والأرقام
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        numbers = "0123456789"
        
        # أنماط لوحات مختلفة
        patterns = [
            lambda: f"{np.random.choice(list(letters), 3, replace=True).tolist()}{np.random.choice(list(numbers), 3, replace=True).tolist()}",
            lambda: f"{np.random.choice(list(numbers), 3, replace=True).tolist()}{np.random.choice(list(letters), 3, replace=True).tolist()}",
            lambda: f"{np.random.choice(list(letters), 2, replace=True).tolist()}{np.random.choice(list(numbers), 4, replace=True).tolist()}",
        ]
        
        for i in tqdm(range(num_samples), desc="إنشاء العينات"):
            # اختيار نمط عشوائي
            pattern = np.random.choice(patterns)
            plate_chars = pattern()
            plate_text = ''.join(plate_chars)
            
            # إنشاء صورة اللوحة
            img = self.generate_plate_image(plate_text)
            
            # حفظ الصورة
            img_path = raw_dir / f"plate_{i:04d}.jpg"
            cv2.imwrite(str(img_path), img)
            
            # حفظ التسمية
            label_path = raw_dir / f"plate_{i:04d}.txt"
            with open(label_path, 'w', encoding='utf-8') as f:
                f.write(f"{plate_text}\n")
                
            # حفظ معلومات إضافية (YOLO format)
            yolo_path = raw_dir / f"plate_{i:04d}.json"
            annotation = {
                "image": str(img_path.name),
                "plate_text": plate_text,
                "bbox": [0.1, 0.3, 0.8, 0.4],  # x, y, width, height (normalized)
                "confidence": 1.0
            }
            
            with open(yolo_path, 'w', encoding='utf-8') as f:
                json.dump(annotation, f, ensure_ascii=False, indent=2)
        
        print(f"✅ تم إنشاء {num_samples} عينة في {raw_dir}")
        return raw_dir
    
    def generate_plate_image(self, plate_text, img_size=(300, 100)):
        """إنشاء صورة لوحة سيارة"""
        width, height = img_size
        
        # إنشاء خلفية
        img = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # إضافة ضوضاء خفيفة
        noise = np.random.randint(0, 30, (height, width, 3))
        img = np.clip(img.astype(int) - noise, 0, 255).astype(np.uint8)
        
        # إضافة إطار
        cv2.rectangle(img, (5, 5), (width-5, height-5), (0, 0, 0), 2)
        
        # حساب حجم النص
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 3
        
        # حساب موقع النص
        text_size = cv2.getTextSize(plate_text, font, font_scale, thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2
        
        # إضافة النص
        cv2.putText(img, plate_text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
        
        # إضافة تأثيرات عشوائية
        if np.random.random() > 0.7:
            # إضافة ظل
            cv2.putText(img, plate_text, (text_x+2, text_y+2), font, font_scale, (128, 128, 128), thickness)
            cv2.putText(img, plate_text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
        
        if np.random.random() > 0.8:
            # إضافة تشويش
            blur_kernel = np.random.choice([3, 5])
            img = cv2.GaussianBlur(img, (blur_kernel, blur_kernel), 0)
        
        return img
    
    def download_from_url(self, url, filename=None):
        """تحميل ملف من رابط"""
        if filename is None:
            filename = url.split('/')[-1]
        
        filepath = self.data_dir / filename
        
        print(f"📥 تحميل من: {url}")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f, tqdm(
                desc=filename,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            print(f"✅ تم التحميل: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"❌ فشل التحميل: {e}")
            return None
    
    def extract_archive(self, archive_path, extract_to=None):
        """استخراج ملف مضغوط"""
        if extract_to is None:
            extract_to = self.data_dir / "extracted"
        
        extract_to = Path(extract_to)
        extract_to.mkdir(exist_ok=True)
        
        print(f"📦 استخراج: {archive_path}")
        
        try:
            if archive_path.suffix.lower() == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
            elif archive_path.suffix.lower() in ['.tar', '.tar.gz', '.tgz']:
                with tarfile.open(archive_path, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_to)
            else:
                print(f"⚠️ نوع ملف غير مدعوم: {archive_path.suffix}")
                return None
            
            print(f"✅ تم الاستخراج إلى: {extract_to}")
            return extract_to
            
        except Exception as e:
            print(f"❌ فشل الاستخراج: {e}")
            return None
    
    def setup_kaggle_dataset(self, dataset_name):
        """إعداد تحميل من Kaggle"""
        print("🔧 إعداد Kaggle...")
        print("📋 المتطلبات:")
        print("   1. إنشاء حساب على Kaggle")
        print("   2. تحميل kaggle.json من Account Settings")
        print("   3. وضع الملف في ~/.kaggle/kaggle.json")
        print("   4. تثبيت: pip install kaggle")
        print()
        print("🔗 رابط إنشاء API Token:")
        print("   https://www.kaggle.com/settings/account")
        print()
        print(f"📥 لتحميل البيانات، استخدم:")
        print(f"   kaggle datasets download -d {dataset_name}")
    
    def get_dataset_info(self, dataset_name):
        """الحصول على معلومات قاعدة البيانات"""
        if dataset_name in self.available_datasets:
            return self.available_datasets[dataset_name]
        else:
            print(f"⚠️ قاعدة البيانات '{dataset_name}' غير متوفرة")
            return None

def main():
    """تشغيل أداة تحميل البيانات"""
    print("📊 أداة تحميل قواعد بيانات لوحات السيارات")
    print("=" * 50)
    
    downloader = DatasetDownloader()
    
    # عرض قواعد البيانات المتاحة
    downloader.list_available_datasets()
    
    # إنشاء قاعدة بيانات تجريبية
    print("🎨 إنشاء قاعدة بيانات تجريبية...")
    raw_dir = downloader.create_sample_dataset(num_samples=100)
    
    print("\n✅ تم إعداد البيانات!")
    print(f"📁 مسار البيانات: {raw_dir}")
    print("\n📝 الخطوات التالية:")
    print("   1. تشغيل complete_alpr_project.py")
    print("   2. أو إضافة بيانات حقيقية إلى مجلد data/raw/")

if __name__ == "__main__":
    main()
