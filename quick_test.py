#!/usr/bin/env python3
"""
اختبار سريع للمشروع
Quick Project Test

يتحقق من:
1. استيراد جميع المكتبات
2. إنشاء البيانات التجريبية
3. بناء النموذج
4. تشغيل عملية تدريب قصيرة
"""

import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def test_imports():
    """اختبار استيراد المكتبات"""
    print("🔍 اختبار استيراد المكتبات...")
    
    try:
        import numpy as np
        print("✅ NumPy")
    except ImportError:
        print("❌ NumPy - pip install numpy")
        return False
    
    try:
        import cv2
        print("✅ OpenCV")
    except ImportError:
        print("❌ OpenCV - pip install opencv-python")
        return False
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__}")
    except ImportError:
        print("❌ TensorFlow - pip install tensorflow")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✅ Matplotlib")
    except ImportError:
        print("❌ Matplotlib - pip install matplotlib")
        return False
    
    try:
        import sklearn
        print("✅ Scikit-learn")
    except ImportError:
        print("❌ Scikit-learn - pip install scikit-learn")
        return False
    
    try:
        import pandas as pd
        print("✅ Pandas")
    except ImportError:
        print("❌ Pandas - pip install pandas")
        return False
    
    try:
        import seaborn as sns
        print("✅ Seaborn")
    except ImportError:
        print("❌ Seaborn - pip install seaborn")
        return False
    
    print("✅ جميع المكتبات متوفرة!")
    return True

def test_project_files():
    """اختبار وجود ملفات المشروع"""
    print("\n📁 اختبار ملفات المشروع...")
    
    required_files = [
        "data_downloader.py",
        "complete_alpr_project.py", 
        "train_and_evaluate.py",
        "run_complete_project.py",
        "PROJECT_GUIDE.md"
    ]
    
    missing_files = []
    for file in required_files:
        if Path(file).exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file}")
            missing_files.append(file)
    
    if missing_files:
        print(f"⚠️ ملفات مفقودة: {missing_files}")
        return False
    
    print("✅ جميع ملفات المشروع موجودة!")
    return True

def test_data_creation():
    """اختبار إنشاء البيانات"""
    print("\n🎨 اختبار إنشاء البيانات...")
    
    try:
        from data_downloader import DatasetDownloader
        
        downloader = DatasetDownloader()
        
        # إنشاء عينة صغيرة
        raw_dir = downloader.create_sample_dataset(num_samples=10)
        
        # التحقق من الملفات
        image_files = list(raw_dir.glob("*.jpg"))
        label_files = list(raw_dir.glob("*.txt"))
        
        if len(image_files) == 10 and len(label_files) == 10:
            print("✅ تم إنشاء البيانات التجريبية بنجاح!")
            return True
        else:
            print(f"❌ عدد الملفات غير صحيح: {len(image_files)} صور، {len(label_files)} تسميات")
            return False
            
    except Exception as e:
        print(f"❌ خطأ في إنشاء البيانات: {e}")
        return False

def test_model_building():
    """اختبار بناء النموذج"""
    print("\n🏗️ اختبار بناء النموذج...")
    
    try:
        from complete_alpr_project import ALPRModel
        
        # بناء نموذج صغير للاختبار
        model = ALPRModel(input_shape=(64, 128, 3), num_classes=10)
        cnn_model = model.build_cnn_model()
        
        # التحقق من النموذج
        if cnn_model is not None:
            print(f"✅ تم بناء النموذج بنجاح!")
            print(f"📊 عدد المعاملات: {cnn_model.count_params():,}")
            return True
        else:
            print("❌ فشل في بناء النموذج")
            return False
            
    except Exception as e:
        print(f"❌ خطأ في بناء النموذج: {e}")
        return False

def test_data_splitting():
    """اختبار تقسيم البيانات"""
    print("\n📊 اختبار تقسيم البيانات...")
    
    try:
        from complete_alpr_project import ALPRDatasetManager
        
        dataset_manager = ALPRDatasetManager()
        
        # تقسيم البيانات
        splits = dataset_manager.split_dataset(
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1
        )
        
        # التحقق من التقسيم
        if splits and len(splits) == 3:
            print("✅ تم تقسيم البيانات بنجاح!")
            for split_name, files in splits.items():
                print(f"   📂 {split_name}: {len(files)} ملف")
            return True
        else:
            print("❌ فشل في تقسيم البيانات")
            return False
            
    except Exception as e:
        print(f"❌ خطأ في تقسيم البيانات: {e}")
        return False

def test_quick_training():
    """اختبار تدريب سريع"""
    print("\n🚀 اختبار تدريب سريع (عصر واحد)...")
    
    try:
        import tensorflow as tf
        from complete_alpr_project import ALPRModel
        
        # إنشاء بيانات تجريبية صغيرة
        X_train = tf.random.normal((20, 64, 128, 3))
        y_train = tf.random.uniform((20,), maxval=10, dtype=tf.int32)
        
        X_val = tf.random.normal((5, 64, 128, 3))
        y_val = tf.random.uniform((5,), maxval=10, dtype=tf.int32)
        
        # بناء النموذج
        model = ALPRModel(input_shape=(64, 128, 3), num_classes=10)
        cnn_model = model.build_cnn_model()
        
        # تدريب لعصر واحد
        history = cnn_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=1,
            verbose=0
        )
        
        if history and len(history.history['loss']) > 0:
            print("✅ تم التدريب التجريبي بنجاح!")
            print(f"📈 خسارة التدريب: {history.history['loss'][0]:.4f}")
            return True
        else:
            print("❌ فشل في التدريب التجريبي")
            return False
            
    except Exception as e:
        print(f"❌ خطأ في التدريب التجريبي: {e}")
        return False

def run_all_tests():
    """تشغيل جميع الاختبارات"""
    print("🧪 اختبار شامل للمشروع")
    print("=" * 50)
    
    tests = [
        ("استيراد المكتبات", test_imports),
        ("ملفات المشروع", test_project_files),
        ("إنشاء البيانات", test_data_creation),
        ("تقسيم البيانات", test_data_splitting),
        ("بناء النموذج", test_model_building),
        ("التدريب التجريبي", test_quick_training)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 اختبار: {test_name}")
        print("-" * 30)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ نجح اختبار: {test_name}")
            else:
                print(f"❌ فشل اختبار: {test_name}")
        except Exception as e:
            print(f"❌ خطأ في اختبار {test_name}: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 النتائج النهائية: {passed}/{total} اختبارات نجحت")
    
    if passed == total:
        print("🎉 جميع الاختبارات نجحت! المشروع جاهز للتشغيل.")
        print("\n🚀 الخطوات التالية:")
        print("   1. python run_complete_project.py --mode full")
        print("   2. أو python data_downloader.py")
        print("   3. ثم python complete_alpr_project.py")
        return True
    else:
        print("⚠️ بعض الاختبارات فشلت. يرجى مراجعة الأخطاء أعلاه.")
        print("\n🔧 حلول مقترحة:")
        print("   1. تثبيت المكتبات المفقودة")
        print("   2. التأكد من وجود جميع ملفات المشروع")
        print("   3. إعادة تشغيل الاختبار")
        return False

def main():
    """الدالة الرئيسية"""
    print("🚗 اختبار سريع لمشروع التعرف على لوحات السيارات")
    print("=" * 60)
    
    success = run_all_tests()
    
    if success:
        print("\n✅ المشروع جاهز للاستخدام!")
    else:
        print("\n❌ يحتاج المشروع إلى إصلاحات قبل التشغيل.")
    
    return success

if __name__ == "__main__":
    main()
