#!/usr/bin/env python3
"""
تشغيل المشروع الكامل للتعرف على لوحات السيارات
Complete ALPR Project Runner

🎯 الهدف: تطبيق عملي شامل لمفاهيم التعلم العميق في التعرف على لوحات السيارات

المراحل:
1. ✅ تحديد الفكرة: التعرف على لوحات السيارات
2. 📊 جلب وتجهيز قاعدة البيانات
3. 🏗️ بناء نموذج تعلم عميق مناسب
4. 💻 كتابة الكود بلغة البايثون
5. 🚀 تدريب النموذج
6. 📈 تحليل النتائج وتقييم الأداء
"""

import os
import sys
import argparse
from pathlib import Path
import json

# إضافة المجلد الحالي للمسار
sys.path.append(str(Path(__file__).parent))

def print_banner():
    """طباعة شعار المشروع"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║        🚗 مشروع التعرف على لوحات السيارات 🚗                ║
    ║           Automatic License Plate Recognition                ║
    ║                                                              ║
    ║  📚 تطبيق عملي لمفاهيم التعلم العميق والرؤية الحاسوبية      ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def print_project_overview():
    """عرض نظرة عامة على المشروع"""
    overview = """
    🎯 أهداف المشروع:
    ═══════════════════
    
    1. 🔍 تحديد الفكرة: التعرف الآلي على لوحات السيارات
    2. 📊 تجهيز قاعدة البيانات: تحميل وتنظيم صور اللوحات
    3. 🏗️ بناء النموذج: شبكة عصبية تطبيقية (CNN)
    4. 💻 البرمجة: تطبيق شامل بلغة Python
    5. 🚀 التدريب: تدريب النموذج على البيانات
    6. 📈 التقييم: تحليل الأداء والنتائج
    
    📋 مكونات المشروع:
    ═══════════════════
    
    • data_downloader.py     - تحميل وإعداد البيانات
    • complete_alpr_project.py - النموذج الأساسي
    • train_and_evaluate.py  - التدريب والتقييم
    • run_complete_project.py - التشغيل الشامل
    
    📊 تقسيم البيانات:
    ═══════════════════
    
    • 80% للتدريب (Training)
    • 10% للتحقق (Validation) 
    • 10% للاختبار (Testing)
    """
    print(overview)

def setup_project_structure():
    """إعداد هيكل المشروع"""
    print("📁 إعداد هيكل المشروع...")
    
    directories = [
        "data/raw",
        "data/processed", 
        "data/train/images", "data/train/labels",
        "data/val/images", "data/val/labels",
        "data/test/images", "data/test/labels",
        "models",
        "results",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ تم إنشاء هيكل المشروع")
    
    # إنشاء ملف README
    readme_content = """# مشروع التعرف على لوحات السيارات

## نظرة عامة
هذا مشروع تطبيقي شامل للتعرف على لوحات السيارات باستخدام التعلم العميق.

## هيكل المشروع
```
├── data/                    # قاعدة البيانات
│   ├── raw/                # البيانات الخام
│   ├── train/              # بيانات التدريب
│   ├── val/                # بيانات التحقق
│   └── test/               # بيانات الاختبار
├── models/                 # النماذج المدربة
├── results/                # النتائج والتحليلات
└── logs/                   # سجلات التدريب
```

## كيفية التشغيل
```bash
python run_complete_project.py --mode full
```

## المتطلبات
- Python 3.8+
- TensorFlow 2.x
- OpenCV
- Matplotlib
- Scikit-learn
"""
    
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    print("📄 تم إنشاء ملف README.md")

def run_data_preparation():
    """تشغيل مرحلة إعداد البيانات"""
    print("\n" + "="*60)
    print("📊 المرحلة 1: إعداد قاعدة البيانات")
    print("="*60)
    
    try:
        from data_downloader import DatasetDownloader
        
        downloader = DatasetDownloader()
        
        # عرض قواعد البيانات المتاحة
        downloader.list_available_datasets()
        
        # إنشاء قاعدة بيانات تجريبية
        print("\n🎨 إنشاء قاعدة بيانات تجريبية...")
        raw_dir = downloader.create_sample_dataset(num_samples=200)
        
        print("✅ تم إعداد البيانات بنجاح!")
        return True
        
    except Exception as e:
        print(f"❌ خطأ في إعداد البيانات: {e}")
        return False

def run_data_splitting():
    """تشغيل مرحلة تقسيم البيانات"""
    print("\n" + "="*60)
    print("📈 المرحلة 2: تقسيم البيانات (80% - 10% - 10%)")
    print("="*60)
    
    try:
        from complete_alpr_project import ALPRDatasetManager
        
        dataset_manager = ALPRDatasetManager()
        
        # تقسيم البيانات
        splits = dataset_manager.split_dataset(
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1
        )
        
        # عرض عينات من البيانات
        dataset_manager.visualize_dataset('train', num_samples=6)
        
        print("✅ تم تقسيم البيانات بنجاح!")
        return True
        
    except Exception as e:
        print(f"❌ خطأ في تقسيم البيانات: {e}")
        return False

def run_model_building():
    """تشغيل مرحلة بناء النموذج"""
    print("\n" + "="*60)
    print("🏗️ المرحلة 3: بناء نموذج التعلم العميق")
    print("="*60)
    
    try:
        from complete_alpr_project import ALPRModel
        
        # بناء النموذج
        model = ALPRModel(input_shape=(128, 384, 3), num_classes=36)
        cnn_model = model.build_cnn_model()
        
        # عرض هيكل النموذج
        print("\n📋 هيكل النموذج:")
        cnn_model.summary()
        
        # حفظ هيكل النموذج
        with open("results/model_architecture.txt", "w") as f:
            cnn_model.summary(print_fn=lambda x: f.write(x + '\n'))
        
        print("✅ تم بناء النموذج بنجاح!")
        return model
        
    except Exception as e:
        print(f"❌ خطأ في بناء النموذج: {e}")
        return None

def run_training():
    """تشغيل مرحلة التدريب"""
    print("\n" + "="*60)
    print("🚀 المرحلة 4: تدريب النموذج")
    print("="*60)
    
    try:
        from train_and_evaluate import ALPRTrainer
        
        # إنشاء المدرب
        trainer = ALPRTrainer()
        
        # تدريب النموذج
        print("🔥 بدء التدريب...")
        history, test_gen = trainer.train_model(epochs=20, batch_size=16)
        
        print("✅ تم الانتهاء من التدريب!")
        return trainer, history, test_gen
        
    except Exception as e:
        print(f"❌ خطأ في التدريب: {e}")
        return None, None, None

def run_evaluation(trainer, test_gen):
    """تشغيل مرحلة التقييم"""
    print("\n" + "="*60)
    print("📊 المرحلة 5: تقييم الأداء وتحليل النتائج")
    print("="*60)
    
    try:
        # تقييم النموذج
        results = trainer.evaluate_model(test_gen)
        
        # رسم النتائج
        trainer.plot_results(trainer.model.history, results)
        
        # إنشاء التقرير
        trainer.generate_report(results)
        
        print("✅ تم الانتهاء من التقييم!")
        return results
        
    except Exception as e:
        print(f"❌ خطأ في التقييم: {e}")
        return None

def print_final_summary(results):
    """طباعة الملخص النهائي"""
    print("\n" + "="*60)
    print("🎉 ملخص نتائج المشروع")
    print("="*60)
    
    if results:
        summary = f"""
    📊 نتائج الأداء النهائية:
    ═══════════════════════════
    
    🎯 دقة النموذج: {results['test_accuracy']:.4f} ({results['test_accuracy']*100:.2f}%)
    📉 خسارة الاختبار: {results['test_loss']:.4f}
    🔢 عدد الفئات: {len(results['class_labels'])}
    
    📁 الملفات المُنتجة:
    ═══════════════════
    
    • models/best_model.h5           - أفضل نموذج مدرب
    • models/final_model.h5          - النموذج النهائي  
    • results/evaluation_results.json - نتائج التقييم
    • results/complete_analysis.png   - التحليل البصري
    • results/project_report.md       - التقرير الشامل
    
    ✅ تم إنجاز جميع مراحل المشروع بنجاح!
    """
    else:
        summary = """
    ⚠️ لم يتم إكمال جميع مراحل المشروع.
    يرجى مراجعة الأخطاء أعلاه وإعادة المحاولة.
    """
    
    print(summary)

def main():
    """الدالة الرئيسية"""
    parser = argparse.ArgumentParser(description="مشروع التعرف على لوحات السيارات")
    parser.add_argument("--mode", choices=["full", "data", "train", "eval"], 
                       default="full", help="وضع التشغيل")
    parser.add_argument("--epochs", type=int, default=20, help="عدد عصور التدريب")
    parser.add_argument("--batch_size", type=int, default=16, help="حجم الدفعة")
    
    args = parser.parse_args()
    
    # طباعة الشعار والنظرة العامة
    print_banner()
    print_project_overview()
    
    # إعداد هيكل المشروع
    setup_project_structure()
    
    results = None
    
    if args.mode in ["full", "data"]:
        # 1. إعداد البيانات
        if not run_data_preparation():
            return
        
        # 2. تقسيم البيانات
        if not run_data_splitting():
            return
    
    if args.mode in ["full", "train"]:
        # 3. بناء النموذج
        model = run_model_building()
        if model is None:
            return
        
        # 4. التدريب
        trainer, history, test_gen = run_training()
        if trainer is None:
            return
    
    if args.mode in ["full", "eval"]:
        # 5. التقييم
        if 'trainer' in locals() and 'test_gen' in locals():
            results = run_evaluation(trainer, test_gen)
    
    # الملخص النهائي
    print_final_summary(results)
    
    print("\n🎓 تم إنجاز المشروع التطبيقي بنجاح!")
    print("📚 هذا المشروع يغطي جميع مراحل تطبيق التعلم العميق عملياً")

if __name__ == "__main__":
    main()
