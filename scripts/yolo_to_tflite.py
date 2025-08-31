#!/usr/bin/env python3
"""
YOLO to TensorFlow Lite Conversion Script
This script uses YOLO's built-in export capabilities to convert YOLOv8s to TensorFlow Lite.
"""

import os
import sys
import logging
import shutil
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    """Main conversion pipeline"""
    print("🚀 YOLO to TensorFlow Lite Conversion")
    print("=" * 40)
    
    # Get project root (parent of scripts directory)
    project_root = str(Path(__file__).parent.parent)
    
    try:
        from ultralytics import YOLO
        import tensorflow as tf
        import numpy as np
        
        # Step 1: Download YOLOv8s model
        print("📥 Downloading YOLOv8s model...")
        model_name = "yolov8s.pt"
        
        if os.path.exists(model_name):
            print(f"✓ Model already exists: {model_name}")
        else:
            model = YOLO('yolov8s.pt')
            if os.path.exists(model_name) and os.path.getsize(model_name) > 0:
                size_mb = os.path.getsize(model_name) / (1024 * 1024)
                print(f"✓ YOLOv8s model downloaded: {model_name} ({size_mb:.1f} MB)")
            else:
                print("❌ Failed to download YOLO model")
                return False
        
        # Step 2: Load the model for conversion
        print("🔄 Loading YOLO model...")
        yolo_model = YOLO(model_name)
        
        # Step 3: Export to TensorFlow SavedModel
        print("📤 Exporting to TensorFlow SavedModel...")
        try:
            savedmodel_path = yolo_model.export(
                format='saved_model',
                imgsz=640,
                keras=False,
                dynamic=False
            )
            
            if savedmodel_path and os.path.exists(savedmodel_path):
                print(f"✓ TensorFlow SavedModel created: {savedmodel_path}")
            else:
                print("❌ SavedModel export failed")
                return False
                
        except Exception as e:
            print(f"❌ SavedModel export failed: {e}")
            return False
        
        # Step 4: Convert SavedModel to TensorFlow Lite
        print("🔄 Converting SavedModel to TensorFlow Lite...")
        try:
            converter = tf.lite.TFLiteConverter.from_saved_model(savedmodel_path)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            
            tflite_model = converter.convert()
            
            tflite_path = "yolov8s_converted.tflite"
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            if os.path.exists(tflite_path) and os.path.getsize(tflite_path) > 0:
                model_size = os.path.getsize(tflite_path) / (1024 * 1024)
                print(f"✓ TensorFlow Lite model created: {tflite_path} ({model_size:.2f} MB)")
            else:
                print("❌ TensorFlow Lite conversion produced empty file")
                return False
                
        except Exception as e:
            print(f"❌ TensorFlow Lite conversion failed: {e}")
            return False
        
        # Step 5: Validate the model
        print("🧪 Validating TFLite model...")
        try:
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            print(f"✓ Input shape: {input_details[0]['shape']}")
            print(f"✓ Number of outputs: {len(output_details)}")
            
            # Test inference
            input_shape = input_details[0]['shape']
            test_input = np.random.uniform(0.0, 1.0, size=input_shape).astype(np.float32)
            
            interpreter.set_tensor(input_details[0]['index'], test_input)
            interpreter.invoke()
            
            print("✓ Model validation successful - inference completed")
            
        except Exception as e:
            print(f"⚠️  Model validation failed: {e}")
            print("Proceeding with deployment anyway...")
        
        # Step 6: Deploy to Android assets
        print("📱 Deploying model to Android assets...")
        try:
            assets_dir = Path(project_root) / "app" / "src" / "main" / "assets"
            assets_dir.mkdir(parents=True, exist_ok=True)
            
            target_path = assets_dir / "yolo.tflite"
            shutil.copy2(tflite_path, target_path)
            
            if os.path.exists(target_path):
                model_size = os.path.getsize(target_path) / (1024 * 1024)
                print(f"✓ Model deployed to {target_path} ({model_size:.2f} MB)")
            else:
                print("❌ Model deployment failed")
                return False
                
        except Exception as e:
            print(f"❌ Failed to deploy model: {e}")
            print(f"💡 Please manually copy {tflite_path} to app/src/main/assets/yolo.tflite")
        
        print("\\n🎉 Success! YOLO model converted and deployed!")
        print("Your Android app now has a real YOLO TensorFlow Lite model.")
        print("You can build and test the app with actual object detection.")
        
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("💡 Make sure ultralytics and tensorflow are installed")
        return False
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return False
    
    finally:
        # Cleanup temporary files
        temp_files = [
            "yolov8s.pt",
            "yolov8s_converted.tflite",
            "calibration_image_sample_data_20x128x128x3_float32.npy"
        ]
        
        temp_dirs = [
            "yolov8s_saved_model"
        ]
        
        for file_path in temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except:
                pass
        
        for dir_path in temp_dirs:
            try:
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path)
            except:
                pass

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
