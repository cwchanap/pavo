#!/usr/bin/env python3
"""
Script to download YOLOv12 model and convert it to TensorFlow Lite format
Based on: https://ai.google.dev/edge/litert/models/pytorch_to_tflite
"""

import os
import sys
import urllib.request
import subprocess
from pathlib import Path

def install_requirements():
    """Install required packages"""
    packages = [
        'torch',
        'torchvision', 
        'tensorflow',
        'ultralytics',
        'onnx',
        'onnx-tf'
    ]
    
    print("Installing required packages...")
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✓ {package} installed successfully")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {package}")
            return False
    return True

def download_model():
    """Download YOLOv12s model from the specified URL"""
    url = "https://github.com/sunsmarterjie/yolov12/releases/download/turbo/yolov12s.pt"
    model_path = "yolov12s.pt"
    
    print(f"Downloading YOLOv12s model from {url}...")
    
    try:
        urllib.request.urlretrieve(url, model_path)
        print(f"✓ Model downloaded successfully: {model_path}")
        return model_path
    except Exception as e:
        print(f"✗ Failed to download model: {e}")
        return None

def convert_pytorch_to_onnx(pytorch_model_path):
    """Convert PyTorch model to ONNX format"""
    print("Converting PyTorch model to ONNX...")
    
    try:
        import torch
        from ultralytics import YOLO
        
        # Load the YOLOv12 model
        model = YOLO(pytorch_model_path)
        
        # Export to ONNX format
        onnx_path = model.export(format='onnx', imgsz=640, simplify=True)
        print(f"✓ ONNX model exported: {onnx_path}")
        return onnx_path
        
    except Exception as e:
        print(f"✗ Failed to convert to ONNX: {e}")
        return None

def convert_onnx_to_tensorflow(onnx_model_path):
    """Convert ONNX model to TensorFlow SavedModel format"""
    print("Converting ONNX model to TensorFlow...")
    
    try:
        import onnx
        from onnx_tf.backend import prepare
        
        # Load ONNX model
        onnx_model = onnx.load(onnx_model_path)
        
        # Convert to TensorFlow
        tf_rep = prepare(onnx_model)
        
        # Export as SavedModel
        savedmodel_path = "yolov12_savedmodel"
        tf_rep.export_graph(savedmodel_path)
        print(f"✓ TensorFlow SavedModel exported: {savedmodel_path}")
        return savedmodel_path
        
    except Exception as e:
        print(f"✗ Failed to convert to TensorFlow: {e}")
        return None

def convert_tensorflow_to_tflite(savedmodel_path):
    """Convert TensorFlow SavedModel to TensorFlow Lite"""
    print("Converting TensorFlow model to TensorFlow Lite...")
    
    try:
        import tensorflow as tf
        
        # Load the SavedModel
        converter = tf.lite.TFLiteConverter.from_saved_model(savedmodel_path)
        
        # Configure converter for better performance
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        
        # Convert to TensorFlow Lite
        tflite_model = converter.convert()
        
        # Save the model
        tflite_path = "yolov12.tflite"
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
            
        print(f"✓ TensorFlow Lite model saved: {tflite_path}")
        return tflite_path
        
    except Exception as e:
        print(f"✗ Failed to convert to TensorFlow Lite: {e}")
        return None

def alternative_ultralytics_conversion(pytorch_model_path):
    """Alternative method using Ultralytics direct TFLite export"""
    print("Trying alternative conversion method using Ultralytics...")
    
    try:
        from ultralytics import YOLO
        
        # Load the model
        model = YOLO(pytorch_model_path)
        
        # Export directly to TensorFlow Lite
        tflite_path = model.export(format='tflite', imgsz=640, int8=False)
        print(f"✓ TensorFlow Lite model exported: {tflite_path}")
        return tflite_path
        
    except Exception as e:
        print(f"✗ Alternative conversion failed: {e}")
        return None

def copy_to_assets(tflite_path):
    """Copy the TensorFlow Lite model to the Android assets folder"""
    assets_dir = Path("app/src/main/assets")
    assets_dir.mkdir(parents=True, exist_ok=True)
    
    target_path = assets_dir / "yolov12.tflite"
    
    try:
        import shutil
        shutil.copy2(tflite_path, target_path)
        print(f"✓ Model copied to {target_path}")
        return True
    except Exception as e:
        print(f"✗ Failed to copy model to assets: {e}")
        return False

def cleanup_temp_files():
    """Clean up temporary files"""
    temp_files = [
        "yolov12s.pt",
        "yolov12s.onnx", 
        "yolov12.tflite",
        "yolov12_savedmodel"
    ]
    
    for file_path in temp_files:
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                import shutil
                shutil.rmtree(file_path)
        except:
            pass

def main():
    """Main conversion pipeline"""
    print("🚀 YOLOv12 to TensorFlow Lite Conversion Script")
    print("=" * 50)
    
    # Step 1: Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements")
        return False
    
    # Step 2: Download model
    pytorch_model = download_model()
    if not pytorch_model:
        return False
    
    # Step 3: Try direct Ultralytics conversion first (simpler)
    tflite_model = alternative_ultralytics_conversion(pytorch_model)
    
    # Step 4: If direct conversion fails, try the full pipeline
    if not tflite_model:
        print("\nTrying full conversion pipeline...")
        
        # Convert PyTorch -> ONNX
        onnx_model = convert_pytorch_to_onnx(pytorch_model)
        if not onnx_model:
            return False
        
        # Convert ONNX -> TensorFlow
        tf_model = convert_onnx_to_tensorflow(onnx_model)
        if not tf_model:
            return False
        
        # Convert TensorFlow -> TensorFlow Lite
        tflite_model = convert_tensorflow_to_tflite(tf_model)
        if not tflite_model:
            return False
    
    # Step 5: Copy to assets
    if copy_to_assets(tflite_model):
        print("\n🎉 Success! YOLOv12 model is ready for use in your Android app!")
        print("You can now build and run the app with real object detection.")
    else:
        print("\n⚠️  Model converted but failed to copy to assets folder.")
        print(f"Please manually copy {tflite_model} to app/src/main/assets/yolov12.tflite")
    
    # Step 6: Cleanup
    cleanup_temp_files()
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Conversion interrupted by user")
        cleanup_temp_files()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        cleanup_temp_files()
        sys.exit(1)