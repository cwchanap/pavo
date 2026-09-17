#!/usr/bin/env python3
"""
Script to download YOLO model and convert it to TensorFlow Lite format
"""

import os
import sys
import urllib.request
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('model_conversion.log')
    ]
)
logger = logging.getLogger(__name__)

def install_requirements():
    """Install required packages"""
    required_packages = [
        'ultralytics',
    ]
    
    print("Installing required packages...")
    logger.info("Starting package installation")
    
    for package in required_packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✓ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install required package {package}: {e}")
            print(f"✗ Failed to install required package {package}")
            return False
    
    logger.info("Package installation completed")
    return True

def download_model(model_name, model_url):
    """Download a model from a URL"""
    logger.info(f"Downloading {model_name} from {model_url}")
    print(f"Downloading {model_name}...")
    
    try:
        if os.path.exists(model_name):
            logger.info(f"Model already exists: {model_name}")
            print(f"✓ Model already exists: {model_name}")
            return model_name
            
        urllib.request.urlretrieve(model_url, model_name)
        
        if os.path.exists(model_name) and os.path.getsize(model_name) > 0:
            logger.info(f"Model downloaded successfully: {model_name}")
            print(f"✓ Model downloaded successfully: {model_name}")
            return model_name
        else:
            raise FileNotFoundError(f"Downloaded model is empty or doesn't exist")
            
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        print(f"✗ Failed to download model: {e}")
        return None

def convert_to_tflite_cli(model_name: str, imgsz: int):
    """Convert a model to TFLite using the yolo CLI"""
    print(f"Converting {model_name} to TFLite using the yolo CLI...")
    try:
        subprocess.check_call([sys.executable, "scripts/load_and_export.py", model_name, str(imgsz)])
        tflite_path = model_name.replace(".pt", ".tflite")
        if os.path.exists(tflite_path) and os.path.getsize(tflite_path) > 0:
            model_size = os.path.getsize(tflite_path) / (1024 * 1024)
            print(f"✓ Model converted to TFLite: {tflite_path} ({model_size:.2f} MB)")
            return tflite_path
        else:
            raise FileNotFoundError("TFLite conversion failed")
    except Exception as e:
        print(f"✗ Failed to convert model to TFLite: {e}")
        logger.error(f"Model conversion failed: {e}")
        return None

def copy_to_assets(tflite_path: str, target_filename: str) -> bool:
    """Copy the TensorFlow Lite model to the Android assets folder"""
    logger.info(f"Copying model to assets folder: {tflite_path}")
    
    assets_dir = Path("app/src/main/assets")
    assets_dir.mkdir(parents=True, exist_ok=True)
    
    target_path = assets_dir / target_filename
    
    try:
        import shutil
        
        if not os.path.exists(tflite_path):
            raise FileNotFoundError(f"Source file not found: {tflite_path}")
        
        shutil.copy2(tflite_path, target_path)
        
        if os.path.exists(target_path) and os.path.getsize(target_path) == os.path.getsize(tflite_path):
            logger.info(f"Model copied successfully to {target_path}")
            print(f"✓ Model copied to {target_path}")
            return True
        else:
            raise FileNotFoundError("Copy verification failed")
            
    except Exception as e:
        logger.error(f"Failed to copy model to assets: {e}")
        print(f"✗ Failed to copy model to assets: {e}")
        return False

def main():
    """Main conversion pipeline"""
    print("🚀 YOLO to TensorFlow Lite Conversion Script")
    print("=" * 50)
    
    if not install_requirements():
        print("❌ Failed to install requirements")
        return

    # Detection model
    det_model_name = "yolov12s.pt"
    det_model_url = "https://github.com/sunsmarterjie/yolov12/releases/download/turbo/yolov12s.pt"
    det_pytorch_model = download_model(det_model_name, det_model_url)
    if det_pytorch_model:
        det_tflite_model = convert_to_tflite_cli(det_pytorch_model, 640)
        if det_tflite_model:
            copy_to_assets(det_tflite_model, "yolo.tflite")

    # Classification model
    cls_model_name = "yolov12n-cls.pt"
    cls_model_url = "https://github.com/sunsmarterjie/yolov12/releases/download/cls/yolov12n-cls.pt"
    cls_pytorch_model = download_model(cls_model_name, cls_model_url)
    if cls_pytorch_model:
        cls_tflite_model = convert_to_tflite_cli(cls_pytorch_model, 224)
        if cls_tflite_model:
            copy_to_assets(cls_tflite_model, "yolov12n-cls.tflite")

if __name__ == "__main__":
    main()
