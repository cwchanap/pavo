#!/usr/bin/env python3
"""
Simple YOLO to TensorFlow Lite Conversion Script
This script uses YOLO's built-in TensorFlow export to create a SavedModel, then converts to TFLite.
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
logger = logging.getLogger(__name__)

def download_yolo_model() -> Optional[str]:
    """Download YOLOv8s model using Ultralytics"""
    logger.info("Downloading YOLOv8s model...")
    print("📥 Downloading YOLOv8s model...")
    
    try:
        from ultralytics import YOLO
        
        model_name = "yolov8s.pt"
        
        # Check if model already exists
        if os.path.exists(model_name):
            logger.info(f"Model already exists: {model_name}")
            print(f"✓ Model already exists: {model_name}")
            return model_name
        
        # Download the model (this will automatically download if not present)
        model = YOLO('yolov8s.pt')
        
        # Verify the download
        if os.path.exists(model_name) and os.path.getsize(model_name) > 0:
            size_mb = os.path.getsize(model_name) / (1024 * 1024)
            logger.info(f"Model downloaded successfully: {model_name} ({size_mb:.1f} MB)")
            print(f"✓ YOLOv8s model downloaded: {model_name} ({size_mb:.1f} MB)")
            return model_name
        else:
            raise FileNotFoundError("Download failed or file is empty")
            
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        print(f"✗ Failed to download model: {e}")
        return None

def convert_yolo_to_savedmodel(pytorch_model_path: str) -> Optional[str]:
    """Convert YOLO PyTorch model to TensorFlow SavedModel"""
    logger.info("Converting YOLO model to TensorFlow SavedModel...")
    print("🔄 Converting YOLO model to TensorFlow SavedModel...")
    
    try:
        from ultralytics import YOLO
        
        # Load the YOLO model
        logger.info(f"Loading YOLO model from {pytorch_model_path}")
        yolo_model = YOLO(pytorch_model_path)
        
        # Export to TensorFlow SavedModel format
        print("📤 Exporting to TensorFlow SavedModel...")
        logger.info("Exporting YOLO model to TensorFlow SavedModel")
        
        # Use YOLO's built-in TensorFlow export
        savedmodel_path = yolo_model.export(
            format='saved_model',
            imgsz=640,
            keras=False,  # Use standard TensorFlow format
            dynamic=False  # Disable dynamic shapes
        )
        
        if savedmodel_path and os.path.exists(savedmodel_path):
            logger.info(f"TensorFlow SavedModel export successful: {savedmodel_path}")
            print(f"✓ TensorFlow SavedModel created: {savedmodel_path}")
            return savedmodel_path
        else:
            raise FileNotFoundError("SavedModel export failed")
            
    except Exception as e:
        logger.error(f"SavedModel conversion failed: {e}")
        print(f"✗ SavedModel conversion failed: {e}")
        return None

def convert_savedmodel_to_tflite(savedmodel_path: str) -> Optional[str]:
    """Convert TensorFlow SavedModel to TensorFlow Lite"""
    logger.info("Converting SavedModel to TensorFlow Lite...")
    print("🔄 Converting SavedModel to TensorFlow Lite...")
    
    try:
        import tensorflow as tf
        
        # Load the SavedModel and convert to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(savedmodel_path)
        
        # Set optimization flags
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Allow TensorFlow ops for operations that don't have TFLite equivalents
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        
        # Convert to TFLite
        logger.info("Performing TensorFlow Lite conversion...")
        tflite_model = converter.convert()
        
        # Save TFLite model
        tflite_path = "yolov8s_converted.tflite"
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        if os.path.exists(tflite_path) and os.path.getsize(tflite_path) > 0:
            model_size = os.path.getsize(tflite_path) / (1024 * 1024)
            logger.info(f"TensorFlow Lite conversion successful: {tflite_path} ({model_size:.2f} MB)")
            print(f"✓ TensorFlow Lite model created: {tflite_path} ({model_size:.2f} MB)")
            return tflite_path
        else:
            raise FileNotFoundError("TensorFlow Lite conversion produced empty file")
            
    except Exception as e:
        logger.error(f"TensorFlow Lite conversion failed: {e}")
        print(f"✗ TensorFlow Lite conversion failed: {e}")
        return None

def validate_tflite_model(tflite_path: str) -> bool:
    """Validate the TensorFlow Lite model"""
    logger.info(f"Validating TFLite model: {tflite_path}")
    print(f"🧪 Validating TFLite model: {tflite_path}")
    
    try:
        import tensorflow as tf
        import numpy as np
        
        # Load the TFLite model
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        logger.info(f"Model input shape: {input_details[0]['shape']}")
        logger.info(f"Model has {len(output_details)} outputs")
        print(f"✓ Input shape: {input_details[0]['shape']}")
        print(f"✓ Number of outputs: {len(output_details)}")
        
        # Print output shapes
        for i, output in enumerate(output_details):
            print(f"  Output {i}: {output['shape']}")
        
        # Test inference with random input
        input_shape = input_details[0]['shape']
        test_input = np.random.uniform(0.0, 1.0, size=input_shape).astype(np.float32)
        
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        
        # Get all outputs
        outputs = []
        for output_detail in output_details:
            output_data = interpreter.get_tensor(output_detail['index'])
            outputs.append(output_data)
        
        logger.info("TFLite model validation successful")
        print("✓ Model validation successful - inference completed")
        return True
        
    except Exception as e:
        logger.error(f"Model validation failed: {e}")
        print(f"✗ Model validation failed: {e}")
        return False

def deploy_to_assets(tflite_path: str, project_root: str) -> bool:
    """Copy the TensorFlow Lite model to Android assets folder"""
    logger.info(f"Deploying model to assets: {tflite_path}")
    print("📱 Deploying model to Android assets...")
    
    try:
        assets_dir = Path(project_root) / "app" / "src" / "main" / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        
        target_path = assets_dir / "yolo.tflite"
        
        # Copy the model
        shutil.copy2(tflite_path, target_path)
        
        # Verify the copy
        if os.path.exists(target_path) and os.path.getsize(target_path) == os.path.getsize(tflite_path):
            model_size = os.path.getsize(target_path) / (1024 * 1024)
            logger.info(f"Model deployed successfully to {target_path}")
            print(f"✓ Model deployed to {target_path} ({model_size:.2f} MB)")
            return True
        else:
            raise FileNotFoundError("Copy verification failed")
            
    except Exception as e:
        logger.error(f"Failed to deploy model: {e}")
        print(f"✗ Failed to deploy model: {e}")
        return False

def cleanup_temp_files():
    """Clean up temporary files"""
    temp_files = [
        "yolov8s.pt",
        "yolov8s_converted.tflite",
        "calibration_image_sample_data_20x128x128x3_float32.npy"
    ]
    
    temp_dirs = [
        "yolov8s_saved_model"
    ]
    
    logger.info("Cleaning up temporary files...")
    for file_path in temp_files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"Removed: {file_path}")
        except Exception as e:
            logger.debug(f"Could not remove {file_path}: {e}")
    
    for dir_path in temp_dirs:
        try:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
                logger.debug(f"Removed directory: {dir_path}")
        except Exception as e:
            logger.debug(f"Could not remove directory {dir_path}: {e}")

def main():
    """Main conversion pipeline"""
    print("🚀 YOLO to TensorFlow Lite Conversion (Simple)")
    print("=" * 50)
    
    # Get project root (parent of scripts directory)
    project_root = str(Path(__file__).parent.parent)
    
    try:
        # Step 1: Download model
        pytorch_model = download_yolo_model()
        if not pytorch_model:
            print("❌ Failed to download YOLO model")
            return False
        
        # Step 2: Convert to SavedModel first
        savedmodel_path = convert_yolo_to_savedmodel(pytorch_model)
        if not savedmodel_path:
            print("❌ Failed to convert YOLO model to SavedModel")
            print("💡 This requires TensorFlow to be installed")
            return False
        
        # Step 3: Convert SavedModel to TFLite
        tflite_model = convert_savedmodel_to_tflite(savedmodel_path)
        if not tflite_model:
            print("❌ Failed to convert SavedModel to TFLite")
            return False
        
        # Step 4: Validate the model
        if not validate_tflite_model(tflite_model):
            print("⚠️  Model validation failed, but proceeding...")
        
        # Step 5: Deploy to assets
        if deploy_to_assets(tflite_model, project_root):
            print("\n🎉 Success! YOLO model converted and deployed!")
            print("Your Android app now has a real YOLO TensorFlow Lite model.")
            print("You can build and test the app with actual object detection.")
        else:
            print("\n⚠️  Model converted but deployment failed")
            print(f"Please manually copy {tflite_model} to app/src/main/assets/yolo.tflite")
        
        return True
        
    except Exception as e:
        logger.error(f"Conversion pipeline failed: {e}")
        print(f"\n❌ Conversion failed: {e}")
        return False
    
    finally:
        # Always cleanup temporary files
        cleanup_temp_files()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
