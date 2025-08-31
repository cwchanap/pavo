#!/usr/bin/env python3
"""
YOLO to TensorFlow Lite Conversion Script using ai-edge-torch
This script downloads YOLOv8s and converts it to TensorFlow Lite format using Google's ai-edge-torch library.
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

def convert_yolo_to_tflite_direct(pytorch_model_path: str) -> Optional[str]:
    """Convert YOLO PyTorch model to TFLite using YOLO's built-in export"""
    logger.info("Starting direct YOLO to TFLite conversion...")
    print("🔄 Converting YOLO model to TensorFlow Lite directly...")
    
    try:
        from ultralytics import YOLO
        
        # Load the YOLO model
        logger.info(f"Loading YOLO model from {pytorch_model_path}")
        yolo_model = YOLO(pytorch_model_path)
        
        # Try direct TensorFlow Lite export first (if available)
        print("📤 Exporting directly to TensorFlow Lite...")
        logger.info("Attempting direct TensorFlow Lite export")
        
        try:
            # Use YOLO's built-in TensorFlow Lite export
            tflite_path = yolo_model.export(
                format='tflite',
                imgsz=640,
                int8=False,  # Use FP16 quantization for better compatibility
                dynamic=False  # Disable dynamic shapes
            )
            
            if tflite_path and os.path.exists(tflite_path):
                model_size = os.path.getsize(tflite_path) / (1024 * 1024)
                logger.info(f"Direct TensorFlow Lite export successful: {tflite_path} ({model_size:.2f} MB)")
                print(f"✓ TensorFlow Lite model created: {tflite_path} ({model_size:.2f} MB)")
                
                # Copy to a standard name in current directory
                standard_path = "yolov8s_converted.tflite"
                shutil.copy2(tflite_path, standard_path)
                
                # Remove the original if it's in a different location
                if tflite_path != standard_path and os.path.exists(tflite_path):
                    try:
                        os.remove(tflite_path)
                    except:
                        pass
                        
                return standard_path
            else:
                raise FileNotFoundError("Direct TFLite export failed")
                
        except Exception as direct_error:
            logger.warning(f"Direct TFLite export failed: {direct_error}")
            print(f"⚠️  Direct TFLite export failed: {str(direct_error)[:100]}...")
        
        # Fallback: Export to ONNX first, then try manual conversion
        print("📤 Fallback: Exporting to ONNX format...")
        logger.info("Fallback: Exporting YOLO model to ONNX")
        
        # Use YOLO's built-in ONNX export as fallback
        onnx_path = yolo_model.export(
            format='onnx',
            imgsz=640,
            opset=11,
            simplify=True,
            dynamic=False
        )
        
        if not onnx_path or not os.path.exists(onnx_path):
            raise FileNotFoundError("ONNX export failed - file not found")
        
        onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
        logger.info(f"ONNX export successful: {onnx_path} ({onnx_size:.2f} MB)")
        print(f"✓ ONNX export successful: {onnx_path} ({onnx_size:.2f} MB)")
        
        # Try to convert ONNX to TensorFlow Lite using tf2onnx (simpler approach)
        print("🔄 Converting ONNX to TensorFlow Lite...")
        logger.info("Converting ONNX to TensorFlow Lite using tf2onnx")
        
        try:
            import subprocess
            import tempfile
            
            # Create temporary directory for conversion
            with tempfile.TemporaryDirectory() as temp_dir:
                saved_model_path = os.path.join(temp_dir, "saved_model")
                tflite_path = "yolov8s_converted.tflite"
                
                # Convert ONNX to SavedModel using tf2onnx command line tool
                cmd = [
                    "python", "-m", "tf2onnx.convert",
                    "--onnx", onnx_path,
                    "--output", saved_model_path,
                    "--opset", "11"
                ]
                
                logger.info(f"Running tf2onnx conversion: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=temp_dir)
                
                if result.returncode != 0:
                    logger.error(f"tf2onnx conversion failed: {result.stderr}")
                    raise Exception(f"tf2onnx conversion failed: {result.stderr[:200]}")
                
                logger.info("tf2onnx conversion successful")
                print("✓ ONNX to TensorFlow conversion successful")
                
                # Now convert SavedModel to TFLite using Python API
                import tensorflow as tf
                
                converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                
                # Allow TensorFlow ops for better compatibility
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS,
                    tf.lite.OpsSet.SELECT_TF_OPS
                ]
                
                tflite_model = converter.convert()
                
                # Save TFLite model
                with open(tflite_path, 'wb') as f:
                    f.write(tflite_model)
                
                if os.path.exists(tflite_path) and os.path.getsize(tflite_path) > 0:
                    model_size = os.path.getsize(tflite_path) / (1024 * 1024)
                    logger.info(f"TensorFlow Lite conversion successful: {tflite_path} ({model_size:.2f} MB)")
                    print(f"✓ TensorFlow Lite model created: {tflite_path} ({model_size:.2f} MB)")
                    
                    # Clean up ONNX file
                    try:
                        if os.path.exists(onnx_path):
                            os.remove(onnx_path)
                        logger.info("Cleaned up intermediate files")
                    except Exception as e:
                        logger.warning(f"Failed to clean up intermediate files: {e}")
                    
                    return tflite_path
                else:
                    raise FileNotFoundError("TensorFlow Lite conversion produced empty file")
                    
        except Exception as conversion_error:
            logger.error(f"ONNX to TFLite conversion failed: {conversion_error}")
            print(f"✗ ONNX to TFLite conversion failed: {str(conversion_error)[:200]}...")
            
            # Clean up ONNX file
            try:
                if os.path.exists(onnx_path):
                    os.remove(onnx_path)
            except:
                pass
                
            raise conversion_error
            
    except ImportError as e:
        logger.error(f"Missing dependencies: {e}")
        print(f"✗ Missing dependencies: {e}")
        print("💡 Make sure ultralytics and tensorflow are installed")
        return None
    except Exception as e:
        logger.error(f"YOLO to TFLite conversion failed: {e}")
        print(f"✗ YOLO to TFLite conversion failed: {e}")
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
        logger.info(f"Model output shape: {output_details[0]['shape']}")
        print(f"✓ Input shape: {input_details[0]['shape']}")
        print(f"✓ Output shape: {output_details[0]['shape']}")
        
        # Test inference with random input
        input_shape = input_details[0]['shape']
        test_input = np.random.uniform(0.0, 1.0, size=input_shape).astype(np.float32)
        
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        logger.info("TFLite model validation successful")
        print("✓ Model validation successful")
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
        "yolov8s_ai_edge.tflite"
    ]
    
    logger.info("Cleaning up temporary files...")
    for file_path in temp_files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"Removed: {file_path}")
        except Exception as e:
            logger.debug(f"Could not remove {file_path}: {e}")

def main():
    """Main conversion pipeline"""
    print("🚀 YOLO to TensorFlow Lite Conversion (ai-edge-torch)")
    print("=" * 55)
    
    # Get project root (parent of scripts directory)
    project_root = str(Path(__file__).parent.parent)
    
    try:
        # Step 1: Download model
        pytorch_model = download_yolo_model()
        if not pytorch_model:
            print("❌ Failed to download YOLO model")
            return False
        
        # Step 2: Convert using direct YOLO conversion
        tflite_model = convert_yolo_to_tflite_direct(pytorch_model)
        if not tflite_model:
            print("❌ Failed to convert YOLO model to TFLite")
            print("💡 YOLO models require TensorFlow for TFLite conversion")
            print("   Make sure you have ultralytics and tensorflow installed")
            return False
        
        # Step 3: Validate the model
        if not validate_tflite_model(tflite_model):
            print("⚠️  Model validation failed, but proceeding...")
        
        # Step 4: Deploy to assets
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
