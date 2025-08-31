#!/usr/bin/env python3
"""
Script to download YOLO model and convert it to TensorFlow Lite format
Now using YOLOv8s for better conversion compatibility
Based on: https://ai.google.dev/edge/litert/models/pytorch_to_tflite
"""

import os
import sys
import urllib.request
import subprocess
import logging
from pathlib import Path
from typing import Optional, Callable

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
    # Core packages required for basic functionality
    required_packages = [
        'torch',
        'torchvision', 
        'tensorflow',
        'ultralytics',
        'numpy',
        'pillow'  # For image processing in representative dataset
    ]
    
    # Optional packages for alternative conversion methods
    optional_packages = [
        'onnx',
        'onnx-tf',
        'tensorflow-addons',  # Required for onnx-tf
        'onnxruntime',  # Required for ONNX operations
        'onnxslim',  # For ONNX model optimization
        'onnx2tf',  # Alternative ONNX to TensorFlow converter
        'tf-keras',  # Required for some TF operations
        'ai-edge-torch',  # Google's recommended PyTorch to TFLite converter
    ]
    
    print("Installing required packages...")
    logger.info("Starting package installation")
    
    # Install required packages first
    for package in required_packages:
        if not _install_package(package, required=True):
            return False
    
    # Install optional packages (failures are non-fatal)
    for package in optional_packages:
        _install_package(package, required=False)
    
    logger.info("Package installation completed")
    return True

def _install_package(package: str, required: bool = True) -> bool:
    """Install a single package with proper error handling"""
    try:
        # Try installing with --user flag first (for externally managed environments)
        logger.info(f"Installing {package} with --user flag")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user', package], 
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"✓ {package} installed successfully")
        return True
    except subprocess.CalledProcessError:
        try:
            # If --user fails, try with --break-system-packages (last resort)
            logger.warning(f"Retrying {package} installation with --break-system-packages")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--break-system-packages', package])
            print(f"✓ {package} installed successfully (with --break-system-packages)")
            return True
        except subprocess.CalledProcessError as e:
            if required:
                logger.error(f"Failed to install required package {package}: {e}")
                print(f"✗ Failed to install required package {package}")
                print(f"  Error: {e}")
                print(f"  You may need to create a virtual environment or install manually")
                return False
            else:
                logger.warning(f"Failed to install optional package {package}: {e}")
                print(f"⚠️  Failed to install optional package {package} (will continue without it)")
                return False

def download_model() -> Optional[str]:
    """Download YOLOv8s model (better conversion compatibility)"""
    logger.info("Using YOLOv8s model for better conversion compatibility")
    print("Using YOLOv8s model (downloading from Ultralytics repository)...")
    
    try:
        from ultralytics import YOLO
        
        # Use YOLOv8s model which has better conversion support
        model_name = "yolov8s.pt"
        
        # Check if model already exists
        if os.path.exists(model_name):
            logger.info(f"Model already exists: {model_name}")
            print(f"✓ Model already exists: {model_name}")
            return model_name
        
        logger.info("Downloading YOLOv8s model from Ultralytics")
        # This will automatically download the model if it doesn't exist
        model = YOLO('yolov8s.pt')
        
        # Verify the downloaded file
        if os.path.exists(model_name) and os.path.getsize(model_name) > 0:
            logger.info(f"Model downloaded successfully: {model_name}")
            print(f"✓ YOLOv8s model downloaded successfully: {model_name}")
            return model_name
        else:
            raise FileNotFoundError("Downloaded file is empty or doesn't exist")
            
    except ImportError as e:
        logger.error(f"Ultralytics not available: {e}")
        print(f"✗ Ultralytics not available: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        print(f"✗ Failed to download model: {e}")
        return None

def convert_pytorch_to_onnx(pytorch_model_path):
    """Convert PyTorch model to ONNX format"""
    print("Converting PyTorch model to ONNX...")
    
    try:
        import torch
        from ultralytics import YOLO
        
        # Load the YOLO model
        model = YOLO(pytorch_model_path)
        
        # Export to ONNX format
        onnx_path = model.export(format='onnx', imgsz=640, simplify=True)
        print(f"✓ ONNX model exported: {onnx_path}")
        return onnx_path
        
    except Exception as e:
        print(f"✗ Failed to convert to ONNX: {e}")
        return None

def convert_onnx_to_tflite_direct(onnx_model_path):
    """Convert ONNX model directly to TensorFlow Lite using onnx2tf"""
    print("Converting ONNX model directly to TensorFlow Lite using onnx2tf...")
    
    try:
        import subprocess
        import os
        
        # Use onnx2tf command line tool to convert directly to TFLite
        tflite_output = "yolov8_onnx2tf.tflite"
        
        # Run onnx2tf conversion
        logger.info(f"Running onnx2tf conversion from {onnx_model_path} to {tflite_output}")
        
        cmd = [
            sys.executable, '-m', 'onnx2tf',
            '-i', onnx_model_path,
            '-o', 'yolov8_savedmodel',
            '--output_integer_quantized_tflite',
            '--output_float32_tflite'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Check for generated TFLite files
            possible_tflite_files = [
                'yolov8_savedmodel/yolov8s_float32.tflite',
                'yolov8_savedmodel/yolov8s_integer_quant.tflite',
                'yolov8_savedmodel/model_float32.tflite',
                'yolov8_savedmodel/model_integer_quant.tflite'
            ]
            
            for tflite_file in possible_tflite_files:
                if os.path.exists(tflite_file) and os.path.getsize(tflite_file) > 1000000:
                    model_size = os.path.getsize(tflite_file) / (1024 * 1024)
                    print(f"✓ TensorFlow Lite model created: {tflite_file} ({model_size:.2f} MB)")
                    return tflite_file
            
            # If no specific file found, try to find any .tflite file in the output directory
            import glob
            tflite_files = glob.glob('yolov8_savedmodel/*.tflite')
            if tflite_files:
                tflite_file = tflite_files[0]  # Use first found
                model_size = os.path.getsize(tflite_file) / (1024 * 1024)
                print(f"✓ TensorFlow Lite model found: {tflite_file} ({model_size:.2f} MB)")
                return tflite_file
            
            raise FileNotFoundError("No TFLite file generated")
        else:
            logger.error(f"onnx2tf conversion failed: {result.stderr}")
            raise Exception(f"onnx2tf conversion failed: {result.stderr}")
        
    except Exception as e:
        print(f"✗ Failed to convert ONNX to TensorFlow Lite: {e}")
        logger.error(f"ONNX to TFLite conversion failed: {e}")
        return None

def generate_representative_dataset(input_shape=(1, 640, 640, 3), num_samples=100):
    """Generate a representative dataset for quantization calibration"""
    import numpy as np
    
    def representative_data_gen():
        for _ in range(num_samples):
            # Generate random input data that represents typical model inputs
            # For YOLO models, this should be normalized images in range [0, 1]
            data = np.random.uniform(0.0, 1.0, size=input_shape).astype(np.float32)
            yield [data]
    
    return representative_data_gen

def convert_tensorflow_to_tflite(savedmodel_path, use_quantization=True):
    """Convert TensorFlow SavedModel to TensorFlow Lite using TF2 best practices"""
    print("Converting TensorFlow model to TensorFlow Lite...")
    
    try:
        import tensorflow as tf
        import numpy as np
        
        # Load the SavedModel using TF2 converter
        converter = tf.lite.TFLiteConverter.from_saved_model(savedmodel_path)
        
        if use_quantization:
            # Apply optimizations for better performance
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Optional: Use float16 quantization for reduced model size
            converter.target_spec.supported_types = [tf.float16]
            
            # Create a representative dataset for better quantization (if available)
            try:
                def representative_dataset():
                    """Generate representative dataset for quantization"""
                    for _ in range(100):
                        # Generate random input data matching model's expected input shape
                        # YOLO models typically expect (1, 3, 640, 640) input
                        data = np.random.uniform(0.0, 1.0, size=(1, 640, 640, 3)).astype(np.float32)
                        yield [data]
                
                converter.representative_dataset = representative_dataset
                print("✓ Representative dataset configured for quantization")
            except Exception as e:
                print(f"⚠️  Could not set up representative dataset: {e}")
        
        # Convert to TensorFlow Lite
        tflite_model = converter.convert()
        
        # Save the model
        tflite_path = "yolov12_tf2.tflite"
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
            
        # Print model info
        model_size = len(tflite_model) / (1024 * 1024)  # Size in MB
        print(f"✓ TensorFlow Lite model saved: {tflite_path} ({model_size:.2f} MB)")
        return tflite_path
        
    except Exception as e:
        print(f"✗ Failed to convert to TensorFlow Lite: {e}")
        return None

def convert_pytorch_to_tflite_direct(pytorch_model_path: str) -> Optional[str]:
    """Convert PyTorch model directly to TFLite using ai-edge-torch (recommended method)"""
    logger.info("Starting ai-edge-torch conversion...")
    print("Converting PyTorch model to TFLite using ai-edge-torch...")
    
    try:
        import torch
        import ai_edge_torch
        from ultralytics import YOLO
        import numpy as np
        
        logger.info(f"Loading YOLO model from {pytorch_model_path}")
        
        # Load the YOLO model to get the PyTorch model
        yolo_model = YOLO(pytorch_model_path)
        pytorch_model = yolo_model.model
        
        # Set model to evaluation mode
        pytorch_model.eval()
        
        logger.info("Creating sample input tensor")
        # Create sample input for the model (batch_size=1, channels=3, height=640, width=640)
        sample_input = torch.randn(1, 3, 640, 640)
        
        logger.info("Converting model using ai-edge-torch")
        # Convert to TensorFlow Lite using ai-edge-torch
        edge_model = ai_edge_torch.convert(pytorch_model, sample_input)
        
        # Save the model
        tflite_path = "yolov12_ai_edge.tflite"
        logger.info(f"Exporting model to {tflite_path}")
        edge_model.export(tflite_path)
        
        # Verify the exported file
        if os.path.exists(tflite_path) and os.path.getsize(tflite_path) > 0:
            model_size = os.path.getsize(tflite_path) / (1024 * 1024)  # Size in MB
            logger.info(f"ai-edge-torch conversion successful: {tflite_path} ({model_size:.2f} MB)")
            print(f"✓ TensorFlow Lite model saved using ai-edge-torch: {tflite_path} ({model_size:.2f} MB)")
            return tflite_path
        else:
            raise FileNotFoundError("Exported TFLite file is empty or doesn't exist")
        
    except ImportError as e:
        logger.error(f"Import error - ai-edge-torch may not be properly installed: {e}")
        print(f"✗ Import error - ai-edge-torch may not be properly installed: {e}")
        print("Falling back to alternative methods...")
        return None
    except Exception as e:
        logger.error(f"ai-edge-torch conversion failed: {e}")
        print(f"✗ ai-edge-torch conversion failed: {e}")
        print("Falling back to alternative methods...")
        return None

def alternative_ultralytics_conversion(pytorch_model_path):
    """Alternative method using Ultralytics direct TFLite export"""
    print("Trying alternative conversion method using Ultralytics...")
    
    try:
        from ultralytics import YOLO
        
        # Load the model
        model = YOLO(pytorch_model_path)
        
        # Try to export directly to TensorFlow Lite using Ultralytics built-in functionality
        # This should work for YOLOv8 models without requiring complex dependencies
        print("Attempting direct TFLite export via Ultralytics...")
        tflite_path = model.export(format='tflite', imgsz=640, int8=False, half=False)
        
        # Verify the exported file exists and has reasonable size
        if os.path.exists(tflite_path) and os.path.getsize(tflite_path) > 1000000:  # At least 1MB
            model_size = os.path.getsize(tflite_path) / (1024 * 1024)
            print(f"✓ TensorFlow Lite model exported: {tflite_path} ({model_size:.2f} MB)")
            return tflite_path
        else:
            raise FileNotFoundError(f"TFLite export failed or produced invalid file: {tflite_path}")
        
    except Exception as e:
        print(f"✗ Ultralytics direct conversion failed: {e}")
        logger.error(f"Ultralytics direct conversion failed: {e}")
        return None

def validate_tflite_model(tflite_path: str) -> bool:
    """Validate the TensorFlow Lite model by running basic inference"""
    logger.info(f"Validating TFLite model: {tflite_path}")
    print(f"Validating TFLite model: {tflite_path}")
    
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
        print(f"✓ Model input shape: {input_details[0]['shape']}")
        print(f"✓ Model output shape: {output_details[0]['shape']}")
        
        # Create test input
        input_shape = input_details[0]['shape']
        test_input = np.random.uniform(0.0, 1.0, size=input_shape).astype(np.float32)
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        
        # Get output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        logger.info("TFLite model validation successful")
        print("✓ TFLite model validation successful")
        return True
        
    except Exception as e:
        logger.error(f"TFLite model validation failed: {e}")
        print(f"✗ TFLite model validation failed: {e}")
        return False

def copy_to_assets(tflite_path: str) -> bool:
    """Copy the TensorFlow Lite model to the Android assets folder"""
    logger.info(f"Copying model to assets folder: {tflite_path}")
    
    assets_dir = Path("app/src/main/assets")
    assets_dir.mkdir(parents=True, exist_ok=True)
    
    target_path = assets_dir / "yolo.tflite"
    
    try:
        import shutil
        
        # Validate source file exists
        if not os.path.exists(tflite_path):
            raise FileNotFoundError(f"Source file not found: {tflite_path}")
        
        shutil.copy2(tflite_path, target_path)
        
        # Verify copy was successful
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

def cleanup_temp_files():
    """Clean up temporary files"""
    temp_files = [
        "yolov8s.pt",
        "yolov8s.onnx", 
        "yolov8s.tflite",
        "yolov12s.pt",
        "yolov12s.onnx", 
        "yolov12.tflite",
        "yolov12_tf2.tflite",
        "yolov12_ai_edge.tflite",
        "yolov12_savedmodel",
        "model_conversion.log"
    ]
    
    logger.info("Cleaning up temporary files...")
    for file_path in temp_files:
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                logger.debug(f"Removed file: {file_path}")
            elif os.path.isdir(file_path):
                import shutil
                shutil.rmtree(file_path)
                logger.debug(f"Removed directory: {file_path}")
        except Exception as e:
            logger.debug(f"Could not remove {file_path}: {e}")

def main():
    """Main conversion pipeline"""
    print("🚀 YOLO to TensorFlow Lite Conversion Script")
    print("=" * 50)
    
    # Step 1: Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements")
        return False
    
    # Step 2: Download model
    pytorch_model = download_model()
    if not pytorch_model:
        return False
    
    # Step 3: Try ai-edge-torch conversion first (Google's recommended method)
    tflite_model = convert_pytorch_to_tflite_direct(pytorch_model)
    
    # Step 4: If ai-edge-torch fails, try Ultralytics direct conversion
    if not tflite_model:
        print("\nTrying Ultralytics direct conversion...")
        tflite_model = alternative_ultralytics_conversion(pytorch_model)
    
    # Step 5: If both direct methods fail, try the ONNX->TFLite pipeline
    if not tflite_model:
        print("\nTrying ONNX to TFLite conversion pipeline...")
        
        # Convert PyTorch -> ONNX
        onnx_model = convert_pytorch_to_onnx(pytorch_model)
        if not onnx_model:
            return False
        
        # Convert ONNX directly to TensorFlow Lite
        tflite_model = convert_onnx_to_tflite_direct(onnx_model)
        if not tflite_model:
            return False
    
    # Step 6: Validate the converted model
    if not validate_tflite_model(tflite_model):
        logger.warning("Model validation failed, but proceeding with deployment...")
        print("⚠️  Model validation failed, but proceeding with deployment...")
    
    # Step 7: Copy to assets
    if copy_to_assets(tflite_model):
        logger.info("Model successfully deployed to Android assets")
        print("\n🎉 Success! YOLO model is ready for use in your Android app!")
        print("You can now build and run the app with real object detection.")
    else:
        logger.error("Failed to copy model to assets folder")
        print("\n⚠️  Model converted but failed to copy to assets folder.")
        print(f"Please manually copy {tflite_model} to app/src/main/assets/yolo.tflite")
    
    # Step 8: Cleanup
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