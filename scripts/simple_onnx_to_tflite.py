#!/usr/bin/env python3
"""
Simple script to convert the existing ONNX model to TensorFlow Lite
This avoids the complex dependency issues with onnx2tf and tf-keras
"""

import os
import sys
import subprocess
import tempfile
import numpy as np

def convert_onnx_to_tflite_simple(onnx_path):
    """Convert ONNX to TFLite using TensorFlow's built-in converter"""
    print(f"Converting {onnx_path} to TensorFlow Lite...")
    
    try:
        import tensorflow as tf
        import onnx
        
        # Load ONNX model
        print("Loading ONNX model...")
        onnx_model = onnx.load(onnx_path)
        
        # Create a temporary SavedModel directory
        with tempfile.TemporaryDirectory() as temp_dir:
            savedmodel_path = os.path.join(temp_dir, "model")
            
            # Try to convert using tf.experimental.onnx_to_savedmodel 
            try:
                print("Converting ONNX to SavedModel...")
                import tf2onnx
                from onnx_tf.backend import prepare
                
                # Convert ONNX to TensorFlow representation
                tf_rep = prepare(onnx_model)
                tf_rep.export_graph(savedmodel_path)
                
                print("Converting SavedModel to TFLite...")
                # Convert to TensorFlow Lite
                converter = tf.lite.TFLiteConverter.from_saved_model(savedmodel_path)
                
                # Set optimization options
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]
                
                # Convert
                tflite_model = converter.convert()
                
                # Save the TFLite model
                tflite_path = "yolov8s_simple.tflite"
                with open(tflite_path, 'wb') as f:
                    f.write(tflite_model)
                
                model_size = len(tflite_model) / (1024 * 1024)
                print(f"✓ TensorFlow Lite model saved: {tflite_path} ({model_size:.2f} MB)")
                return tflite_path
                
            except Exception as e:
                print(f"Method 1 failed: {e}")
                return None
                
    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        return None

def validate_and_deploy(tflite_path):
    """Validate the TFLite model and copy to assets"""
    if not tflite_path or not os.path.exists(tflite_path):
        print("No valid TFLite model to deploy")
        return False
    
    try:
        import tensorflow as tf
        import numpy as np
        from pathlib import Path
        import shutil
        
        # Validate the model
        print(f"Validating TFLite model: {tflite_path}")
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"✓ Model input shape: {input_details[0]['shape']}")
        print(f"✓ Model output shape: {output_details[0]['shape']}")
        
        # Test inference
        input_shape = input_details[0]['shape']
        test_input = np.random.uniform(0.0, 1.0, size=input_shape).astype(np.float32)
        
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        print("✓ TFLite model validation successful")
        
        # Copy to assets folder
        assets_dir = Path("app/src/main/assets")
        assets_dir.mkdir(parents=True, exist_ok=True)
        target_path = assets_dir / "yolo.tflite"
        
        shutil.copy2(tflite_path, target_path)
        print(f"✓ Model copied to {target_path}")
        
        return True
        
    except Exception as e:
        print(f"✗ Validation/deployment failed: {e}")
        return False

def main():
    """Simple conversion main function"""
    print("🔧 Simple ONNX to TensorFlow Lite Converter")
    print("=" * 45)
    
    onnx_file = "yolov8s.onnx"
    
    if not os.path.exists(onnx_file):
        print(f"✗ ONNX file not found: {onnx_file}")
        print("Run the main conversion script first to generate the ONNX model")
        return False
    
    # Convert ONNX to TFLite
    tflite_model = convert_onnx_to_tflite_simple(onnx_file)
    
    if tflite_model:
        # Validate and deploy
        if validate_and_deploy(tflite_model):
            print("\n🎉 Success! YOLO model converted and deployed!")
            return True
        else:
            print("\n⚠️  Model converted but deployment failed")
            return False
    else:
        print("\n❌ All conversion methods failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
