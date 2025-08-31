#!/usr/bin/env python3
"""
Create a basic TensorFlow Lite model with YOLO-like structure for testing Android app integration
"""

import os
import sys
import numpy as np
from pathlib import Path

def create_basic_yolo_tflite():
    """Create a basic TFLite model with YOLO-like structure"""
    print("Creating basic TFLite model with YOLO structure...")
    
    try:
        import tensorflow as tf
        
        # Create a model that matches YOLO's input/output structure
        # Input: (640, 640, 3) - height, width, channels
        # Output: (84, 8400) - classes+coords, detections
        
        inputs = tf.keras.Input(shape=(640, 640, 3), name='input')
        
        # Simple feature extraction layers
        x = tf.keras.layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')(inputs)
        x = tf.keras.layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2D(128, 3, strides=2, padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2D(256, 3, strides=2, padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2D(512, 3, strides=2, padding='same', activation='relu')(x)
        
        # Global average pooling and reshape to match YOLO output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(84 * 8400)(x)
        outputs = tf.keras.layers.Reshape((84, 8400))(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        
        # Save the model
        tflite_path = "yolo_basic.tflite"
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        model_size = len(tflite_model) / (1024 * 1024)
        print(f"✓ Basic TFLite model created: {tflite_path} ({model_size:.2f} MB)")
        print("⚠️  Note: This is a basic model structure for testing - not the actual YOLO weights")
        
        return tflite_path
        
    except Exception as e:
        print(f"✗ Failed to create basic model: {e}")
        return None

def validate_and_deploy(tflite_path):
    """Validate the TFLite model and copy to assets"""
    if not tflite_path or not os.path.exists(tflite_path):
        print("No valid TFLite model to deploy")
        return False
    
    try:
        import tensorflow as tf
        import numpy as np
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
    """Main function"""
    print("🔧 Basic TensorFlow Lite Model Creator")
    print("=" * 40)
    
    # Create basic TFLite model
    tflite_model = create_basic_yolo_tflite()
    
    if tflite_model:
        # Validate and deploy
        if validate_and_deploy(tflite_model):
            print("\n🎉 Success! Basic YOLO model created and deployed!")
            print("This model has the correct structure but placeholder weights.")
            print("You can now test the Android app's TFLite integration.")
            return True
        else:
            print("\n⚠️  Model created but deployment failed")
            return False
    else:
        print("\n❌ Model creation failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
