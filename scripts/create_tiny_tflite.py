#!/usr/bin/env python3
"""
Create a tiny TensorFlow Lite model for testing Android app integration
This creates a very small model with minimal parameters but correct YOLO structure
"""

import os
import sys
import numpy as np
from pathlib import Path

def create_tiny_yolo_tflite():
    """Create a tiny TFLite model with YOLO-like structure"""
    print("Creating tiny TFLite model with YOLO structure...")
    
    try:
        import tensorflow as tf
        
        # Create a very small model for testing
        # Input: (640, 640, 3) - height, width, channels  
        # Output: (84, 8400) - classes+coords, detections
        
        inputs = tf.keras.Input(shape=(640, 640, 3), name='input')
        
        # Very aggressive downsampling to minimize parameters
        x = tf.keras.layers.Conv2D(4, 16, strides=16, padding='same', activation='relu')(inputs)  # 40x40x4
        x = tf.keras.layers.Conv2D(8, 8, strides=8, padding='same', activation='relu')(x)        # 5x5x8
        x = tf.keras.layers.Conv2D(16, 5, strides=5, padding='same', activation='relu')(x)       # 1x1x16
        
        # Flatten and create minimal dense layers
        x = tf.keras.layers.Flatten()(x)  # 16 features
        
        # Use much smaller intermediate layer
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        
        # Create a small "prototype" output that we then tile/repeat to match YOLO dimensions
        # Instead of 84*8400 = 705,600 outputs, create a small prototype and replicate it
        prototype_size = 84  # Just the feature dimension
        x = tf.keras.layers.Dense(prototype_size)(x)
        
        # Reshape and tile to create the 8400 dimension
        x = tf.keras.layers.Reshape((84, 1))(x)
        x = tf.keras.layers.Lambda(lambda t: tf.tile(t, [1, 1, 8400]))(x)
        
        outputs = x  # Shape: (None, 84, 8400)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Print model summary
        model.summary()
        
        # Convert to TFLite with light optimization (no int8 to keep it simple)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # Use float16 for smaller size but simpler quantization
        converter.target_spec.supported_types = [tf.float16]
        
        tflite_model = converter.convert()
        
        # Save the model
        tflite_path = "yolo_tiny.tflite"
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        model_size = len(tflite_model) / (1024 * 1024)
        print(f"✓ Tiny TFLite model created: {tflite_path} ({model_size:.2f} MB)")
        print("⚠️  Note: This is a minimal model for testing - not actual YOLO weights")
        
        return tflite_path
        
    except Exception as e:
        print(f"✗ Failed to create tiny model: {e}")
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
        print(f"✓ Model input type: {input_details[0]['dtype']}")
        print(f"✓ Model output shape: {output_details[0]['shape']}")
        print(f"✓ Model output type: {output_details[0]['dtype']}")
        
        # Test inference
        input_shape = input_details[0]['shape']
        test_input = np.random.uniform(0.0, 1.0, size=input_shape).astype(np.float32)
        
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        print("✓ TFLite model validation successful")
        
        # Copy to assets folder (overwrite the previous model)
        assets_dir = Path("app/src/main/assets")
        assets_dir.mkdir(parents=True, exist_ok=True)
        target_path = assets_dir / "yolo.tflite"
        
        shutil.copy2(tflite_path, target_path)
        
        # Check final model size
        final_size = os.path.getsize(target_path) / (1024 * 1024)
        print(f"✓ Tiny model copied to {target_path} ({final_size:.2f} MB)")
        
        return True
        
    except Exception as e:
        print(f"✗ Validation/deployment failed: {e}")
        return False

def main():
    """Main function"""
    print("🔧 Tiny TensorFlow Lite Model Creator")
    print("=" * 40)
    
    # Create tiny TFLite model
    tflite_model = create_tiny_yolo_tflite()
    
    if tflite_model:
        # Validate and deploy
        if validate_and_deploy(tflite_model):
            print("\n🎉 Success! Tiny YOLO model created and deployed!")
            print("This model has the correct I/O structure with minimal size.")
            print("Perfect for testing the Android app's TFLite integration.")
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
