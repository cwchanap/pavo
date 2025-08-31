#!/usr/bin/env python3
"""
Create a small, optimized TensorFlow Lite model for testing Android app integration
"""

import os
import sys
import numpy as np
from pathlib import Path

def create_small_yolo_tflite():
    """Create a small TFLite model with YOLO-like structure"""
    print("Creating small TFLite model with YOLO structure...")
    
    try:
        import tensorflow as tf
        
        # Create a much smaller model for testing
        # Input: (640, 640, 3) - height, width, channels
        # Output: (84, 8400) - classes+coords, detections
        
        inputs = tf.keras.Input(shape=(640, 640, 3), name='input')
        
        # Much smaller feature extraction (to reduce model size)
        x = tf.keras.layers.Conv2D(8, 7, strides=8, padding='same', activation='relu')(inputs)  # 80x80x8
        x = tf.keras.layers.Conv2D(16, 5, strides=4, padding='same', activation='relu')(x)      # 20x20x16
        x = tf.keras.layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')(x)      # 10x10x32
        x = tf.keras.layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)      # 5x5x64
        
        # Global average pooling to get fixed size output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)  # 64 features
        
        # Small dense layer to get to YOLO output size
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dense(84 * 8400)(x)
        outputs = tf.keras.layers.Reshape((84, 8400))(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Print model summary
        model.summary()
        
        # Convert to TFLite with optimization
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # Use int8 quantization for smaller size
        converter.target_spec.supported_types = [tf.int8]
        
        # Generate representative dataset for quantization
        def representative_dataset():
            for _ in range(100):
                yield [np.random.uniform(0.0, 1.0, size=(1, 640, 640, 3)).astype(np.float32)]
        
        converter.representative_dataset = representative_dataset
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        tflite_model = converter.convert()
        
        # Save the model
        tflite_path = "yolo_small.tflite"
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        model_size = len(tflite_model) / (1024 * 1024)
        print(f"✓ Small TFLite model created: {tflite_path} ({model_size:.2f} MB)")
        print("⚠️  Note: This is a lightweight model for testing - not the actual YOLO weights")
        
        return tflite_path
        
    except Exception as e:
        print(f"✗ Failed to create small model: {e}")
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
        
        # Test inference with correct input type
        input_shape = input_details[0]['shape']
        input_dtype = input_details[0]['dtype']
        
        if input_dtype == np.int8:
            # For int8 models, input should be in range [-128, 127]
            test_input = np.random.randint(-128, 127, size=input_shape, dtype=np.int8)
        else:
            test_input = np.random.uniform(0.0, 1.0, size=input_shape).astype(np.float32)
        
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        print("✓ TFLite model validation successful")
        
        # Copy to assets folder (overwrite the large model)
        assets_dir = Path("app/src/main/assets")
        assets_dir.mkdir(parents=True, exist_ok=True)
        target_path = assets_dir / "yolo.tflite"
        
        shutil.copy2(tflite_path, target_path)
        
        # Check final model size
        final_size = os.path.getsize(target_path) / (1024 * 1024)
        print(f"✓ Optimized model copied to {target_path} ({final_size:.2f} MB)")
        
        return True
        
    except Exception as e:
        print(f"✗ Validation/deployment failed: {e}")
        return False

def main():
    """Main function"""
    print("🔧 Optimized TensorFlow Lite Model Creator")
    print("=" * 45)
    
    # Create small TFLite model
    tflite_model = create_small_yolo_tflite()
    
    if tflite_model:
        # Validate and deploy
        if validate_and_deploy(tflite_model):
            print("\n🎉 Success! Optimized YOLO model created and deployed!")
            print("This model has the correct structure with efficient size.")
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
