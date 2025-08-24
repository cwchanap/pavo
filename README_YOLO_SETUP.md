# YOLOv12 Model Setup for Pavo App

This guide explains how to set up a real YOLOv12 model for object detection in the Pavo Android app.

## Quick Setup

1. **Run the conversion script:**
   ```bash
   python3 download_and_convert_yolo.py
   ```

2. **Build and run the app:**
   ```bash
   ./gradlew build
   ```

## What the Script Does

The `download_and_convert_yolo.py` script automatically:

1. **Downloads YOLOv12s model** from the official repository
2. **Installs required dependencies** (PyTorch, TensorFlow, Ultralytics, etc.)
3. **Converts the model** from PyTorch (.pt) to TensorFlow Lite (.tflite)
4. **Copies the model** to `app/src/main/assets/yolov12.tflite`
5. **Cleans up** temporary files

## Conversion Methods

The script tries two conversion approaches:

### Method 1: Direct Ultralytics Export (Recommended)
- Uses Ultralytics' built-in TensorFlow Lite export
- Simpler and more reliable
- Maintains model accuracy

### Method 2: Full Pipeline (Fallback)
- PyTorch → ONNX → TensorFlow → TensorFlow Lite
- More complex but handles edge cases
- Based on Google's official guide

## Manual Setup (Alternative)

If the script fails, you can manually:

1. **Download the model:**
   ```bash
   wget https://github.com/sunsmarterjie/yolov12/releases/download/turbo/yolov12s.pt
   ```

2. **Install Ultralytics:**
   ```bash
   pip install ultralytics
   ```

3. **Convert to TensorFlow Lite:**
   ```python
   from ultralytics import YOLO
   model = YOLO('yolov12s.pt')
   model.export(format='tflite', imgsz=640)
   ```

4. **Copy to assets:**
   ```bash
   cp yolov12s.tflite app/src/main/assets/yolov12.tflite
   ```

## Model Details

- **Model:** YOLOv12s (Small variant)
- **Input Size:** 640x640 pixels
- **Classes:** 80 COCO dataset classes
- **Format:** TensorFlow Lite (.tflite)
- **Optimization:** Float16 precision for better performance

## Features

The YoloV12Detector class provides:

- ✅ **GPU acceleration** (when available)
- ✅ **Non-Maximum Suppression** (NMS)
- ✅ **Confidence thresholding**
- ✅ **Proper scaling** for different image sizes
- ✅ **Memory management** and cleanup
- ✅ **Error handling** for robustness

## Troubleshooting

### Common Issues:

1. **Download fails:**
   - Check internet connection
   - Try downloading manually from the GitHub link

2. **Conversion fails:**
   - Ensure Python 3.7+ is installed
   - Try installing dependencies manually: `pip install torch ultralytics tensorflow`

3. **Model not found in app:**
   - Verify `app/src/main/assets/yolov12.tflite` exists
   - Check file size (should be ~20-50MB)

4. **App crashes:**
   - Check Android logs for TensorFlow Lite errors
   - Ensure device has sufficient memory

### Performance Tips:

- **GPU acceleration** is automatically enabled on compatible devices
- **Model size** vs **accuracy** tradeoff: YOLOv12s balances both
- **Confidence threshold** can be adjusted in `YoloV12Detector.detect()`

## Next Steps

After successful setup:

1. **Test the app** on a real device with camera
2. **Adjust confidence thresholds** if needed
3. **Customize object classes** for your use case
4. **Optimize performance** based on your requirements

For more information, visit:
- [YOLOv12 Repository](https://github.com/sunsmarterjie/yolov12)
- [Google LiteRT Documentation](https://ai.google.dev/edge/litert)
- [Ultralytics Documentation](https://docs.ultralytics.com/)