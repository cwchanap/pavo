# Gemini Code Assistant Context

## Project Overview

This project, named "Pavo," is an Android application that utilizes a YOLOv12 model for real-time object detection. The application is built using Kotlin and Jetpack Compose. The project also includes a suite of Python scripts for downloading, converting, and managing the YOLOv12 model, which is then used by the Android app.

The core functionality of the project is to perform object detection on the device, using the device's camera as the input source. The YOLOv12 model is converted to the TensorFlow Lite (`.tflite`) format for efficient execution on mobile devices.

### Key Technologies

*   **Android:**
    *   Kotlin
    *   Jetpack Compose for the UI
    *   CameraX for camera integration
    *   TensorFlow Lite for on-device machine learning
*   **Model Preparation (Python):**
    *   PyTorch
    *   Ultralytics YOLO
    *   TensorFlow
    *   ONNX

## Building and Running

### 1. Model Preparation

Before building the Android application, the YOLOv12 model must be downloaded and converted to the TensorFlow Lite format. The project provides a Python script to automate this process.

```bash
# Navigate to the scripts directory
cd scripts

# Run the conversion script
python3 download_and_convert_yolo.py
```

This script will download the YOLOv12 model, convert it to `.tflite`, and place it in the `app/src/main/assets/` directory.

### 2. Building the Android App

Once the model is in place, you can build the Android application using Gradle.

```bash
# From the project root directory
./gradlew build
```

### 3. Running the Android App

The application can be run on an Android device or emulator through Android Studio or by using the following Gradle command:

```bash
./gradlew installDebug
```

## Development Conventions

*   **Android:** The Android application follows standard Android development practices. The UI is built with Jetpack Compose, and the architecture appears to be a single-Activity structure.
*   **Python:** The Python scripts are well-structured and include error handling and logging. They are designed to be run from the command line.
*   **Model:** The project uses a YOLOv12s model, which is a small and efficient variant suitable for mobile devices. The model is converted to TensorFlow Lite with optimizations for performance.
