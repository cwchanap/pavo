package com.example.pavo.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import androidx.core.graphics.scale
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

data class Box(val box: RectF, val text: String)

class YoloV12Detector(private val context: Context) {
    
    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    private val inputSize = 640 // YOLOv12 typically uses 640x640
    private val numClasses = 80 // COCO dataset has 80 classes
    private val numDetections = 8400 // YOLOv12 output detections
    
    companion object {
        private const val TAG = "YoloV12Detector"
        private const val MODEL_NAME = "yolo.tflite"
        
        val cocoLabels = listOf(
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"
        )
    }
    
    fun initialize(): Boolean {
        return try {
            val model = loadModelFile(MODEL_NAME)
            
            // Check if GPU acceleration is available
            val compatList = CompatibilityList()
            val options = Interpreter.Options().apply {
                if (compatList.isDelegateSupportedOnThisDevice) {
                    gpuDelegate = GpuDelegate()
                    addDelegate(gpuDelegate)
                    Log.d(TAG, "GPU acceleration enabled")
                } else {
                    Log.d(TAG, "GPU acceleration not available, using CPU")
                }
                setNumThreads(4)
            }
            
            interpreter = Interpreter(model, options)
            Log.d(TAG, "YOLOv12 model loaded successfully")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize YOLOv12 detector", e)
            false
        }
    }
    
    fun detect(bitmap: Bitmap, confidenceThreshold: Float = 0.5f): List<Box> {
        val interpreter = this.interpreter ?: return emptyList()
        
        return try {
            // Preprocess image
            val resizedBitmap = bitmap.scale(inputSize, inputSize, true)
            val inputBuffer = convertBitmapToByteBuffer(resizedBitmap)
            
            // Prepare output buffer
            val outputBuffer = Array(1) { Array(numDetections) { FloatArray(4 + 1 + numClasses) } }
            
            // Run inference
            interpreter.run(inputBuffer, outputBuffer)
            
            // Post-process results
            processOutput(outputBuffer[0], confidenceThreshold, bitmap.width, bitmap.height)
        } catch (e: Exception) {
            Log.e(TAG, "Error during detection", e)
            emptyList()
        }
    }
    
    private fun loadModelFile(modelName: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(modelName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
    
    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(4 * inputSize * inputSize * 3)
        byteBuffer.order(ByteOrder.nativeOrder())
        
        val intValues = IntArray(inputSize * inputSize)
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        
        var pixel = 0
        for (i in 0 until inputSize) {
            for (j in 0 until inputSize) {
                val value = intValues[pixel++]
                // Normalize to [0, 1] range
                byteBuffer.putFloat(((value shr 16) and 0xFF) / 255.0f)
                byteBuffer.putFloat(((value shr 8) and 0xFF) / 255.0f)
                byteBuffer.putFloat((value and 0xFF) / 255.0f)
            }
        }
        return byteBuffer
    }
    
    private fun processOutput(
        output: Array<FloatArray>, 
        confidenceThreshold: Float,
        originalWidth: Int,
        originalHeight: Int
    ): List<Box> {
        val detections = mutableListOf<Detection>()
        
        // Parse detections
        for (i in output.indices) {
            val detection = output[i]
            
            // YOLOv12 format: [x_center, y_center, width, height, confidence, class_scores...]
            val centerX = detection[0]
            val centerY = detection[1]
            val width = detection[2]
            val height = detection[3]
            val confidence = detection[4]
            
            if (confidence > confidenceThreshold) {
                // Find best class
                var maxClassScore = 0f
                var bestClassIndex = -1
                
                for (j in 5 until detection.size) {
                    if (detection[j] > maxClassScore) {
                        maxClassScore = detection[j]
                        bestClassIndex = j - 5
                    }
                }
                
                val finalConfidence = confidence * maxClassScore
                if (finalConfidence > confidenceThreshold && bestClassIndex < cocoLabels.size) {
                    detections.add(
                        Detection(
                            centerX, centerY, width, height,
                            finalConfidence, bestClassIndex
                        )
                    )
                }
            }
        }
        
        // Apply Non-Maximum Suppression
        val nmsDetections = applyNMS(detections, 0.4f)
        
        // Convert to Box objects with proper scaling
        return nmsDetections.map { detection ->
            val scaleX = originalWidth.toFloat() / inputSize
            val scaleY = originalHeight.toFloat() / inputSize
            
            val left = (detection.centerX - detection.width / 2) * scaleX
            val top = (detection.centerY - detection.height / 2) * scaleY
            val right = (detection.centerX + detection.width / 2) * scaleX
            val bottom = (detection.centerY + detection.height / 2) * scaleY
            
            val label = cocoLabels[detection.classIndex]
            val text = "$label %.2f".format(detection.confidence)
            
            Box(RectF(left, top, right, bottom), text)
        }
    }
    
    private fun applyNMS(detections: List<Detection>, iouThreshold: Float): List<Detection> {
        val sortedDetections = detections.sortedByDescending { it.confidence }
        val selectedDetections = mutableListOf<Detection>()
        
        for (detection in sortedDetections) {
            var shouldSelect = true
            
            for (selectedDetection in selectedDetections) {
                if (calculateIoU(detection, selectedDetection) > iouThreshold) {
                    shouldSelect = false
                    break
                }
            }
            
            if (shouldSelect) {
                selectedDetections.add(detection)
            }
        }
        
        return selectedDetections
    }
    
    private fun calculateIoU(det1: Detection, det2: Detection): Float {
        val left1 = det1.centerX - det1.width / 2
        val top1 = det1.centerY - det1.height / 2
        val right1 = det1.centerX + det1.width / 2
        val bottom1 = det1.centerY + det1.height / 2
        
        val left2 = det2.centerX - det2.width / 2
        val top2 = det2.centerY - det2.height / 2
        val right2 = det2.centerX + det2.width / 2
        val bottom2 = det2.centerY + det2.height / 2
        
        val intersectionLeft = maxOf(left1, left2)
        val intersectionTop = maxOf(top1, top2)
        val intersectionRight = minOf(right1, right2)
        val intersectionBottom = minOf(bottom1, bottom2)
        
        if (intersectionLeft >= intersectionRight || intersectionTop >= intersectionBottom) {
            return 0f
        }
        
        val intersectionArea = (intersectionRight - intersectionLeft) * (intersectionBottom - intersectionTop)
        val area1 = det1.width * det1.height
        val area2 = det2.width * det2.height
        val unionArea = area1 + area2 - intersectionArea
        
        return intersectionArea / unionArea
    }
    
    fun close() {
        interpreter?.close()
        gpuDelegate?.close()
        interpreter = null
        gpuDelegate = null
    }
    
    private data class Detection(
        val centerX: Float,
        val centerY: Float,
        val width: Float,
        val height: Float,
        val confidence: Float,
        val classIndex: Int
    )
}