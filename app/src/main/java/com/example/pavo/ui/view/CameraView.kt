
package com.example.pavo.ui.view

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import com.google.accompanist.permissions.PermissionStatus
import com.google.accompanist.permissions.rememberPermissionState
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

@OptIn(ExperimentalPermissionsApi::class)
@Composable
fun CameraView() {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val cameraPermissionState = rememberPermissionState(Manifest.permission.CAMERA)

    val cameraProviderFuture = remember { ProcessCameraProvider.getInstance(context) }
    val cameraProvider = remember(cameraProviderFuture) { cameraProviderFuture.get() }
    val previewView = remember { PreviewView(context) }
    val boundingBoxOverlay = remember { BoundingBoxOverlay(context) }

    val interpreter = remember {
        try {
            val model = loadModelFile(context, "yolov12.tflite")
            Interpreter(model)
        } catch (e: Exception) {
            null // Return null if model loading fails
        }
    }

    LaunchedEffect(cameraPermissionState.status) {
        if (cameraPermissionState.status != PermissionStatus.Granted) {
            cameraPermissionState.launchPermissionRequest()
        }
    }

    if (cameraPermissionState.status == PermissionStatus.Granted) {
        Box(modifier = Modifier.fillMaxSize()) {
            AndroidView(
                factory = { previewView },
                modifier = Modifier.fillMaxSize()
            ) {
                val preview = Preview.Builder().build().also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

                val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

                val imageAnalysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                    .also {
                        it.setAnalyzer(ContextCompat.getMainExecutor(context)) { imageProxy ->
                            try {
                                if (interpreter != null) {
                                    val bitmap = imageProxy.toBitmap()
                                    val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 320, 320, true)
                                    val byteBuffer = convertBitmapToByteBuffer(resizedBitmap)

                                    val output = Array(1) { Array(25200) { FloatArray(85) } }
                                    interpreter.run(byteBuffer, output)

                                    val boxes = processOutput(output[0])
                                    boundingBoxOverlay.setBoxes(boxes)
                                }
                            } catch (e: Exception) {
                                // Handle any processing errors gracefully
                            } finally {
                                imageProxy.close()
                            }
                        }
                    }

                try {
                    cameraProvider.unbindAll()
                    cameraProvider.bindToLifecycle(
                        lifecycleOwner,
                        cameraSelector,
                        preview,
                        imageAnalysis
                    )
                } catch (e: Exception) {
                    // handle exception
                }
            }
            AndroidView(
                factory = { boundingBoxOverlay },
                modifier = Modifier.fillMaxSize()
            )
        }
    }
}

private fun loadModelFile(context: Context, modelName: String): MappedByteBuffer {
    val fileDescriptor = context.assets.openFd(modelName)
    val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
    val fileChannel = inputStream.channel
    val startOffset = fileDescriptor.startOffset
    val declaredLength = fileDescriptor.declaredLength
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
}

private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
    val byteBuffer = ByteBuffer.allocateDirect(4 * 320 * 320 * 3)
    byteBuffer.order(ByteOrder.nativeOrder())
    val intValues = IntArray(320 * 320)
    bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
    var pixel = 0
    for (i in 0 until 320) {
        for (j in 0 until 320) {
            val `val` = intValues[pixel++]
            byteBuffer.putFloat(((`val` shr 16) and 0xFF) / 255.0f)
            byteBuffer.putFloat(((`val` shr 8) and 0xFF) / 255.0f)
            byteBuffer.putFloat((`val` and 0xFF) / 255.0f)
        }
    }
    return byteBuffer
}

private fun processOutput(output: Array<FloatArray>): List<Box> {
    val boxes = mutableListOf<Box>()
    for (i in output.indices) {
        val x = output[i][0]
        val y = output[i][1]
        val w = output[i][2]
        val h = output[i][3]
        val confidence = output[i][4]

        if (confidence > 0.5) {
            val left = (x - w / 2) * 320
            val top = (y - h / 2) * 320
            val right = (x + w / 2) * 320
            val bottom = (y + h / 2) * 320

            var maxConfidence = 0f
            var maxIndex = -1
            for (j in 5 until 85) {
                if (output[i][j] > maxConfidence) {
                    maxConfidence = output[i][j]
                    maxIndex = j
                }
            }

            if (maxConfidence > 0.5) {
                boxes.add(Box(RectF(left, top, right, bottom), "${cocoLabels[maxIndex]} $maxConfidence"))
            }
        }
    }
    return boxes
}

class BoundingBoxOverlay(context: Context, attrs: AttributeSet? = null) : View(context, attrs) {
    private var boxes = listOf<Box>()
    private val paint = Paint().apply {
        color = Color.RED
        style = Paint.Style.STROKE
        strokeWidth = 4f
    }
    private val textPaint = Paint().apply {
        color = Color.RED
        textSize = 40f
    }

    fun setBoxes(boxes: List<Box>) {
        this.boxes = boxes
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        boxes.forEach { box ->
            canvas.drawRect(box.box, paint)
            canvas.drawText(box.text, box.box.left, box.box.top - 10, textPaint)
        }
    }
}

data class Box(val box: RectF, val text: String)

fun ImageProxy.toBitmap(): Bitmap {
    val buffer = planes[0].buffer
    val bytes = ByteArray(buffer.remaining())
    buffer.get(bytes)
    return android.graphics.BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
}

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
