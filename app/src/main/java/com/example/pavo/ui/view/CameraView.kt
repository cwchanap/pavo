package com.example.pavo.ui.view

import android.Manifest
import android.content.ContentValues
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.os.Build
import android.provider.MediaStore
import android.util.AttributeSet
import android.view.View
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import com.google.accompanist.permissions.PermissionStatus
import com.google.accompanist.permissions.rememberPermissionState
import com.example.pavo.ml.Box
import com.example.pavo.ml.YoloV12Detector
import java.io.File

@OptIn(ExperimentalPermissionsApi::class)
@Composable
fun CameraView(modifier: Modifier = Modifier, onPhotoCaptured: (File) -> Unit) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val cameraPermissionState = rememberPermissionState(Manifest.permission.CAMERA)

    val cameraProviderFuture = remember { ProcessCameraProvider.getInstance(context) }
    val cameraProvider = remember(cameraProviderFuture) { cameraProviderFuture.get() }
    val previewView = remember { PreviewView(context) }
    val boundingBoxOverlay = remember { BoundingBoxOverlay(context) }
    val imageCapture = remember { ImageCapture.Builder().build() }

    val detector = remember {
        YoloV12Detector(context).apply {
            initialize()
        }
    }

    DisposableEffect(detector) {
        onDispose {
            detector.close()
        }
    }

    LaunchedEffect(cameraPermissionState.status) {
        if (cameraPermissionState.status != PermissionStatus.Granted) {
            cameraPermissionState.launchPermissionRequest()
        }
    }

    if (cameraPermissionState.status == PermissionStatus.Granted) {
        Box(modifier = modifier.fillMaxSize()) {
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
                                val bitmap = imageProxy.toBitmap()
                                val boxes = detector.detect(bitmap)
                                boundingBoxOverlay.setBoxes(boxes)
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
                        imageAnalysis,
                        imageCapture
                    )
                } catch (e: Exception) {
                    // handle exception
                }
            }
            AndroidView(
                factory = { boundingBoxOverlay },
                modifier = Modifier.fillMaxSize()
            )
            Button(
                onClick = {
                    val name = "pavo-photo.jpg"
                    val contentValues = ContentValues().apply {
                        put(MediaStore.MediaColumns.DISPLAY_NAME, name)
                        put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
                        if(Build.VERSION.SDK_INT > Build.VERSION_CODES.P) {
                            put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/Pavo-Photos")
                        }
                    }
                    val outputOptions = ImageCapture.OutputFileOptions
                        .Builder(context.contentResolver,
                            MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                            contentValues)
                        .build()
                    imageCapture.takePicture(
                        outputOptions,
                        ContextCompat.getMainExecutor(context),
                        object : ImageCapture.OnImageSavedCallback {
                            override fun onError(exc: ImageCaptureException) {
                                // Handle error
                            }

                            override fun onImageSaved(output: ImageCapture.OutputFileResults){
                                output.savedUri?.let { uri ->
                                    val file = File(context.cacheDir, "pavo-photo.jpg")
                                    context.contentResolver.openInputStream(uri)?.use { inputStream ->
                                        file.outputStream().use { outputStream ->
                                            inputStream.copyTo(outputStream)
                                        }
                                    }
                                    onPhotoCaptured(file)
                                }
                            }
                        }
                    )
                },
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .padding(16.dp)
            ) {
                Text("Take Photo")
            }
        }
    }
}


class BoundingBoxOverlay(context: Context, attrs: AttributeSet? = null) : View(context, attrs) {
    private var boxes = listOf<com.example.pavo.ml.Box>()
    private val paint = Paint().apply {
        color = Color.RED
        style = Paint.Style.STROKE
        strokeWidth = 4f
    }
    private val textPaint = Paint().apply {
        color = Color.RED
        textSize = 40f
    }

    fun setBoxes(boxes: List<com.example.pavo.ml.Box>) {
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

fun ImageProxy.toBitmap(): Bitmap {
    val buffer = planes[0].buffer
    val bytes = ByteArray(buffer.remaining())
    buffer.get(bytes)
    return android.graphics.BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
}