
package com.example.pavo.ui.view

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import coil.compose.rememberImagePainter
import java.io.File
import java.text.SimpleDateFormat
import java.util.*

@Composable
fun CameraView() {
    val context = LocalContext.current
    var hasCameraPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED
        )
    }
    val cameraPermissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        hasCameraPermission = isGranted
    }

    var imageUri by remember { mutableStateOf<Uri?>(null) }
    val cameraLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.TakePicture()
    ) { success ->
        if (success) {
            // The image is saved to the specified URI
        }
    }

    Box(modifier = Modifier.fillMaxSize()) {
        if (hasCameraPermission) {
            if (imageUri == null) {
                Button(onClick = {
                    val uri = createImageUri(context)
                    imageUri = uri
                    cameraLauncher.launch(uri)
                }) {
                    Text("Take Photo")
                }
            } else {
                Image(
                    painter = rememberImagePainter(imageUri),
                    contentDescription = "Captured Image",
                    modifier = Modifier.fillMaxSize()
                )
            }
        } else {
            Button(onClick = { cameraPermissionLauncher.launch(Manifest.permission.CAMERA) }) {
                Text("Request Camera Permission")
            }
        }
    }
}

private fun createImageUri(context: Context): Uri {
    val name = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
    val file = File(context.filesDir, "$name.jpg")
    return FileProvider.getUriForFile(context, "${context.packageName}.provider", file)
}
