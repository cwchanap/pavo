package com.example.pavo

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Scaffold
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.example.pavo.ui.theme.PavoTheme
import com.example.pavo.ui.view.CameraView

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            PavoTheme {
                val navController = rememberNavController()
                var classificationResult by remember { mutableStateOf<String?>(null) }
                val imageClassifier = remember { ImageClassifier(applicationContext) }
                var bitmap by remember { mutableStateOf<Bitmap?>(null) }

                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    NavHost(
                        navController = navController,
                        startDestination = "camera",
                        modifier = Modifier.padding(innerPadding)
                    ) {
                        composable("camera") {
                            CameraView(onPhotoCaptured = {
                                val decodedBitmap = BitmapFactory.decodeFile(it.path)
                                bitmap = decodedBitmap
                                navController.navigate("post_processing")
                            })
                        }
                        composable("post_processing") {
                            bitmap?.let {
                                PostProcessingScreen(
                                    bitmap = it,
                                    onClassifyClick = {
                                        classificationResult = imageClassifier.classify(it)
                                    },
                                    classificationResult = classificationResult
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}