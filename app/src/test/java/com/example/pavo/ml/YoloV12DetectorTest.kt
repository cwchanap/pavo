package com.example.pavo.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import com.example.pavo.ui.view.Box
import org.junit.Assert.assertEquals
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.Mock
import org.mockito.MockitoAnnotations
import org.mockito.junit.MockitoJUnit
import org.mockito.kotlin.any
import org.mockito.kotlin.doAnswer
import org.mockito.kotlin.whenever
import org.robolectric.RobolectricTestRunner
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import kotlin.math.abs

@RunWith(RobolectricTestRunner::class)
class YoloV12DetectorTest {

    @get:Rule
    val mockitoRule = MockitoJUnit.rule()

    @Mock
    private lateinit var mockContext: Context

    @Mock
    private lateinit var mockInterpreter: Interpreter

    private lateinit var detector: YoloV12Detector

    @Before
    fun setUp() {
        MockitoAnnotations.openMocks(this)
        detector = YoloV12Detector(mockContext)
        
        val interpreterField = YoloV12Detector::class.java.getDeclaredField("interpreter")
        interpreterField.isAccessible = true
        interpreterField.set(detector, mockInterpreter)
    }

    @Test
    fun `detect returns correct boxes for given model output`() {
        // Arrange
        val bitmap = Bitmap.createBitmap(640, 640, Bitmap.Config.ARGB_8888)
        val confidenceThreshold = 0.7f

        val mockOutput = Array(1) { Array(8400) { FloatArray(85) } }
        mockOutput[0][0][0] = 320f 
        mockOutput[0][0][1] = 320f 
        mockOutput[0][0][2] = 100f 
        mockOutput[0][0][3] = 200f 
        mockOutput[0][0][4] = 0.9f 
        mockOutput[0][0][5] = 0.95f

        whenever(mockInterpreter.run(any(), any())).doAnswer { invocation ->
            val outputBuffer = invocation.getArgument<Array<Array<FloatArray>>>(1)
            for (i in mockOutput.indices) {
                for (j in mockOutput[i].indices) {
                    System.arraycopy(mockOutput[i][j], 0, outputBuffer[i][j], 0, mockOutput[i][j].size)
                }
            }
            null
        }

        // Act
        val boxes = detector.detect(bitmap, confidenceThreshold)

        // Assert
        assertEquals(1, boxes.size)
        val box = boxes[0]
        assertEquals(RectF(270f, 220f, 370f, 420f), box.box)
        
        val expectedLabel = "person"
        val expectedConfidence = 0.86f
        val parts = box.text.split(" ")
        assertEquals(expectedLabel, parts[0])
        val actualConfidence = parts[1].toFloat()
        assertEquals(expectedConfidence, actualConfidence, 0.01f)
    }
}
