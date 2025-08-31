package com.example.pavo.ui.view

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import androidx.test.core.app.ApplicationProvider
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.Mock
import org.mockito.Mockito.verify
import org.mockito.junit.MockitoJUnit
import org.robolectric.RobolectricTestRunner
import org.mockito.kotlin.any
import org.mockito.kotlin.eq

@RunWith(RobolectricTestRunner::class)
class BoundingBoxOverlayTest {

    @get:Rule
    val mockitoRule = MockitoJUnit.rule()

    @Mock
    private lateinit var mockCanvas: Canvas

    private lateinit var boundingBoxOverlay: BoundingBoxOverlay

    @Before
    fun setUp() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        boundingBoxOverlay = BoundingBoxOverlay(context)
    }

    @Test
    fun `onDraw draws boxes correctly`() {
        // Arrange
        val boxes = listOf(
            Box(RectF(10f, 20f, 30f, 40f), "box1"),
            Box(RectF(50f, 60f, 70f, 80f), "box2")
        )
        boundingBoxOverlay.setBoxes(boxes)

        // Act
        boundingBoxOverlay.draw(mockCanvas)

        // Assert
        verify(mockCanvas).drawRect(eq(RectF(10f, 20f, 30f, 40f)), any())
        verify(mockCanvas).drawText(eq("box1"), eq(10f), eq(10f), any())
        verify(mockCanvas).drawRect(eq(RectF(50f, 60f, 70f, 80f)), any())
        verify(mockCanvas).drawText(eq("box2"), eq(50f), eq(50f), any())
    }
}