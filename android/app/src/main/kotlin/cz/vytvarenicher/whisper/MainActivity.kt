package cz.vytvarenicher.whisper

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import cz.vytvarenicher.whisper.ui.WhisperApp
import cz.vytvarenicher.whisper.ui.theme.WhisperTheme

class MainActivity : ComponentActivity() {
    private val viewModel: WhisperViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            WhisperTheme {
                WhisperApp(viewModel)
            }
        }
    }
}
