package cz.vytvarenicher.whisper.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import cz.vytvarenicher.whisper.ModelState
import cz.vytvarenicher.whisper.WhisperViewModel

@Composable
fun ModelDownloadScreen(viewModel: WhisperViewModel) {
    val modelState by viewModel.modelState.collectAsState()

    Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Column(
            modifier = Modifier.padding(32.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            Text("Whisper Transcription", style = MaterialTheme.typography.headlineMedium)
            Text(
                "The ggml-tiny.bin model (~75 MB) is required for offline transcription. " +
                    "It will be downloaded once and stored on the device.",
                style = MaterialTheme.typography.bodyMedium,
            )

            when (val s = modelState) {
                is ModelState.NotDownloaded -> {
                    Button(onClick = { viewModel.downloadModel() }) {
                        Text("Download Model")
                    }
                }

                is ModelState.Downloading -> {
                    Text("Downloading… ${(s.progress * 100).toInt()}%")
                    LinearProgressIndicator(
                        progress = { s.progress },
                        modifier = Modifier.fillMaxWidth(),
                    )
                }

                is ModelState.Error -> {
                    Text("Error: ${s.message}", color = MaterialTheme.colorScheme.error)
                    Button(onClick = { viewModel.downloadModel() }) { Text("Retry") }
                }

                ModelState.Ready -> Unit
            }
        }
    }
}
