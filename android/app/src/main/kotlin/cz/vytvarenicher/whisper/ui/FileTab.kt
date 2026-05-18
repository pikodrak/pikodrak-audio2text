package cz.vytvarenicher.whisper.ui

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.unit.dp
import cz.vytvarenicher.whisper.WhisperViewModel

@Composable
fun FileTab(viewModel: WhisperViewModel, modifier: Modifier = Modifier) {
    val context = LocalContext.current
    val transcribeState by viewModel.transcribeState.collectAsState()
    val clipboard = LocalClipboardManager.current

    val filePicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent(),
    ) { uri ->
        uri?.let { viewModel.transcribeFile(context, it) }
    }

    Column(
        modifier = modifier.padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Text("Transcribe Audio File", style = MaterialTheme.typography.titleLarge)
        Text(
            "Supports MP3, WAV, M4A, OGG, and other Android-decodable formats.",
            style = MaterialTheme.typography.bodySmall,
        )

        Button(onClick = { filePicker.launch("audio/*") }) {
            Text("Pick Audio File")
        }

        TranscriptionResult(
            state = transcribeState,
            onCopy = { clipboard.setText(AnnotatedString(it)) },
        )
    }
}
