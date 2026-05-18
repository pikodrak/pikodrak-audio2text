package cz.vytvarenicher.whisper.ui

import android.Manifest
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.unit.dp
import cz.vytvarenicher.whisper.TranscribeState
import cz.vytvarenicher.whisper.WhisperViewModel

@Composable
fun MicTab(viewModel: WhisperViewModel, modifier: Modifier = Modifier) {
    val isRecording by viewModel.isRecording.collectAsState()
    val transcribeState by viewModel.transcribeState.collectAsState()
    val clipboard = LocalClipboardManager.current

    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission(),
    ) { granted ->
        if (granted) viewModel.startRecording()
    }

    val isStreaming = transcribeState is TranscribeState.Streaming

    Column(
        modifier = modifier.padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Text("Record from Microphone", style = MaterialTheme.typography.titleLarge)

        if (!isRecording) {
            Text(
                "Live transcription updates every ~2 s while recording.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.outline,
            )
        }

        Button(
            onClick = {
                if (isRecording) viewModel.stopRecording()
                else permissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
            },
            colors = ButtonDefaults.buttonColors(
                containerColor = if (isRecording) MaterialTheme.colorScheme.error
                else MaterialTheme.colorScheme.primary,
            ),
        ) {
            Text(
                when {
                    isRecording && isStreaming -> "● Recording (live) — Stop"
                    isRecording -> "● Recording… — Stop"
                    else -> "Start Recording"
                },
            )
        }

        TranscriptionResult(
            state = transcribeState,
            onCopy = { clipboard.setText(AnnotatedString(it)) },
        )
    }
}
