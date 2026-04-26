package cz.vytvarenicher.whisper.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.unit.dp
import cz.vytvarenicher.whisper.TranscribeState

@Composable
internal fun TranscriptionResult(
    state: TranscribeState,
    onCopy: (String) -> Unit,
) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        when (state) {
            TranscribeState.Idle -> Unit

            is TranscribeState.Transcribing -> {
                CircularProgressIndicator()
                if (state.step.isNotEmpty()) Text(state.step)
            }

            // Live partial result shown while the microphone is still open.
            // Text updates ~every 2 s via sliding-window inference.
            is TranscribeState.Streaming -> {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    Text(
                        "● Live",
                        color = MaterialTheme.colorScheme.error,
                        style = MaterialTheme.typography.labelSmall,
                    )
                    Text(
                        "last 30 s window",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.outline,
                    )
                }
                if (state.text.isNotEmpty()) {
                    OutlinedTextField(
                        value = state.text,
                        onValueChange = {},
                        readOnly = true,
                        label = { Text("Live transcript") },
                        modifier = Modifier.fillMaxWidth(),
                        minLines = 3,
                        // Pending / in-progress text: grey italic to signal it may change
                        textStyle = MaterialTheme.typography.bodyMedium.copy(
                            color = MaterialTheme.colorScheme.outline,
                            fontStyle = FontStyle.Italic,
                        ),
                    )
                }
            }

            is TranscribeState.Result -> {
                OutlinedTextField(
                    value = state.text,
                    onValueChange = {},
                    readOnly = true,
                    label = { Text("Transcription") },
                    modifier = Modifier.fillMaxWidth(),
                    minLines = 5,
                )
                Button(onClick = { onCopy(state.text) }) {
                    Text("Copy to Clipboard")
                }
            }

            is TranscribeState.Error -> {
                Text("Error: ${state.message}", color = MaterialTheme.colorScheme.error)
            }
        }
    }
}
