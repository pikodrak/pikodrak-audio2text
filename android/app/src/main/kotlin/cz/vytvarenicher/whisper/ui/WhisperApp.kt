package cz.vytvarenicher.whisper.ui

import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import cz.vytvarenicher.whisper.ModelState
import cz.vytvarenicher.whisper.WhisperViewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun WhisperApp(viewModel: WhisperViewModel) {
    val modelState by viewModel.modelState.collectAsState()

    if (modelState !is ModelState.Ready) {
        ModelDownloadScreen(viewModel)
        return
    }

    var selectedTab by remember { mutableIntStateOf(0) }
    val tabs = listOf("Microphone", "File")

    Scaffold(
        topBar = { TopAppBar(title = { Text("Whisper Transcription") }) },
        bottomBar = {
            NavigationBar {
                tabs.forEachIndexed { index, title ->
                    NavigationBarItem(
                        selected = selectedTab == index,
                        onClick = { selectedTab = index },
                        icon = {},
                        label = { Text(title) },
                    )
                }
            }
        },
    ) { innerPadding ->
        when (selectedTab) {
            0 -> MicTab(viewModel, Modifier.fillMaxSize().padding(innerPadding))
            1 -> FileTab(viewModel, Modifier.fillMaxSize().padding(innerPadding))
        }
    }
}
