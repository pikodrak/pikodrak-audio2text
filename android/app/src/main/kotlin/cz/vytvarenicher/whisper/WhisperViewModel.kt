package cz.vytvarenicher.whisper

import android.app.Application
import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.net.Uri
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File
import java.net.HttpURLConnection
import java.net.URL

private const val TAG = "WhisperViewModel"
private const val SAMPLE_RATE = 16_000
private const val MODEL_URL =
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin"
private const val MODEL_NAME = "ggml-tiny.bin"

// Sliding window: transcribe at most 30 s at a time so inference stays fast on-device
private const val STREAM_WINDOW_SAMPLES = SAMPLE_RATE * 30
private const val STREAM_INTERVAL_MS = 2_000L

sealed class TranscribeState {
    object Idle : TranscribeState()
    data class Transcribing(val step: String = "") : TranscribeState()
    /** Live partial result updated every ~2 s while the microphone is still open. */
    data class Streaming(val text: String) : TranscribeState()
    data class Result(val text: String) : TranscribeState()
    data class Error(val message: String) : TranscribeState()
}

sealed class ModelState {
    object NotDownloaded : ModelState()
    data class Downloading(val progress: Float) : ModelState()
    object Ready : ModelState()
    data class Error(val message: String) : ModelState()
}

class WhisperViewModel(application: Application) : AndroidViewModel(application) {

    private val _transcribeState = MutableStateFlow<TranscribeState>(TranscribeState.Idle)
    val transcribeState: StateFlow<TranscribeState> = _transcribeState

    private val _modelState = MutableStateFlow<ModelState>(ModelState.NotDownloaded)
    val modelState: StateFlow<ModelState> = _modelState

    private val _isRecording = MutableStateFlow(false)
    val isRecording: StateFlow<Boolean> = _isRecording

    private var whisperCtx = 0L

    // whisper_full is NOT thread-safe; serialize every call through this mutex
    private val whisperMutex = Mutex()

    private var audioRecord: AudioRecord? = null
    private val recordedSamples = mutableListOf<Short>()
    private var streamingJob: Job? = null

    private fun modelFile(): File =
        File(getApplication<Application>().filesDir, MODEL_NAME)

    init {
        checkModel()
    }

    private fun checkModel() {
        val f = modelFile()
        if (f.exists() && f.length() > 1_000_000L) {
            _modelState.value = ModelState.Ready
            initWhisper()
        }
    }

    fun downloadModel() {
        viewModelScope.launch(Dispatchers.IO) {
            _modelState.value = ModelState.Downloading(0f)
            val file = modelFile()
            try {
                val conn = URL(MODEL_URL).openConnection() as HttpURLConnection
                conn.connect()
                val total = conn.contentLengthLong
                conn.inputStream.use { input ->
                    file.outputStream().use { out ->
                        val buf = ByteArray(8_192)
                        var read = 0L
                        var n: Int
                        while (input.read(buf).also { n = it } != -1) {
                            out.write(buf, 0, n)
                            read += n
                            if (total > 0) {
                                _modelState.value =
                                    ModelState.Downloading(read.toFloat() / total)
                            }
                        }
                    }
                }
                _modelState.value = ModelState.Ready
                initWhisper()
            } catch (e: Exception) {
                Log.e(TAG, "Download failed", e)
                file.delete()
                _modelState.value = ModelState.Error(e.message ?: "Download failed")
            }
        }
    }

    private fun initWhisper() {
        viewModelScope.launch(Dispatchers.IO) {
            try {
                if (whisperCtx != 0L) WhisperLib.free(whisperCtx)
                whisperCtx = WhisperLib.initFromFile(modelFile().absolutePath)
                if (whisperCtx == 0L) {
                    _modelState.value = ModelState.Error("Failed to load model into whisper.cpp")
                }
            } catch (e: Exception) {
                Log.e(TAG, "initWhisper failed", e)
                _modelState.value = ModelState.Error(e.message ?: "Init failed")
            }
        }
    }

    fun startRecording() {
        if (_isRecording.value) return
        val minBuf = AudioRecord.getMinBufferSize(
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
        )
        val bufSize = maxOf(minBuf, SAMPLE_RATE * 2)
        val ar = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            bufSize,
        )
        audioRecord = ar
        recordedSamples.clear()
        ar.startRecording()
        _isRecording.value = true
        _transcribeState.value = TranscribeState.Idle

        // Continuous capture job
        viewModelScope.launch(Dispatchers.IO) {
            val buf = ShortArray(bufSize / 2)
            while (_isRecording.value) {
                val n = ar.read(buf, 0, buf.size)
                if (n > 0) {
                    synchronized(recordedSamples) {
                        repeat(n) { recordedSamples.add(buf[it]) }
                    }
                }
            }
        }

        // Streaming inference job: run every STREAM_INTERVAL_MS while recording
        streamingJob = viewModelScope.launch(Dispatchers.IO) {
            delay(STREAM_INTERVAL_MS)
            while (_isRecording.value) {
                val snapshot: FloatArray = synchronized(recordedSamples) {
                    val start = maxOf(0, recordedSamples.size - STREAM_WINDOW_SAMPLES)
                    FloatArray(recordedSamples.size - start) { recordedSamples[start + it] / 32_768f }
                }
                if (snapshot.size >= SAMPLE_RATE) { // need at least 1 s of audio
                    whisperMutex.withLock {
                        if (!_isRecording.value) return@withLock // recording ended while we waited
                        try {
                            val text = WhisperLib.transcribe(whisperCtx, snapshot, "auto")
                            _transcribeState.value = TranscribeState.Streaming(text.trim())
                        } catch (e: Exception) {
                            Log.w(TAG, "Streaming inference skipped: ${e.message}")
                        }
                    }
                }
                delay(STREAM_INTERVAL_MS)
            }
        }
    }

    fun stopRecording() {
        _isRecording.value = false
        streamingJob?.cancel()
        streamingJob = null
        audioRecord?.stop()
        audioRecord?.release()
        audioRecord = null

        viewModelScope.launch(Dispatchers.IO) {
            val samples: FloatArray
            synchronized(recordedSamples) {
                samples = FloatArray(recordedSamples.size) { recordedSamples[it] / 32_768f }
            }
            // Final transcription on the full buffer (no 30 s cap)
            runTranscription(samples)
        }
    }

    fun transcribeFile(context: Context, uri: Uri) {
        viewModelScope.launch(Dispatchers.IO) {
            _transcribeState.value = TranscribeState.Transcribing("Decoding audio…")
            try {
                val samples = AudioDecoder.decode(context, uri, SAMPLE_RATE)
                runTranscription(samples)
            } catch (e: Exception) {
                Log.e(TAG, "File decode error", e)
                _transcribeState.value = TranscribeState.Error(e.message ?: "Decode failed")
            }
        }
    }

    private suspend fun runTranscription(samples: FloatArray) {
        if (whisperCtx == 0L) {
            _transcribeState.value = TranscribeState.Error("Model not loaded")
            return
        }
        withContext(Dispatchers.IO) {
            _transcribeState.value = TranscribeState.Transcribing("Running inference…")
            try {
                val text = whisperMutex.withLock {
                    WhisperLib.transcribe(whisperCtx, samples, "auto")
                }
                _transcribeState.value = TranscribeState.Result(text.trim())
            } catch (e: Exception) {
                Log.e(TAG, "Transcription error", e)
                _transcribeState.value = TranscribeState.Error(e.message ?: "Transcription failed")
            }
        }
    }

    override fun onCleared() {
        super.onCleared()
        streamingJob?.cancel()
        if (whisperCtx != 0L) {
            WhisperLib.free(whisperCtx)
            whisperCtx = 0L
        }
        audioRecord?.release()
    }
}
