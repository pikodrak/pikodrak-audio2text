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

/**
 * Grow-on-demand ShortArray buffer. Avoids the per-sample boxing overhead of
 * `MutableList<Short>` — for a 10-minute recording at 16 kHz this is the
 * difference between ~19 MB of primitive shorts and ~150 MB+ of boxed Short
 * objects plus ArrayList resize churn.
 */
private class ShortBuffer(initialCapacity: Int = 16_000) {
    private var data: ShortArray = ShortArray(initialCapacity)
    var size: Int = 0
        private set

    @Synchronized
    fun append(src: ShortArray, len: Int) {
        ensureCapacity(size + len)
        System.arraycopy(src, 0, data, size, len)
        size += len
    }

    @Synchronized
    fun snapshot(fromIndex: Int = 0): ShortArray {
        val n = size - fromIndex
        val out = ShortArray(n)
        System.arraycopy(data, fromIndex, out, 0, n)
        return out
    }

    @Synchronized
    fun clear() {
        size = 0
    }

    private fun ensureCapacity(required: Int) {
        if (required <= data.size) return
        var newCap = data.size
        while (newCap < required) newCap = (newCap * 2).coerceAtLeast(required)
        data = data.copyOf(newCap)
    }
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
    private val recordedSamples = ShortBuffer()
    private var streamingJob: Job? = null
    private var captureJob: Job? = null

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
            var conn: HttpURLConnection? = null
            try {
                conn = (URL(MODEL_URL).openConnection() as HttpURLConnection).apply {
                    connectTimeout = 30_000
                    readTimeout = 60_000
                    instanceFollowRedirects = true
                }
                conn.connect()
                if (conn.responseCode !in 200..299) {
                    throw java.io.IOException(
                        "HTTP ${conn.responseCode} ${conn.responseMessage ?: ""}")
                }
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
                if (file.length() < 1_000_000L) {
                    throw java.io.IOException(
                        "Downloaded file too small (${file.length()} B) — server likely returned an error page")
                }
                _modelState.value = ModelState.Ready
                initWhisper()
            } catch (e: Exception) {
                Log.e(TAG, "Download failed", e)
                file.delete()
                _modelState.value = ModelState.Error(e.message ?: "Download failed")
            } finally {
                conn?.disconnect()
            }
        }
    }

    private fun initWhisper() {
        viewModelScope.launch(Dispatchers.IO) {
            try {
                whisperMutex.withLock {
                    if (whisperCtx != 0L) {
                        WhisperLib.free(whisperCtx)
                        whisperCtx = 0L
                    }
                    val ctx = WhisperLib.initFromFile(modelFile().absolutePath)
                    if (ctx == 0L) {
                        _modelState.value = ModelState.Error("Failed to load model into whisper.cpp")
                    } else {
                        whisperCtx = ctx
                    }
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
        if (minBuf <= 0) {
            _transcribeState.value = TranscribeState.Error(
                "AudioRecord.getMinBufferSize returned $minBuf — recording not supported on this device")
            return
        }
        val bufSize = maxOf(minBuf, SAMPLE_RATE * 2 /* 1 s of 16-bit mono */)
        val ar = try {
            AudioRecord(
                MediaRecorder.AudioSource.MIC,
                SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                bufSize,
            )
        } catch (e: Exception) {
            _transcribeState.value = TranscribeState.Error("Cannot open microphone: ${e.message}")
            return
        }
        if (ar.state != AudioRecord.STATE_INITIALIZED) {
            ar.release()
            _transcribeState.value = TranscribeState.Error(
                "AudioRecord failed to initialize (state=${ar.state}) — check RECORD_AUDIO permission")
            return
        }
        audioRecord = ar
        recordedSamples.clear()
        ar.startRecording()
        _isRecording.value = true
        _transcribeState.value = TranscribeState.Idle

        captureJob = viewModelScope.launch(Dispatchers.IO) {
            val buf = ShortArray(bufSize / 2)
            while (_isRecording.value) {
                val n = ar.read(buf, 0, buf.size)
                if (n > 0) {
                    recordedSamples.append(buf, n)
                } else if (n < 0) {
                    Log.w(TAG, "AudioRecord.read returned $n — stopping capture")
                    break
                }
            }
        }

        streamingJob = viewModelScope.launch(Dispatchers.IO) {
            delay(STREAM_INTERVAL_MS)
            while (_isRecording.value) {
                val all = recordedSamples.snapshot()
                val start = maxOf(0, all.size - STREAM_WINDOW_SAMPLES)
                val window = ShortArray(all.size - start)
                System.arraycopy(all, start, window, 0, window.size)
                val snapshot = FloatArray(window.size) { window[it] / 32_768f }

                if (snapshot.size >= SAMPLE_RATE) { // need ≥1 s
                    whisperMutex.withLock {
                        if (!_isRecording.value || whisperCtx == 0L) return@withLock
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
        if (!_isRecording.value) return
        _isRecording.value = false
        streamingJob?.cancel()
        streamingJob = null
        captureJob?.cancel()
        captureJob = null
        try { audioRecord?.stop() } catch (_: Exception) {}
        audioRecord?.release()
        audioRecord = null

        viewModelScope.launch(Dispatchers.IO) {
            val raw = recordedSamples.snapshot()
            val samples = FloatArray(raw.size) { raw[it] / 32_768f }
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
        if (samples.isEmpty()) {
            _transcribeState.value = TranscribeState.Error("No audio to transcribe")
            return
        }
        if (whisperCtx == 0L) {
            _transcribeState.value = TranscribeState.Error("Model not loaded")
            return
        }
        withContext(Dispatchers.IO) {
            _transcribeState.value = TranscribeState.Transcribing("Running inference…")
            try {
                val text = whisperMutex.withLock {
                    if (whisperCtx == 0L) ""
                    else WhisperLib.transcribe(whisperCtx, samples, "auto")
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
        _isRecording.value = false
        streamingJob?.cancel()
        captureJob?.cancel()
        try { audioRecord?.stop() } catch (_: Exception) {}
        audioRecord?.release()
        audioRecord = null
        viewModelScope.launch(Dispatchers.IO) {
            whisperMutex.withLock {
                if (whisperCtx != 0L) {
                    WhisperLib.free(whisperCtx)
                    whisperCtx = 0L
                }
            }
        }
    }
}
