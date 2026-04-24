package cz.vytvarenicher.whisper

object WhisperLib {
    init {
        System.loadLibrary("whisper-jni")
    }

    external fun initFromFile(modelPath: String): Long
    external fun transcribe(ctxPtr: Long, samples: FloatArray, language: String): String
    external fun free(ctxPtr: Long)
    external fun systemInfo(): String
}
