package cz.vytvarenicher.whisper

import android.content.Context
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import java.nio.ByteOrder

object AudioDecoder {
    fun decode(context: Context, uri: Uri, targetSampleRate: Int): FloatArray {
        val extractor = MediaExtractor()
        extractor.setDataSource(context, uri, null)

        var audioTrack = -1
        var format: MediaFormat? = null
        for (i in 0 until extractor.trackCount) {
            val fmt = extractor.getTrackFormat(i)
            val mime = fmt.getString(MediaFormat.KEY_MIME) ?: continue
            if (mime.startsWith("audio/")) {
                audioTrack = i
                format = fmt
                break
            }
        }
        checkNotNull(format) { "No audio track found in file" }
        extractor.selectTrack(audioTrack)

        val mime = format.getString(MediaFormat.KEY_MIME)!!
        val codec = MediaCodec.createDecoderByType(mime)
        codec.configure(format, null, null, 0)
        codec.start()

        val pcmShorts = ArrayList<Short>(1 shl 20)
        val info = MediaCodec.BufferInfo()
        var inputDone = false

        while (true) {
            if (!inputDone) {
                val inIdx = codec.dequeueInputBuffer(10_000L)
                if (inIdx >= 0) {
                    val inBuf = codec.getInputBuffer(inIdx)!!
                    val n = extractor.readSampleData(inBuf, 0)
                    if (n < 0) {
                        codec.queueInputBuffer(inIdx, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                        inputDone = true
                    } else {
                        codec.queueInputBuffer(inIdx, 0, n, extractor.sampleTime, 0)
                        extractor.advance()
                    }
                }
            }

            val outIdx = codec.dequeueOutputBuffer(info, 10_000L)
            if (outIdx >= 0) {
                val buf = codec.getOutputBuffer(outIdx)!!
                val shortBuf = buf.order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
                while (shortBuf.hasRemaining()) pcmShorts.add(shortBuf.get())
                codec.releaseOutputBuffer(outIdx, false)
            }
            if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) break
        }

        codec.stop()
        codec.release()
        extractor.release()

        val srcRate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
        val channels = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)

        // Mix down to mono
        val mono: List<Short> = if (channels > 1) {
            pcmShorts.chunked(channels) { it[0] }
        } else {
            pcmShorts
        }

        return if (srcRate == targetSampleRate) {
            FloatArray(mono.size) { mono[it] / 32768f }
        } else {
            // Linear resampling
            val ratio = srcRate.toDouble() / targetSampleRate
            val outSize = (mono.size / ratio).toInt()
            FloatArray(outSize) { i ->
                val srcIdx = (i * ratio).toInt().coerceAtMost(mono.size - 1)
                mono[srcIdx] / 32768f
            }
        }
    }
}
