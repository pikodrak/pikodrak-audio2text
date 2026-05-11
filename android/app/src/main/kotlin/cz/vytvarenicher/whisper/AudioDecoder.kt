package cz.vytvarenicher.whisper

import android.content.Context
import android.media.AudioFormat
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import android.os.Build
import java.nio.ByteOrder
import kotlin.math.min

object AudioDecoder {
    fun decode(context: Context, uri: Uri, targetSampleRate: Int): FloatArray {
        val extractor = MediaExtractor()
        var codec: MediaCodec? = null
        try {
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

            // Force 16-bit PCM output. Newer Android codecs may default to
            // PCM_FLOAT, which would corrupt the asShortBuffer() reads below.
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
                format.setInteger(MediaFormat.KEY_PCM_ENCODING, AudioFormat.ENCODING_PCM_16BIT)
            }

            val mime = format.getString(MediaFormat.KEY_MIME)!!
            codec = MediaCodec.createDecoderByType(mime).apply {
                configure(format, null, null, 0)
                start()
            }

            val srcRate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
            val channels = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)

            // Accumulate mono float32 in chunks to avoid boxing and ArrayList resize churn.
            val monoChunks = ArrayList<FloatArray>()
            // Carry samples across output buffers when count isn't divisible by channels.
            var carry = ShortArray(0)
            val info = MediaCodec.BufferInfo()
            var inputDone = false

            while (true) {
                if (!inputDone) {
                    val inIdx = codec.dequeueInputBuffer(10_000L)
                    if (inIdx >= 0) {
                        val inBuf = codec.getInputBuffer(inIdx)!!
                        val n = extractor.readSampleData(inBuf, 0)
                        if (n < 0) {
                            codec.queueInputBuffer(
                                inIdx, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            inputDone = true
                        } else {
                            codec.queueInputBuffer(inIdx, 0, n, extractor.sampleTime, 0)
                            extractor.advance()
                        }
                    }
                }

                val outIdx = codec.dequeueOutputBuffer(info, 10_000L)
                if (outIdx >= 0) {
                    if (info.size > 0) {
                        val buf = codec.getOutputBuffer(outIdx)!!
                        buf.position(info.offset)
                        buf.limit(info.offset + info.size)
                        val shortBuf = buf.order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
                        val shortCount = shortBuf.remaining()
                        val combined = ShortArray(carry.size + shortCount)
                        if (carry.isNotEmpty()) {
                            System.arraycopy(carry, 0, combined, 0, carry.size)
                        }
                        shortBuf.get(combined, carry.size, shortCount)

                        val drainable = (combined.size / channels) * channels
                        if (drainable > 0) {
                            monoChunks.add(downmixToMono(combined, 0, drainable, channels))
                        }
                        // Keep the channel-misaligned tail for the next iteration.
                        val tailLen = combined.size - drainable
                        carry = if (tailLen > 0) combined.copyOfRange(drainable, combined.size)
                                else ShortArray(0)
                    }
                    codec.releaseOutputBuffer(outIdx, false)
                }
                if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) break
            }

            // Flush any leftover. Pad with zeros to a channel boundary.
            if (carry.isNotEmpty()) {
                val padded = if (carry.size % channels == 0) carry
                             else carry.copyOf(carry.size + (channels - carry.size % channels))
                monoChunks.add(downmixToMono(padded, 0, padded.size, channels))
            }

            val totalMono = monoChunks.sumOf { it.size }
            val mono = FloatArray(totalMono)
            var off = 0
            for (c in monoChunks) {
                System.arraycopy(c, 0, mono, off, c.size)
                off += c.size
            }

            return if (srcRate == targetSampleRate) mono
                   else resample(mono, srcRate, targetSampleRate)
        } finally {
            try { codec?.stop() } catch (_: Exception) {}
            try { codec?.release() } catch (_: Exception) {}
            try { extractor.release() } catch (_: Exception) {}
        }
    }

    private fun downmixToMono(samples: ShortArray, offset: Int, count: Int, channels: Int): FloatArray {
        if (channels <= 1) {
            return FloatArray(count) { samples[offset + it] / 32768f }
        }
        val frames = count / channels
        val out = FloatArray(frames)
        var i = offset
        for (f in 0 until frames) {
            var sum = 0
            for (c in 0 until channels) {
                sum += samples[i++].toInt()
            }
            out[f] = (sum.toFloat() / channels) / 32768f
        }
        return out
    }

    /** Linear-interpolated resampling — adequate quality for Whisper's 16 kHz target. */
    private fun resample(input: FloatArray, srcRate: Int, dstRate: Int): FloatArray {
        if (input.isEmpty() || srcRate == dstRate) return input
        val ratio = srcRate.toDouble() / dstRate
        val outSize = (input.size / ratio).toInt()
        val out = FloatArray(outSize)
        for (i in 0 until outSize) {
            val srcPos = i * ratio
            val i0 = srcPos.toInt()
            val i1 = min(i0 + 1, input.size - 1)
            val frac = (srcPos - i0).toFloat()
            out[i] = input[i0] * (1f - frac) + input[i1] * frac
        }
        return out
    }
}
