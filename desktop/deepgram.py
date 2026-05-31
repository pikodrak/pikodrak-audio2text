"""Real-time transcription + live speaker diarization via Deepgram's streaming
WebSocket API.

The local machine only captures audio and streams 16 kHz mono PCM up; Deepgram
returns word-level results with speaker labels in real time (interim + final).
This is what makes live "who is speaking" work even on a weak PC with no GPU.

Uses websocket-client (sync) on a background thread; nothing here touches the GUI
— results are handed back through the on_words callback.
"""

import json
import time
import threading
import urllib.parse


class DeepgramError(RuntimeError):
    pass


class DeepgramStream:
    """Stream audio to Deepgram and receive diarized words.

    on_words(words, is_final): called from the WS thread. `words` is a list of
        dicts {"start", "end", "text", "speaker"} (speaker is an int). is_final
        marks a committed segment; interim updates have is_final False.
    on_error(message): called on a fatal connection/auth error.
    """

    _HOST_US = "wss://api.deepgram.com/v1/listen"
    _HOST_EU = "wss://api.eu.deepgram.com/v1/listen"

    def __init__(self, api_key, language="cs", sample_rate=16000,
                 on_words=None, on_error=None, eu=True):
        if not api_key:
            raise DeepgramError(
                "Chybí Deepgram API klíč.\n\n"
                "Vytvoř ho zdarma na https://console.deepgram.com "
                "a vlož ho v Advanced → Deepgram API key.")
        self.api_key = api_key
        self.language = language if language and language != "auto" else "cs"
        self.sample_rate = sample_rate
        self.eu = eu                  # EU endpoint → audio stays inside the EU
        self._on_words = on_words
        self._on_error = on_error
        self._ws = None
        self._thread = None
        self._open = threading.Event()
        self._closed = False
        self._fatal = None

    def _url(self):
        params = {
            "model": "nova-2",
            "language": self.language,
            "diarize": "true",
            "interim_results": "true",
            "smart_format": "true",
            "encoding": "linear16",
            "sample_rate": str(self.sample_rate),
            "channels": "1",
            # Privacy: never used to train models; data kept only for the
            # duration of the request (no retention) — for confidential audio.
            "mip_opt_out": "true",
        }
        host = self._HOST_EU if self.eu else self._HOST_US
        return host + "?" + urllib.parse.urlencode(params)

    # ---- lifecycle ------------------------------------------------------

    def start(self, timeout=10):
        import websocket
        self._ws = websocket.WebSocketApp(
            self._url(),
            header=[f"Authorization: Token {self.api_key}"],
            on_open=self._handle_open,
            on_message=self._handle_message,
            on_error=self._handle_error,
            on_close=self._handle_close,
        )
        self._thread = threading.Thread(
            target=self._ws.run_forever, kwargs={"ping_interval": 0}, daemon=True)
        self._thread.start()
        if not self._open.wait(timeout):
            self.stop()
            raise DeepgramError(self._fatal or "Connection to Deepgram timed out.")
        if self._fatal:
            raise DeepgramError(self._fatal)

    def send_audio(self, pcm_bytes):
        """Send a chunk of 16-bit little-endian PCM bytes."""
        if self._ws is None or self._closed:
            return
        try:
            import websocket
            self._ws.send(pcm_bytes, opcode=websocket.ABNF.OPCODE_BINARY)
        except Exception:
            pass  # transient; the WS thread surfaces fatal errors

    def finish(self, timeout=8):
        """Ask Deepgram to flush remaining audio, then close."""
        if self._ws is not None and not self._closed:
            try:
                self._ws.send(json.dumps({"type": "CloseStream"}))
            except Exception:
                pass
            time.sleep(1.0)  # give the final results a moment to arrive
        self.stop()

    def stop(self):
        self._closed = True
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=3)

    # ---- websocket callbacks (run on the WS thread) --------------------

    def _handle_open(self, _ws):
        self._open.set()

    def _handle_error(self, _ws, error):
        msg = str(error)
        low = msg.lower()
        if "401" in low or "403" in low or "unauthorized" in low or "forbidden" in low:
            msg = "Deepgram odmítl API klíč. Zkontroluj klíč v Advanced."
        self._fatal = msg
        self._open.set()  # unblock start()
        if self._on_error:
            self._on_error(msg)

    def _handle_close(self, _ws, *_args):
        self._closed = True
        self._open.set()

    def _handle_message(self, _ws, message):
        try:
            data = json.loads(message)
        except Exception:
            return
        if data.get("type") not in (None, "Results"):
            return  # ignore Metadata / SpeechStarted / UtteranceEnd frames
        alts = (data.get("channel", {}) or {}).get("alternatives", [])
        if not alts:
            return
        raw_words = alts[0].get("words", [])
        if not raw_words:
            return
        words = [{
            "start": float(w.get("start", 0.0)),
            "end": float(w.get("end", 0.0)),
            "text": w.get("punctuated_word") or w.get("word", ""),
            "speaker": int(w.get("speaker", 0)),
        } for w in raw_words]
        if self._on_words:
            self._on_words(words, bool(data.get("is_final")))
