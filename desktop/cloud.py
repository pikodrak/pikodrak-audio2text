"""Cloud transcription engines with strong speaker separation.

Currently implements Google Gemini (google AI Studio API key). Gemini transcribes
the WHOLE recording at once and labels speakers, which is dramatically better at
"who said what" than real-time local diarization — ideal for interviews/podcasts.

No third-party SDK: everything goes through the REST API over urllib, so nothing
extra needs bundling. Large recordings use the Gemini Files API (upload → poll →
generate) so multi-minute audio works.
"""

import io
import json
import time
import wave
import urllib.request
import urllib.error

_BASE = "https://generativelanguage.googleapis.com"
_LANG_NAMES = {
    "cs": "češtině", "en": "angličtině", "de": "němčině", "fr": "francouzštině",
    "es": "španělštině", "it": "italštině", "pl": "polštině", "sk": "slovenštině",
}
# Split long audio into <= this many seconds per request: a full-hour verbatim
# transcript can exceed Gemini's output-token cap and get truncated, and very
# long single inputs hurt diarization. ~15 min keeps output well within limits.
CHUNK_SECONDS = 900
MAX_OUTPUT_TOKENS = 65536


class CloudError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def numpy_to_wav_bytes(audio, sample_rate):
    """float32 mono numpy array -> 16-bit PCM WAV bytes."""
    clipped = (audio * 32767).clip(-32768, 32767).astype("int16")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(clipped.tobytes())
    return buf.getvalue()


def _system_instruction(language, num_speakers):
    lang = _LANG_NAMES.get(language, "daném jazyce")
    if num_speakers <= 1:
        return (
            f"Jsi precizní přepisovač. Přepiš nahrávku DOSLOVNĚ a přesně v {lang}; "
            "NIKDY nepřekládej. Zachovej diakritiku a vlastní jména. Nevracej "
            "žádný úvod ani časové značky — pouze přepis.")
    n = num_speakers
    return (
        f"Jsi precizní přepisovač. Přepisuješ nahrávky rozhovoru v {lang} "
        f"s {n} mluvčími. Pravidla:\n"
        f"- Přepisuj DOSLOVNĚ a přesně v {lang}; NIKDY nepřekládej do jiného jazyka.\n"
        "- Zachovej správnou diakritiku a vlastní jména.\n"
        f"- V nahrávce jsou právě {n} mluvčí. Označuj je 'Mluvčí 1' až "
        f"'Mluvčí {n}'. Mluvčí 1 = první hlas, který je slyšet. Drž stejné "
        "označení pro stejného člověka v CELÉM přepisu, nikdy nepřidávej další.\n"
        "- Při změně mluvčího začni nový řádek ve formátu 'Mluvčí X: text'.\n"
        "- Hlídej i změnu mluvčího uprostřed věty.\n"
        "- Nevracej žádný úvod, vysvětlení ani časové značky — pouze přepis."
    )


def _user_prompt(context):
    if context:
        return ("Toto navazuje na předchozí část stejné nahrávky. Pokračuj v "
                "přepisu a ZACHOVEJ STEJNÉ označení mluvčích. Konec předchozí "
                "části (jen pro kontext, znovu ho neopisuj):\n" + context
                + "\n\nPřepiš tuto navazující část:")
    return "Přepiš tuto nahrávku podle pravidel:"


# ---------------------------------------------------------------------------
# Gemini REST
# ---------------------------------------------------------------------------

class GeminiProvider:
    def __init__(self, api_key, model="gemini-2.5-pro"):
        if not api_key:
            raise CloudError(
                "Chybí Google Gemini API klíč.\n\n"
                "Získej ho zdarma na https://aistudio.google.com/apikey "
                "a vlož ho v Advanced → Gemini API key.")
        self.api_key = api_key
        self.model = model

    # ---- low-level HTTP -------------------------------------------------

    def _req(self, url, data=None, headers=None, method=None):
        req = urllib.request.Request(url, data=data, headers=headers or {},
                                     method=method)
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                return resp.status, dict(resp.headers), resp.read()
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            raise CloudError(self._explain(e.code, body))
        except urllib.error.URLError as e:
            raise CloudError(f"Síťová chyba při volání Gemini: {e.reason}")

    @staticmethod
    def _explain(code, body):
        msg = body
        try:
            msg = json.loads(body).get("error", {}).get("message", body)
        except Exception:
            pass
        if code in (401, 403):
            return ("Gemini odmítl API klíč (HTTP %d). Zkontroluj klíč v Advanced "
                    "a že je pro projekt povolené Generative Language API.\n\n%s"
                    % (code, msg[:300]))
        if code == 429:
            return ("Gemini: překročen limit/kvóta (HTTP 429). Zkus to za chvíli, "
                    "nebo zvyš kvótu v Google AI Studiu.\n\n%s" % msg[:300])
        return "Gemini HTTP %d: %s" % (code, msg[:400])

    # ---- Files API ------------------------------------------------------

    def _upload(self, audio_bytes, mime_type, progress=None):
        # 1) start a resumable upload session
        start_url = f"{_BASE}/upload/v1beta/files?key={self.api_key}"
        start_headers = {
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(len(audio_bytes)),
            "X-Goog-Upload-Header-Content-Type": mime_type,
            "Content-Type": "application/json",
        }
        body = json.dumps({"file": {"display_name": "audio2text-recording"}}).encode()
        status, headers, _ = self._req(start_url, data=body, headers=start_headers,
                                       method="POST")
        upload_url = headers.get("X-Goog-Upload-URL") or headers.get("x-goog-upload-url")
        if not upload_url:
            raise CloudError("Gemini Files API nevrátilo upload URL.")

        # 2) upload the bytes and finalize
        if progress:
            progress("Nahrávám audio do Gemini…")
        up_headers = {
            "Content-Length": str(len(audio_bytes)),
            "X-Goog-Upload-Offset": "0",
            "X-Goog-Upload-Command": "upload, finalize",
            "Content-Type": mime_type,
        }
        _, _, resp = self._req(upload_url, data=audio_bytes, headers=up_headers,
                               method="POST")
        info = json.loads(resp.decode("utf-8", "replace"))
        f = info.get("file", info)
        return f.get("uri"), f.get("name"), f.get("state", "PROCESSING")

    def _wait_active(self, name, progress=None, timeout=180):
        """Poll until the uploaded file finishes processing (state ACTIVE)."""
        url = f"{_BASE}/v1beta/{name}?key={self.api_key}"
        waited = 0
        while waited < timeout:
            _, _, resp = self._req(url, method="GET")
            state = json.loads(resp.decode("utf-8", "replace")).get("state")
            if state == "ACTIVE":
                return
            if state == "FAILED":
                raise CloudError("Gemini se nepodařilo zpracovat nahrané audio.")
            if progress:
                progress("Gemini zpracovává audio…")
            time.sleep(2)
            waited += 2
        raise CloudError("Gemini: zpracování audia trvalo příliš dlouho.")

    # ---- generation -----------------------------------------------------

    def _generate(self, wav_bytes, language, num_speakers, context, progress):
        """Upload one WAV chunk and return its diarized transcript."""
        uri, name, state = self._upload(wav_bytes, "audio/wav", progress)
        if state != "ACTIVE":
            self._wait_active(name, progress)
        if progress:
            progress("Gemini přepisuje a rozlišuje mluvčí…")
        gen_url = f"{_BASE}/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "systemInstruction": {
                "parts": [{"text": _system_instruction(language, num_speakers)}]},
            "contents": [{
                "parts": [
                    {"text": _user_prompt(context)},
                    {"file_data": {"mime_type": "audio/wav", "file_uri": uri}},
                ]
            }],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": MAX_OUTPUT_TOKENS,  # avoid truncating long audio
            },
        }
        _, _, resp = self._req(
            gen_url, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        data = json.loads(resp.decode("utf-8", "replace"))
        cands = data.get("candidates") or []
        if not cands:
            fb = data.get("promptFeedback", {})
            raise CloudError("Gemini nevrátil žádný výsledek. %s" % json.dumps(fb)[:200])
        parts = cands[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts).strip()

    # ---- public API -----------------------------------------------------

    def transcribe(self, audio, sample_rate, *, language="cs", num_speakers=2,
                   progress=None):
        """Transcribe a mono float32 numpy array, chunking long audio.

        Long recordings are split into <= CHUNK_SECONDS pieces; each later piece
        gets the tail of the previous transcript as context so speaker labels
        stay consistent across chunk boundaries.
        """
        total = len(audio)
        chunk = int(CHUNK_SECONDS * sample_rate)
        if total <= chunk:
            wav = numpy_to_wav_bytes(audio, sample_rate)
            text = self._generate(wav, language, num_speakers, None, progress)
            if not text:
                raise CloudError("Gemini vrátil prázdný přepis.")
            return text

        n_chunks = (total + chunk - 1) // chunk
        parts, context = [], None
        for idx in range(n_chunks):
            seg = audio[idx * chunk:(idx + 1) * chunk]
            if progress:
                progress(f"Gemini: úsek {idx + 1}/{n_chunks}…")
            wav = numpy_to_wav_bytes(seg, sample_rate)
            text = self._generate(wav, language, num_speakers, context, progress)
            if text:
                parts.append(text)
                context = "\n".join(text.splitlines()[-3:])  # anchor speaker labels
        if not parts:
            raise CloudError("Gemini vrátil prázdný přepis.")
        return "\n".join(parts)


def transcribe_gemini(api_key, model, audio, sample_rate, *,
                      language="cs", num_speakers=2, progress=None):
    """Convenience wrapper used by the UI. `audio` is a mono float32 numpy array."""
    return GeminiProvider(api_key, model).transcribe(
        audio, sample_rate, language=language, num_speakers=num_speakers,
        progress=progress)
