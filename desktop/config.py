import sys
import os
import json

MODELS = ["tiny", "base", "small", "medium", "large-v3"]
MODEL_HINTS = {
    "tiny":     "~75 MB  |  fastest, lowest latency, least accurate",
    "base":     "~150 MB  |  fast, good for real-time",
    "small":    "~250 MB  |  recommended default",
    "medium":   "~800 MB  |  accurate, needs a strong CPU/GPU for real-time",
    "large-v3": "~3 GB  |  best accuracy, GPU strongly recommended",
}
LANGUAGES = ["auto", "cs", "en", "de", "fr", "es", "it", "pl", "sk"]
INPUT_SOURCES = ["Microphone", "System audio (loopback)", "Audio file"]
MIC_DEFAULT_LABEL = "System default"
BEAM_SIZES = [1, 2, 3, 5, 8]
MAX_SPEAKERS = 6

# Transcription engine: local Whisper+pyannote, or a cloud service with much
# stronger speaker separation. Cloud engines transcribe the whole recording at
# once (on Stop / for a file), which is far more accurate at "who said what".
ENGINE_LOCAL = "Local (Whisper + pyannote)"
ENGINE_GEMINI = "Google Gemini (cloud)"
ENGINE_DEEPGRAM = "Deepgram (real-time cloud)"
ENGINES = [ENGINE_LOCAL, ENGINE_GEMINI, ENGINE_DEEPGRAM]
# Cloud engines don't need the local pyannote model.
CLOUD_ENGINES = (ENGINE_GEMINI, ENGINE_DEEPGRAM)
# Gemini models good for audio + diarization (most capable first as default).
GEMINI_MODELS = ["gemini-2.5-pro", "gemini-2.5-flash"]
# Deepgram streaming model with Czech + live diarization.
DEEPGRAM_MODEL = "nova-2"

# Distinct, readable foreground colors used to tell speakers apart in the GUI.
SPEAKER_COLORS = ["#1565c0", "#c62828", "#2e7d32", "#6a1b9a",
                  "#e65100", "#00838f", "#ad1457", "#4e342e"]

# ── Audio / streaming engine ────────────────────────────────────────────────
SAMPLE_RATE = 16000
CAPTURE_BLOCK_SECS = 0.25     # how often the capture thread drains the audio device
PROCESS_INTERVAL_SECS = 1.0   # how often the streaming buffer is re-transcribed
BUFFER_TRIM_SEC = 15.0        # max unconfirmed audio held before trimming

# faster-whisper VAD tuned per the maintainers' guidance: short min-speech to
# keep brief words, ~400ms silence to close segments promptly without merging
# across pauses, modest padding to avoid silence bleeding into a hallucination.
LIVE_VAD_PARAMS = {"threshold": 0.5, "min_speech_duration_ms": 250,
                   "min_silence_duration_ms": 400, "speech_pad_ms": 200}
FILE_VAD_PARAMS = {"threshold": 0.5, "min_speech_duration_ms": 250,
                   "min_silence_duration_ms": 700, "speech_pad_ms": 200}
# Skip transcribing detected silent gaps longer than this (needs word_timestamps);
# the most reliable cure for "repeated phrase during silence" hallucinations.
HALLUCINATION_SILENCE_SEC = 2.0
# A short, correctly-accented Czech sentence biases Whisper toward Czech +
# diacritics. Used when language is cs and the user hasn't set their own prompt.
DEFAULT_CS_PROMPT = "Tady je přepis rozhovoru v češtině."

# ── Live diarization (whole-recording re-diarization) ────────────────────────
# We diarize the ENTIRE session audio at once (periodically + once on stop) so
# pyannote sees full context and speaker labels stay consistent — much more
# accurate at "who said what" than stitching independent short windows.
DIAR_MIN_SEC = 6.0            # don't diarize until there's at least this much audio
DIAR_INTERVAL_SECS = 8.0      # min gap between live re-diarization passes
DIAR_MAX_SESSION_SEC = 3600   # cap kept audio at 1h to bound memory
DIAR_MODEL_ID = "pyannote/speaker-diarization-3.1"

_KEYRING_SERVICE = "audio2text"
_KEYRING_HF_KEY = "hf_token"
_KEYRING_GEMINI_KEY = "gemini_api_key"
_KEYRING_DEEPGRAM_KEY = "deepgram_api_key"

DEFAULTS = {
    "model": "small",
    "language": "cs",         # default to Czech
    "translate": False,
    "source": "Microphone",
    "mic_device": "",
    "num_speakers": 2,        # 1 = no speaker separation; >=2 enables diarization
    "beam_size": 5,           # used for file transcription (live always greedy)
    "use_vad": True,
    "engine": ENGINE_LOCAL,
    "gemini_model": "gemini-2.5-pro",
    "initial_prompt": "",     # optional Whisper bias (names/terms with diacritics)
    "deepgram_eu": True,      # use Deepgram's EU endpoint (data stays in the EU)
}


def whisper_initial_prompt(language, user_prompt):
    """Pick the Whisper initial_prompt: user's text, else a Czech primer for cs."""
    user_prompt = (user_prompt or "").strip()
    if user_prompt:
        return user_prompt
    if language == "cs":
        return DEFAULT_CS_PROMPT
    return None


def frozen_base_dir():
    """Return the portable base directory for models/, settings.json, and diarize/.

    Windows EXE: the folder containing audio2text-windows.exe.
    macOS .app:  the folder containing Audio2Text.app (sys.executable lives 4
                 levels deep inside the bundle: .app/Contents/MacOS/<binary>).
    """
    exe = sys.executable
    if sys.platform == "darwin":
        base = exe
        for _ in range(4):
            base = os.path.dirname(base)
        return base
    return os.path.dirname(exe)


def model_cache_dir():
    hf_home = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hf_home:
        return hf_home
    if getattr(sys, "frozen", False):
        return os.path.join(frozen_base_dir(), "models")
    return os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")


def whisper_device_and_compute_type():
    """Pick a (device, compute_type) pair faster-whisper can actually load.

    float16 fails on CPU with "Requested float16 compute type, but the target
    device or backend do not support efficient float16 computation". Detect
    CUDA via ctranslate2 (already a faster-whisper dependency, no torch needed).
    """
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


def config_path():
    # Portable mode: bundles store settings.json next to the executable (Windows)
    # or next to the .app bundle (macOS) so the distributable folder is self-contained.
    if getattr(sys, "frozen", False):
        return os.path.join(frozen_base_dir(), "settings.json")
    if sys.platform == "win32":
        base = os.environ.get("APPDATA", os.path.expanduser("~"))
    else:
        base = os.environ.get("XDG_CONFIG_HOME",
                               os.path.join(os.path.expanduser("~"), ".config"))
    return os.path.join(base, "audio2text", "settings.json")


def load_hf_token():
    """Return HF token from keyring, env var, or legacy JSON config."""
    try:
        import keyring
        token = keyring.get_password(_KEYRING_SERVICE, _KEYRING_HF_KEY)
        if token:
            return token
    except Exception:
        pass
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return token
    # Backward compat: read from old plaintext JSON (migration path)
    try:
        with open(config_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("hf_token", "")
    except Exception:
        return ""


def save_hf_token(token):
    """Save HF token to OS keyring. Falls back silently if keyring unavailable."""
    try:
        import keyring
        if token:
            keyring.set_password(_KEYRING_SERVICE, _KEYRING_HF_KEY, token)
        else:
            try:
                keyring.delete_password(_KEYRING_SERVICE, _KEYRING_HF_KEY)
            except Exception:
                pass
        return True
    except Exception:
        return False


def load_gemini_key():
    """Return the Google Gemini API key from keyring or the GEMINI_API_KEY env var."""
    try:
        import keyring
        key = keyring.get_password(_KEYRING_SERVICE, _KEYRING_GEMINI_KEY)
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")


def save_gemini_key(key):
    """Save the Gemini API key to the OS keyring. Falls back silently."""
    try:
        import keyring
        if key:
            keyring.set_password(_KEYRING_SERVICE, _KEYRING_GEMINI_KEY, key)
        else:
            try:
                keyring.delete_password(_KEYRING_SERVICE, _KEYRING_GEMINI_KEY)
            except Exception:
                pass
        return True
    except Exception:
        return False


def load_deepgram_key():
    """Return the Deepgram API key from keyring or the DEEPGRAM_API_KEY env var."""
    try:
        import keyring
        key = keyring.get_password(_KEYRING_SERVICE, _KEYRING_DEEPGRAM_KEY)
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("DEEPGRAM_API_KEY", "")


def save_deepgram_key(key):
    """Save the Deepgram API key to the OS keyring. Falls back silently."""
    try:
        import keyring
        if key:
            keyring.set_password(_KEYRING_SERVICE, _KEYRING_DEEPGRAM_KEY, key)
        else:
            try:
                keyring.delete_password(_KEYRING_SERVICE, _KEYRING_DEEPGRAM_KEY)
            except Exception:
                pass
        return True
    except Exception:
        return False


def load_settings():
    """Load persisted settings, merging with DEFAULTS for missing/invalid keys."""
    data = dict(DEFAULTS)
    try:
        with open(config_path(), "r", encoding="utf-8") as f:
            stored = json.load(f)
        if stored.get("model") in MODELS:
            data["model"] = stored["model"]
        if stored.get("language") in LANGUAGES:
            data["language"] = stored["language"]
        if isinstance(stored.get("translate"), bool):
            data["translate"] = stored["translate"]
        if stored.get("source") in INPUT_SOURCES:
            data["source"] = stored["source"]
        if isinstance(stored.get("mic_device"), str):
            data["mic_device"] = stored["mic_device"]
        n = stored.get("num_speakers")
        if isinstance(n, int):
            data["num_speakers"] = max(1, min(MAX_SPEAKERS, n))
        elif stored.get("diarize") is True:
            # Migrate from the old schema (diarize flag + speaker_mode/count).
            mode = stored.get("speaker_mode")
            if mode == "Two persons":
                data["num_speakers"] = 2
            elif mode == "Custom" and isinstance(stored.get("custom_speaker_count"), int):
                data["num_speakers"] = max(2, min(MAX_SPEAKERS, stored["custom_speaker_count"]))
            else:
                data["num_speakers"] = 2
        if stored.get("beam_size") in BEAM_SIZES:
            data["beam_size"] = stored["beam_size"]
        if isinstance(stored.get("use_vad"), bool):
            data["use_vad"] = stored["use_vad"]
        if stored.get("engine") in ENGINES:
            data["engine"] = stored["engine"]
        if stored.get("gemini_model") in GEMINI_MODELS:
            data["gemini_model"] = stored["gemini_model"]
        if isinstance(stored.get("initial_prompt"), str):
            data["initial_prompt"] = stored["initial_prompt"]
        if isinstance(stored.get("deepgram_eu"), bool):
            data["deepgram_eu"] = stored["deepgram_eu"]
    except Exception:
        pass
    return data


def sanitize_settings(data, *, diarization_available=True):
    """Force LOCAL speaker separation off when pyannote isn't installed.

    Only applies to the local engine — the cloud engine (Gemini) does its own
    diarization and must keep num_speakers>=2 regardless of pyannote.
    """
    if diarization_available or data.get("engine") in CLOUD_ENGINES:
        return data
    if data.get("num_speakers", 1) <= 1:
        return data
    out = dict(data)
    out["num_speakers"] = 1
    return out


def save_settings(data):
    """Persist settings to JSON. HF token is never written here (use keyring)."""
    try:
        path = config_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        safe = {k: v for k, v in data.items() if k != "hf_token"}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(safe, f, indent=2)
    except Exception:
        pass
