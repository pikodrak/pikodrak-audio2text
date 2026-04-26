import sys
import os
import json

MODELS = ["tiny", "base", "small", "medium", "large-v3"]
MODEL_HINTS = {
    "tiny":     "~75 MB  |  fastest, least accurate",
    "base":     "~150 MB  |  fast, reasonable quality",
    "small":    "~250 MB  |  recommended default",
    "medium":   "~800 MB  |  high accuracy, slower",
    "large-v3": "~3 GB  |  best accuracy, very slow",
}
LANGUAGES = ["auto", "cs", "en", "de", "fr", "es", "it", "pl", "sk"]
INPUT_SOURCES = ["Audio file", "Microphone", "System audio (loopback)"]
CHUNK_OPTIONS = [3, 4, 5, 6, 8, 15, 30]
MIC_DEFAULT_LABEL = "System default"
BEAM_SIZES = [1, 2, 3, 5, 8]
OUTPUT_FORMATS = ["Plain text", "Timestamped"]
SPEAKER_MODES = ["Auto", "Two persons", "Custom"]

SAMPLE_RATE = 16000
CHUNK_DEFAULT = 8
OVERLAP_SECS = 0.5
LIVE_VAD_PARAMS = {"min_speech_duration_ms": 100, "speech_pad_ms": 400}
PREVIEW_INTERVAL_SECS = 2.0
DIAR_MODEL_ID = "pyannote/speaker-diarization-3.1"

_KEYRING_SERVICE = "audio2text"
_KEYRING_HF_KEY = "hf_token"

DEFAULTS = {
    "model": "small",
    "language": "auto",
    "translate": False,
    "source": "Audio file",
    "chunk_seconds": CHUNK_DEFAULT,
    "mic_device": "",
    "diarize": False,
    "beam_size": 5,
    "output_format": "Plain text",
    "speaker_mode": "Auto",
    "custom_speaker_count": 2,
    "speaker_a_name": "Person A",
    "speaker_b_name": "Person B",
}


def model_cache_dir():
    hf_home = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hf_home:
        return hf_home
    if getattr(sys, "frozen", False):
        return os.path.join(os.path.dirname(sys.executable), "models")
    return os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")


def config_path():
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
        if stored.get("chunk_seconds") in CHUNK_OPTIONS:
            data["chunk_seconds"] = stored["chunk_seconds"]
        if isinstance(stored.get("mic_device"), str):
            data["mic_device"] = stored["mic_device"]
        if isinstance(stored.get("diarize"), bool):
            data["diarize"] = stored["diarize"]
        if stored.get("beam_size") in BEAM_SIZES:
            data["beam_size"] = stored["beam_size"]
        if stored.get("output_format") in OUTPUT_FORMATS:
            data["output_format"] = stored["output_format"]
        if stored.get("speaker_mode") in SPEAKER_MODES:
            data["speaker_mode"] = stored["speaker_mode"]
        count = stored.get("custom_speaker_count")
        if isinstance(count, int):
            data["custom_speaker_count"] = max(2, min(10, count))
        for key in ("speaker_a_name", "speaker_b_name"):
            val = stored.get(key)
            if isinstance(val, str) and val.strip():
                data[key] = val.strip()
    except Exception:
        pass
    return data


def sanitize_settings(data, *, diarization_available=True):
    """Return a copy of data with settings that require unavailable features forced off.

    Pass diarization_available=False (EXE mode) to clear any stale diarize=True
    entry before it reaches the UI or transcription code, preventing a pyannote
    import error even when the user has old settings saved from a source-code run.
    """
    if diarization_available:
        return data
    out = dict(data)
    out["diarize"] = False
    out["speaker_mode"] = "Auto"
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
