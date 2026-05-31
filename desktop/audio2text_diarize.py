"""Diarization worker — run by the venv Python, not the frozen EXE.

Two operating modes:

One-shot (file path on argv):
    python audio2text_diarize.py <audio_path> <hf_token> [min_speakers max_speakers]
    Prints a JSON array to stdout and exits.

Persistent (no argv):
    Reads JSON request lines from stdin, writes JSON response lines to stdout.
    Keeps the pyannote pipeline loaded between requests for fast live-mode chunks.

Request  (persistent): {"hf_token": str, "audio_b64": str,
                         "min_speakers": int|null, "max_speakers": int|null}
Response (persistent): {"ok": true,  "result": [{"start": f, "end": f, "speaker": str}, ...]}
                     | {"ok": false, "error": str}
"""

import sys
import os
import json


_MODEL_ID = "pyannote/speaker-diarization-3.1"


def _load_pipeline(hf_token):
    # Pass the token via the environment too: across pyannote/huggingface_hub
    # versions the kwarg name flips between use_auth_token and token (and some
    # hub versions silently ignore it), but HF_TOKEN is always honored.
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    from pyannote.audio import Pipeline as _Pipeline
    try:
        pipeline = _Pipeline.from_pretrained(_MODEL_ID, token=hf_token)
    except TypeError:
        pipeline = _Pipeline.from_pretrained(_MODEL_ID, use_auth_token=hf_token)
    # Run on the GPU when a CUDA build of torch + an NVIDIA device are present.
    try:
        import torch
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
    except Exception:
        pass
    return pipeline


def _load_wav_tensor(source):
    """Decode a 16-bit PCM WAV (path OR raw bytes) with the stdlib.

    Newer pyannote pulls in torchcodec, which needs FFmpeg shared libraries that
    aren't present; letting pyannote load the file itself then fails. We always
    produce plain 16-bit PCM WAV upstream, so decode it ourselves and hand
    pyannote a ready waveform tensor — sidestepping the audio backend entirely.
    Accepting bytes lets the live worker skip the temp-file round-trip per chunk.
    """
    import io
    import wave
    import numpy as np
    import torch
    wav_in = io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source
    with wave.open(wav_in, "rb") as wf:
        sr = wf.getframerate()
        channels = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
    data = np.frombuffer(raw, dtype=np.int16).astype("float32") / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    # Light peak normalization keeps levels near the embedding model's training
    # distribution (mildly improves accuracy on quiet/loud recordings).
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak > 1e-4:
        data = data / peak * 0.95
    waveform = torch.from_numpy(data).unsqueeze(0)  # shape (1, num_samples)
    return waveform, sr


def _speaker_kwargs(min_speakers, max_speakers):
    """A fixed count is the strongest constraint for pyannote — prefer
    num_speakers when min==max, else pass the range."""
    if min_speakers is not None and min_speakers == max_speakers:
        return {"num_speakers": min_speakers}
    kw = {}
    if min_speakers is not None:
        kw["min_speakers"] = min_speakers
    if max_speakers is not None:
        kw["max_speakers"] = max_speakers
    return kw


def _run_pipeline(pipeline, source, min_speakers=None, max_speakers=None):
    """source: a WAV path (str) or raw WAV bytes."""
    kwargs = _speaker_kwargs(min_speakers, max_speakers)
    try:
        waveform, sr = _load_wav_tensor(source)
        diarization = pipeline({"waveform": waveform, "sample_rate": sr}, **kwargs)
    except Exception:
        if isinstance(source, (bytes, bytearray)):
            raise  # no file path to fall back to
        # Fall back to letting pyannote open the file (e.g. non-PCM input).
        diarization = pipeline(source, **kwargs)
    return [
        {"start": float(t.start), "end": float(t.end), "speaker": spk}
        for t, _, spk in diarization.itertracks(yield_label=True)
    ]


def _one_shot(argv):
    audio_path = argv[0]
    hf_token = argv[1]
    min_speakers = int(argv[2]) if len(argv) > 2 else None
    max_speakers = int(argv[3]) if len(argv) > 3 else None
    pipeline = _load_pipeline(hf_token)
    result = _run_pipeline(pipeline, audio_path, min_speakers, max_speakers)
    print(json.dumps(result))


def _persistent():
    import base64

    pipeline = None
    pipeline_token = None

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            hf_token = req.get("hf_token", "")
            min_speakers = req.get("min_speakers")
            max_speakers = req.get("max_speakers")
            audio_b64 = req.get("audio_b64", "")

            if pipeline is None or pipeline_token != hf_token:
                pipeline = _load_pipeline(hf_token)
                pipeline_token = hf_token

            # Decode the WAV in memory — no temp-file round-trip per chunk.
            audio_bytes = base64.b64decode(audio_b64)
            result = _run_pipeline(pipeline, audio_bytes, min_speakers, max_speakers)
            print(json.dumps({"ok": True, "result": result}), flush=True)
        except Exception as exc:
            print(json.dumps({"ok": False, "error": str(exc)}), flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args:
        _one_shot(args)
    else:
        _persistent()
