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
    from pyannote.audio import Pipeline as _Pipeline
    return _Pipeline.from_pretrained(_MODEL_ID, use_auth_token=hf_token)


def _run_pipeline(pipeline, audio_path, min_speakers=None, max_speakers=None):
    kwargs = {}
    if min_speakers is not None:
        kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers
    diarization = pipeline(audio_path, **kwargs)
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
    import tempfile

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

            audio_bytes = base64.b64decode(audio_b64)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_bytes)
                tmp_path = f.name
            try:
                result = _run_pipeline(pipeline, tmp_path, min_speakers, max_speakers)
                print(json.dumps({"ok": True, "result": result}), flush=True)
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        except Exception as exc:
            print(json.dumps({"ok": False, "error": str(exc)}), flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args:
        _one_shot(args)
    else:
        _persistent()
