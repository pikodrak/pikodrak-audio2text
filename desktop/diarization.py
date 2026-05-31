import sys
import os
import io
import json
import glob
import threading
import subprocess

from config import DIAR_MODEL_ID, frozen_base_dir


# Keep the GUI clean: a windowed (no-console) EXE otherwise flashes a black
# console window for every child process (python probe, venv, pip, the diarize
# worker). CREATE_NO_WINDOW suppresses those; it only exists on Windows.
_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _no_window_kwargs():
    return {"creationflags": _CREATE_NO_WINDOW} if sys.platform == "win32" else {}


# ---------------------------------------------------------------------------
# Venv paths
# ---------------------------------------------------------------------------

def _venv_dir():
    """Return <base>/diarize-env when running as frozen bundle, else None."""
    if getattr(sys, "frozen", False):
        return os.path.join(frozen_base_dir(), "diarize-env")
    return None


def _venv_python():
    """Return the venv's Python executable path, or None outside EXE mode."""
    venv = _venv_dir()
    if not venv:
        return None
    if sys.platform == "win32":
        return os.path.join(venv, "Scripts", "python.exe")
    return os.path.join(venv, "bin", "python")


def _diarize_runner_path():
    """Path to audio2text_diarize.py, either beside the EXE or in the source tree."""
    if getattr(sys, "frozen", False):
        return os.path.join(frozen_base_dir(), "audio2text_diarize.py")
    return os.path.join(os.path.dirname(__file__), "audio2text_diarize.py")


def _ensure_runner():
    """Return the runner path, extracted to a CLEAN directory next to the EXE.

    Two traps to avoid, both fixed here:
      • The runner must never be executed from inside _internal/: Python puts the
        script's own directory first on sys.path, so the venv's Python would then
        import the EXE's bundled packages (huggingface_hub, numpy, …) instead of
        the venv's pinned ones — which broke diarization with errors like
        "backports.zstd has no attribute 'ZstdFile'".
      • A stale older copy must not linger, so we overwrite it every launch.
    frozen_base_dir() (the folder holding the EXE) has no Python packages, so the
    venv's site-packages win there.
    """
    dst = _diarize_runner_path()
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        src = os.path.join(meipass, "audio2text_diarize.py")
        if os.path.isfile(src):
            import shutil
            try:
                shutil.copy(src, dst)  # refresh to match this build
            except Exception:
                pass
    if os.path.isfile(dst):
        return dst
    raise FileNotFoundError(f"Diarization runner not found: {dst}")


# ---------------------------------------------------------------------------
# Availability detection
# ---------------------------------------------------------------------------

def venv_ready():
    """Return True when the diarization venv exists and has pyannote installed."""
    if not getattr(sys, "frozen", False):
        # Source mode: check direct import.
        try:
            import pyannote.audio  # noqa: F401
            return True
        except Exception:
            return False
    python = _venv_python()
    if not python or not os.path.isfile(python):
        return False
    venv = _venv_dir()
    patterns = [
        os.path.join(venv, "Lib", "site-packages", "pyannote"),   # Windows venv
        os.path.join(venv, "lib", "python*", "site-packages", "pyannote"),  # Unix venv
    ]
    return any(glob.glob(p) for p in patterns)


DIARIZATION_AVAILABLE = venv_ready()


# ---------------------------------------------------------------------------
# Speaker helpers (file mode)
# ---------------------------------------------------------------------------

def _speaker_kwargs(min_speakers, max_speakers):
    """A fixed count is pyannote's strongest constraint — prefer num_speakers
    when min==max (best accuracy for a known speaker count), else a range."""
    if min_speakers is not None and min_speakers == max_speakers:
        return {"num_speakers": min_speakers}
    kw = {}
    if min_speakers is not None:
        kw["min_speakers"] = min_speakers
    if max_speakers is not None:
        kw["max_speakers"] = max_speakers
    return kw


def normalize_speaker(label, speaker_mode="Auto", two_person_names=None, label_map=None):
    """Map a pyannote speaker label to a display name."""
    if label_map is not None and label in label_map:
        return label_map[label]
    try:
        idx = int(label.split("_")[-1])
    except (ValueError, IndexError):
        name = label
        if label_map is not None:
            label_map[label] = name
        return name
    if speaker_mode == "Two persons" and two_person_names:
        a_name, b_name = two_person_names
        if label_map is not None:
            assigned = set(label_map.values())
            if a_name not in assigned:
                name = a_name
            elif b_name not in assigned:
                name = b_name
            else:
                name = a_name if idx == 0 else b_name
            label_map[label] = name
            return name
        return a_name if idx == 0 else b_name
    name = f"Speaker {idx + 1}"
    if label_map is not None:
        label_map[label] = name
    return name


def assign_speakers(segments, diarization_result, offset=0.0,
                    speaker_mode="Auto", two_person_names=None, label_map=None):
    """Return list of (display_speaker, text) for each non-empty segment.

    diarization_result: list of {"start": float, "end": float, "speaker": str} dicts
    (returned by run_diarize / run_diarize_audio).
    """
    result = []
    for seg in segments:
        if not seg.text.strip():
            continue
        seg_start = offset + seg.start
        seg_end = offset + seg.end
        best_speaker = "Speaker ?"
        best_overlap = 0.0
        for turn in diarization_result:
            overlap = max(0.0, min(seg_end, turn["end"]) - max(seg_start, turn["start"]))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn["speaker"]
        display = normalize_speaker(best_speaker, speaker_mode, two_person_names, label_map)
        result.append((display, seg.text.strip()))
    return result


def format_labeled_segments(labeled):
    """Group consecutive same-speaker segments into one display line each."""
    if not labeled:
        return ""
    lines = []
    cur_speaker, cur_parts = None, []
    for speaker, text in labeled:
        if speaker != cur_speaker:
            if cur_parts:
                lines.append(f"[{cur_speaker}] {' '.join(cur_parts)}")
            cur_speaker, cur_parts = speaker, [text]
        else:
            cur_parts.append(text)
    if cur_parts:
        lines.append(f"[{cur_speaker}] {' '.join(cur_parts)}")
    return "\n".join(lines)


def format_timestamped(segments):
    """Return segments as '[HH:MM:SS] text' lines."""
    lines = []
    for seg in segments:
        if not seg.text.strip():
            continue
        h = int(seg.start // 3600)
        m = int((seg.start % 3600) // 60)
        s = int(seg.start % 60)
        lines.append(f"[{h:02d}:{m:02d}:{s:02d}] {seg.text.strip()}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# System Python finder (for venv creation)
# ---------------------------------------------------------------------------

def _find_python():
    """Find a real Python 3 on PATH — not the frozen EXE, not the Windows Store stub."""
    import shutil
    for candidate in ["python3", "python", "py"]:
        path = shutil.which(candidate)
        if not path:
            continue
        if "WindowsApps" in path:
            continue
        try:
            r = subprocess.run([path, "--version"], capture_output=True, timeout=5,
                               **_no_window_kwargs())
            out = (r.stdout + r.stderr).decode(errors="ignore").strip()
            # "py" launcher prints version on stderr; first line is what we want.
            first_line = out.splitlines()[0] if out else ""
            if r.returncode == 0 and first_line.startswith("Python 3"):
                return path
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Venv setup (EXE mode)
# ---------------------------------------------------------------------------

def setup_venv(progress_callback=None, done_callback=None):
    """Create diarize-env venv and install pyannote.audio (EXE mode only).

    progress_callback(message: str) — called from background thread.
    done_callback(success: bool, error: str | None) — called on completion.
    Both callbacks are called from a daemon thread; callers must marshal to the UI thread.
    """
    def _report(msg):
        if progress_callback:
            progress_callback(msg)

    def _run():
        global DIARIZATION_AVAILABLE
        venv = _venv_dir()
        if not venv:
            if done_callback:
                done_callback(False, "Runtime download is only available in EXE mode")
            return
        try:
            _report("Finding Python…")
            python = _find_python()
            if not python:
                if done_callback:
                    done_callback(
                        False,
                        "Python 3 not found on PATH.\n\n"
                        "Download and install it from python.org "
                        "(use the official installer, not the Microsoft Store version).\n"
                        "During installation, check \"Add Python to PATH\".\n"
                        "Then restart Audio2Text.",
                    )
                return

            _report("Creating virtual environment…")
            r = subprocess.run([python, "-m", "venv", venv],
                               capture_output=True, text=True, **_no_window_kwargs())
            if r.returncode != 0:
                if done_callback:
                    done_callback(False, f"venv creation failed:\n{r.stderr or r.stdout}")
                return

            if sys.platform == "win32":
                pip_exe = os.path.join(venv, "Scripts", "pip.exe")
            else:
                pip_exe = os.path.join(venv, "bin", "pip")

            # Pin a known-good, mutually-compatible stack. The bleeding edge
            # (pyannote.audio 4.x + huggingface_hub 1.x + torch 2.12 + torchcodec
            # + numpy 2.x) is broken on Windows: hub 1.x raises
            # "backports.zstd has no attribute 'ZstdFile'" and mishandles gated
            # auth, and torchcodec needs absent FFmpeg DLLs. pyannote 3.1.1 with
            # torch 2.2.2 and numpy<2 is the tested combo for diarization-3.1.
            torch_ver = "2.2.2"

            # CUDA build of torch for NVIDIA GPUs, else CPU build.
            want_cuda = False
            try:
                import ctranslate2
                want_cuda = ctranslate2.get_cuda_device_count() > 0
            except Exception:
                pass

            torch_pkgs = [f"torch=={torch_ver}", f"torchaudio=={torch_ver}"]
            if want_cuda:
                _report("Downloading CUDA PyTorch (~2.5 GB) — this may take several minutes…")
                r = subprocess.run(
                    [pip_exe, "install", "-q", *torch_pkgs,
                     "--index-url", "https://download.pytorch.org/whl/cu121"],
                    capture_output=True, text=True, **_no_window_kwargs(),
                )
                if r.returncode != 0:
                    want_cuda = False  # fall back to the CPU build below

            if not want_cuda:
                _report("Downloading PyTorch (CPU) — this may take a few minutes…")
                r = subprocess.run(
                    [pip_exe, "install", "-q", *torch_pkgs],
                    capture_output=True, text=True, **_no_window_kwargs(),
                )
                if r.returncode != 0:
                    err = (r.stderr or r.stdout or "").strip()
                    if done_callback:
                        done_callback(False, err or f"pip (torch) exit code {r.returncode}")
                    return

            _report("Downloading pyannote.audio…")
            # Pin huggingface_hub too: 1.x has the "backports.zstd"/httpx breakage,
            # and pyannote 3.1.1 still calls hf_hub_download(use_auth_token=...),
            # which 0.23.4 honors. (Token is also passed via HF_TOKEN env at runtime.)
            r = subprocess.run(
                [pip_exe, "install", "-q", "numpy<2",
                 "huggingface_hub==0.23.4", "pyannote.audio==3.1.1"],
                capture_output=True, text=True, **_no_window_kwargs(),
            )
            if r.returncode != 0:
                err = (r.stderr or r.stdout or "").strip()
                if done_callback:
                    done_callback(False, err or f"pip returned exit code {r.returncode}")
                return

            DIARIZATION_AVAILABLE = True
            if done_callback:
                done_callback(True, None)

        except Exception:
            import traceback
            if done_callback:
                done_callback(False, traceback.format_exc())

    threading.Thread(target=_run, daemon=True).start()


# ---------------------------------------------------------------------------
# Persistent diarization worker (EXE live mode)
# ---------------------------------------------------------------------------

class DiarizeWorker:
    """Persistent subprocess wrapping audio2text_diarize.py for live-mode chunks.

    The venv Python keeps the pyannote pipeline in memory between chunks,
    giving fast per-chunk latency after the initial model load.
    """

    def __init__(self):
        self._proc = None
        self._lock = threading.Lock()

    def start(self, hf_token):
        """Launch the worker subprocess.  hf_token stored for first-chunk pipeline load."""
        venv_py = _venv_python()
        runner = _ensure_runner()
        self._proc = subprocess.Popen(
            [venv_py, runner],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **_no_window_kwargs(),
        )
        self._hf_token = hf_token

    def run_chunk(self, wav_bytes, hf_token, min_speakers=None, max_speakers=None):
        """Send a WAV audio chunk, return list of {start, end, speaker} dicts."""
        import base64
        if self._proc is None or self._proc.poll() is not None:
            raise RuntimeError("Diarization worker is not running")
        req = {
            "hf_token": hf_token,
            "audio_b64": base64.b64encode(wav_bytes).decode(),
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
        }
        line = (json.dumps(req) + "\n").encode()
        with self._lock:
            try:
                self._proc.stdin.write(line)
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise RuntimeError(f"Diarization subprocess died: {exc}") from exc
            resp_line = self._proc.stdout.readline()
        if not resp_line:
            # Subprocess exited; surface stderr to help diagnose pyannote crashes.
            stderr_tail = b""
            try:
                stderr_tail = self._proc.stderr.read() or b""
            except Exception:
                pass
            raise RuntimeError(
                "Diarization subprocess exited unexpectedly: "
                + stderr_tail.decode(errors="ignore").strip()[-500:]
            )
        try:
            resp = json.loads(resp_line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Invalid response from diarization subprocess: {resp_line!r}") from exc
        if not resp.get("ok"):
            raise RuntimeError(resp.get("error", "Diarization subprocess error"))
        return resp["result"]

    def stop(self):
        if self._proc:
            try:
                self._proc.stdin.close()
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
            self._proc = None


# ---------------------------------------------------------------------------
# Diarization entry points (unified interface for ui.py)
# ---------------------------------------------------------------------------

def _numpy_to_wav(audio, sample_rate):
    """Convert float32 numpy array → 16-bit PCM WAV bytes."""
    import wave
    clipped = (audio * 32767).clip(-32768, 32767).astype("int16")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(clipped.tobytes())
    return buf.getvalue()


# Source-mode pipeline cache (not used in EXE mode)
_pipeline_cache = None
_pipeline_token = None


def reset_cache():
    """Force pipeline reload on next call (source mode only)."""
    global _pipeline_cache, _pipeline_token
    _pipeline_cache = None
    _pipeline_token = None


def _ensure_source_pipeline(hf_token):
    global _pipeline_cache, _pipeline_token
    if _pipeline_cache is not None and _pipeline_token == hf_token:
        return _pipeline_cache
    try:
        from pyannote.audio import Pipeline as PyannotePipeline
    except ImportError:
        raise RuntimeError(
            "Speaker diarization requires pyannote.audio.\n\n"
            "Install it with: pip install pyannote.audio"
        )
    if not hf_token:
        raise ValueError(
            "Speaker diarization requires a HuggingFace token.\n\n"
            "Steps:\n"
            "1. Create a free account at huggingface.co\n"
            "2. Accept the model license at:\n"
            "   huggingface.co/pyannote/speaker-diarization-3.1\n"
            "3. Generate a token at huggingface.co/settings/tokens\n"
            "4. Enter the token in Settings → Diarization"
        )
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    try:
        _pipeline_cache = PyannotePipeline.from_pretrained(DIAR_MODEL_ID, token=hf_token)
    except TypeError:
        _pipeline_cache = PyannotePipeline.from_pretrained(DIAR_MODEL_ID, use_auth_token=hf_token)
    try:
        import torch
        if torch.cuda.is_available():
            _pipeline_cache.to(torch.device("cuda"))
    except Exception:
        pass
    _pipeline_token = hf_token
    return _pipeline_cache


def preload_pipeline(hf_token):
    """Pre-load pipeline into memory (source mode only, no-op in EXE mode).

    Call before starting live recording so model load doesn't delay first chunk.
    """
    if not getattr(sys, "frozen", False):
        _ensure_source_pipeline(hf_token)


def run_diarize(audio_path, hf_token, min_speakers=None, max_speakers=None):
    """Run diarization on an audio file.

    Returns list of {"start": float, "end": float, "speaker": str} dicts.
    EXE mode: runs the venv Python subprocess (one-shot).
    Source mode: uses the in-process pyannote pipeline (cached).
    """
    if getattr(sys, "frozen", False):
        venv_py = _venv_python()
        runner = _ensure_runner()
        args = [venv_py, runner, audio_path, hf_token]
        if min_speakers is not None:
            max_s = max_speakers if max_speakers is not None else min_speakers
            args += [str(min_speakers), str(max_s)]
        try:
            result = subprocess.run(args, capture_output=True, text=True,
                                    timeout=900, **_no_window_kwargs())
        except subprocess.TimeoutExpired:
            raise RuntimeError("Diarization timed out (over 15 minutes).")
        if result.returncode != 0:
            raise RuntimeError(f"Diarization failed:\n{result.stderr or result.stdout}")
        return json.loads(result.stdout.strip())
    else:
        pipeline = _ensure_source_pipeline(hf_token)
        diarization = pipeline(audio_path, **_speaker_kwargs(min_speakers, max_speakers))
        return [
            {"start": float(t.start), "end": float(t.end), "speaker": spk}
            for t, _, spk in diarization.itertracks(yield_label=True)
        ]


def run_diarize_audio(audio, sample_rate, hf_token,
                      min_speakers=None, max_speakers=None, worker=None):
    """Run diarization on an in-memory numpy float32 audio array.

    worker: DiarizeWorker instance for EXE live mode (reuses persistent subprocess).
    In source mode, uses the in-process pyannote pipeline (torch tensor path).
    """
    if getattr(sys, "frozen", False):
        wav_bytes = _numpy_to_wav(audio, sample_rate)
        if worker is not None:
            return worker.run_chunk(wav_bytes, hf_token, min_speakers, max_speakers)
        # Fallback: one-shot temp file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            tmp = f.name
        try:
            return run_diarize(tmp, hf_token, min_speakers, max_speakers)
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    else:
        import torch
        import numpy as np
        pipeline = _ensure_source_pipeline(hf_token)
        # Light peak normalization → levels near the embedding model's training
        # distribution (mild accuracy gain on quiet/loud recordings).
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        norm = audio / peak * 0.95 if peak > 1e-4 else audio
        audio_tensor = torch.from_numpy(norm[None, :])
        diarization = pipeline(
            {"waveform": audio_tensor, "sample_rate": sample_rate},
            **_speaker_kwargs(min_speakers, max_speakers))
        return [
            {"start": float(t.start), "end": float(t.end), "speaker": spk}
            for t, _, spk in diarization.itertracks(yield_label=True)
        ]
