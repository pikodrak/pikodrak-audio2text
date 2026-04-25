import sys
import os
import json
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext


MODELS = ["tiny", "base", "small", "medium", "large-v3"]
LANGUAGES = ["auto", "cs", "en", "de", "fr", "es", "it", "pl", "sk"]
INPUT_SOURCES = ["Audio file", "Microphone", "System audio (loopback)"]
CHUNK_OPTIONS = [3, 5, 8, 15, 30]
MIC_DEFAULT_LABEL = "System default"

SAMPLE_RATE = 16000
CHUNK_DEFAULT = 8  # seconds — default buffer size before each Whisper pass
# Carry the last OVERLAP_SECS of each chunk into the next so words that fall at a
# chunk boundary are not silently dropped by the VAD filter.
OVERLAP_SECS = 0.5
# VAD options for live chunks: lower min_speech catches short fragments at boundaries;
# generous speech_pad prevents edge trimming.
_LIVE_VAD_PARAMS = {"min_speech_duration_ms": 100, "speech_pad_ms": 400}

_DIAR_MODEL_ID = "pyannote/speaker-diarization-3.1"


def _model_cache_dir():
    """Return the model cache directory.

    Priority:
    1. HF_HOME / HUGGINGFACE_HUB_CACHE env vars (explicit override)
    2. <exe folder>/models/ when running as a PyInstaller bundle
    3. ~/.cache/huggingface/hub/ when running from source
    """
    hf_home = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hf_home:
        return hf_home
    if getattr(sys, "frozen", False):
        return os.path.join(os.path.dirname(sys.executable), "models")
    return os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")


def _config_path():
    if sys.platform == "win32":
        base = os.environ.get("APPDATA", os.path.expanduser("~"))
    else:
        base = os.environ.get("XDG_CONFIG_HOME",
                               os.path.join(os.path.expanduser("~"), ".config"))
    return os.path.join(base, "audio2text", "settings.json")


# ---------------------------------------------------------------- Diarization helpers --

def _normalize_speaker(label):
    """Convert 'SPEAKER_00' → 'Speaker 1' for display."""
    try:
        num = int(label.split("_")[-1])
        return f"Speaker {num + 1}"
    except (ValueError, IndexError):
        return label


def _assign_speakers(segments, diarization, offset=0.0):
    """Return list of (display_speaker, text) for each non-empty segment.

    Matches each Whisper segment to the diarization turn with the most overlap.
    `offset` shifts segment timestamps when processing a chunk within a longer stream.
    """
    turns = list(diarization.itertracks(yield_label=True))
    result = []
    for seg in segments:
        if not seg.text.strip():
            continue
        seg_start = offset + seg.start
        seg_end = offset + seg.end
        best_speaker = "Speaker ?"
        best_overlap = 0.0
        for turn, _, label in turns:
            overlap = max(0.0, min(seg_end, turn.end) - max(seg_start, turn.start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = label
        result.append((_normalize_speaker(best_speaker), seg.text.strip()))
    return result


def _format_labeled_segments(labeled):
    """Group consecutive same-speaker pairs into one line each."""
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


# ----------------------------------------------------------------------- Tooltip --

class _ToolTip:
    def __init__(self, widget, text):
        self._widget = widget
        self._text = text
        self._tip = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, _event=None):
        x = self._widget.winfo_rootx() + 20
        y = self._widget.winfo_rooty() + self._widget.winfo_height() + 4
        self._tip = tk.Toplevel(self._widget)
        self._tip.wm_overrideredirect(True)
        self._tip.wm_geometry(f"+{x}+{y}")
        ttk.Label(self._tip, text=self._text, background="#ffffe0",
                  relief="solid", borderwidth=1, wraplength=280,
                  padding=(6, 4)).pack()

    def _hide(self, _event=None):
        if self._tip:
            self._tip.destroy()
            self._tip = None


class Audio2TextApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Audio2Text")
        self.geometry("720x630")
        self.resizable(True, True)
        self._recording = False
        self._diar_pipeline = None   # cached pyannote pipeline (loaded on first diarize use)
        self._build_ui()
        self._load_settings()
        self._on_source_change()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------ UI --

    def _build_ui(self):
        # --- Input source ---
        self._src_frame = ttk.LabelFrame(self, text="Input Source", padding=8)
        self._src_frame.pack(fill=tk.X, padx=10, pady=(10, 0))

        self.source_var = tk.StringVar(value="Audio file")
        for src in INPUT_SOURCES:
            ttk.Radiobutton(
                self._src_frame, text=src,
                variable=self.source_var, value=src,
                command=self._on_source_change,
            ).pack(side=tk.LEFT, padx=8)

        # File picker row (hidden when source is not "Audio file")
        self._file_row = ttk.Frame(self, padding=(10, 5, 10, 0))
        self._file_row.pack(fill=tk.X)
        ttk.Label(self._file_row, text="Audio file:").pack(side=tk.LEFT)
        self.file_var = tk.StringVar()
        ttk.Entry(self._file_row, textvariable=self.file_var, width=52).pack(
            side=tk.LEFT, padx=5)
        ttk.Button(self._file_row, text="Browse…", command=self._browse).pack(
            side=tk.LEFT)

        # Microphone device row (hidden unless source is "Microphone")
        self._device_row = ttk.Frame(self, padding=(10, 5, 10, 0))
        # not packed here — _on_source_change manages visibility
        ttk.Label(self._device_row, text="Microphone:").pack(side=tk.LEFT)
        self.mic_var = tk.StringVar(value=MIC_DEFAULT_LABEL)
        self._mic_cb = ttk.Combobox(
            self._device_row, textvariable=self.mic_var, width=44, state="readonly")
        self._mic_cb["values"] = [MIC_DEFAULT_LABEL]
        self._mic_cb.pack(side=tk.LEFT, padx=(5, 3))
        ttk.Button(self._device_row, text="↺", width=2,
                   command=self._refresh_mics).pack(side=tk.LEFT)

        # --- Options ---
        opts = ttk.Frame(self, padding=(10, 6, 10, 6))
        opts.pack(fill=tk.X)

        ttk.Label(opts, text="Model:").pack(side=tk.LEFT)
        self.model_var = tk.StringVar(value="small")
        ttk.Combobox(opts, textvariable=self.model_var, values=MODELS,
                     width=8, state="readonly").pack(side=tk.LEFT, padx=(5, 15))

        ttk.Label(opts, text="Language:").pack(side=tk.LEFT)
        self.lang_var = tk.StringVar(value="auto")
        ttk.Combobox(opts, textvariable=self.lang_var, values=LANGUAGES,
                     width=8, state="readonly").pack(side=tk.LEFT, padx=5)

        self.translate_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="Translate to English",
                        variable=self.translate_var).pack(side=tk.LEFT, padx=(15, 15))

        ttk.Label(opts, text="Chunk:").pack(side=tk.LEFT)
        self.chunk_var = tk.StringVar(value=f"{CHUNK_DEFAULT}s")
        chunk_cb = ttk.Combobox(
            opts, textvariable=self.chunk_var,
            values=[f"{s}s" for s in CHUNK_OPTIONS],
            width=5, state="readonly",
        )
        chunk_cb.pack(side=tk.LEFT, padx=(3, 0))
        _ToolTip(
            chunk_cb,
            "Buffer duration before each Whisper pass.\n"
            "Smaller → lower latency, less context (may cut words).\n"
            "Larger → higher latency, better accuracy.\n"
            "Default: 8s"
        )

        # --- Speaker diarization row ---
        diar_row = ttk.Frame(self, padding=(10, 0, 10, 4))
        diar_row.pack(fill=tk.X)

        self.diarize_var = tk.BooleanVar(value=False)
        diar_cb = ttk.Checkbutton(
            diar_row, text="Speaker diarization",
            variable=self.diarize_var,
            command=self._on_diarize_toggle,
        )
        diar_cb.pack(side=tk.LEFT)
        _ToolTip(
            diar_cb,
            "Label each transcription segment with the speaker ID.\n\n"
            "Requires:\n"
            "  • pip install pyannote.audio\n"
            "  • A free HuggingFace token\n"
            "  • Accept model license at huggingface.co/pyannote/speaker-diarization-3.1\n\n"
            "First run downloads ~1–2 GB. Adds significant processing time.\n"
            "Not available in the bundled EXE (torch excluded)."
        )

        self._hf_token_row = ttk.Frame(diar_row)
        ttk.Label(self._hf_token_row, text="HF Token:").pack(side=tk.LEFT, padx=(15, 3))
        self.hf_token_var = tk.StringVar()
        ttk.Entry(
            self._hf_token_row, textvariable=self.hf_token_var, width=34, show="*"
        ).pack(side=tk.LEFT)
        # _hf_token_row is NOT packed here — _on_diarize_toggle manages visibility

        # --- Action buttons ---
        btn_frame = ttk.Frame(self, padding=(10, 0, 10, 5))
        btn_frame.pack(fill=tk.X)

        self.btn = ttk.Button(btn_frame, text="Transcribe", command=self._start)
        self.btn.pack(side=tk.LEFT)

        self.stop_btn = ttk.Button(btn_frame, text="Stop", command=self._stop,
                                   state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        self.copy_btn = ttk.Button(btn_frame, text="Copy Text", command=self._copy,
                                   state=tk.DISABLED)
        self.copy_btn.pack(side=tk.LEFT, padx=5)

        ttk.Button(btn_frame, text="Clear", command=self._clear).pack(side=tk.LEFT)

        # --- Output ---
        out_frame = ttk.LabelFrame(self, text="Transcription / Log", padding=8)
        out_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 5))
        self.text = scrolledtext.ScrolledText(
            out_frame, wrap=tk.WORD, state=tk.DISABLED, font=("Segoe UI", 10))
        self.text.pack(fill=tk.BOTH, expand=True)
        # Style for info lines
        self.text.tag_config("info", foreground="#888888")
        self.text.tag_config("error_tag", foreground="#cc0000")

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN,
                  anchor=tk.W, padding=(6, 2)).pack(fill=tk.X, side=tk.BOTTOM)

    def _on_source_change(self):
        src = self.source_var.get()
        self._file_row.pack_forget()
        self._device_row.pack_forget()
        if src == "Audio file":
            self._file_row.pack(fill=tk.X, after=self._src_frame)
            self.btn.config(text="Transcribe")
        elif src == "Microphone":
            self._device_row.pack(fill=tk.X, after=self._src_frame)
            self._refresh_mics()
            self.btn.config(text="Start")
        else:
            self.btn.config(text="Start")

    def _on_diarize_toggle(self):
        if self.diarize_var.get():
            self._hf_token_row.pack(side=tk.LEFT)
        else:
            self._hf_token_row.pack_forget()
        # Reset cached pipeline so token changes take effect on next run
        self._diar_pipeline = None

    def _refresh_mics(self):
        """Populate the microphone dropdown with available input devices."""
        try:
            import soundcard as sc
            names = [MIC_DEFAULT_LABEL] + [m.name for m in sc.all_microphones()]
        except Exception:
            names = [MIC_DEFAULT_LABEL]
        current = self.mic_var.get()
        self._mic_cb["values"] = names
        if current not in names:
            self.mic_var.set(MIC_DEFAULT_LABEL)

    # ---------------------------------------------------------- Diarization pipeline --

    def _get_diar_pipeline(self, hf_token):
        """Load (or return cached) pyannote diarization pipeline."""
        if self._diar_pipeline is not None:
            return self._diar_pipeline
        try:
            from pyannote.audio import Pipeline as PyannotePipeline
        except ImportError:
            raise RuntimeError(
                "Speaker diarization requires pyannote.audio.\n\n"
                "Install it with:\n"
                "    pip install pyannote.audio\n\n"
                "Note: this pulls in PyTorch (~2 GB) and is not available\n"
                "in the bundled EXE. Run from source to use diarization."
            )
        if not hf_token:
            raise ValueError(
                "Speaker diarization requires a HuggingFace token.\n\n"
                "Steps:\n"
                "1. Create a free account at huggingface.co\n"
                "2. Accept the model license at:\n"
                "   huggingface.co/pyannote/speaker-diarization-3.1\n"
                "3. Generate a token at huggingface.co/settings/tokens\n"
                "4. Enter the token in the HF Token field"
            )
        self._diar_pipeline = PyannotePipeline.from_pretrained(
            _DIAR_MODEL_ID,
            use_auth_token=hf_token,
        )
        return self._diar_pipeline

    # --------------------------------------------------------------- Settings --

    def _load_settings(self):
        try:
            with open(_config_path(), "r", encoding="utf-8") as f:
                data = json.load(f)
            if "model" in data and data["model"] in MODELS:
                self.model_var.set(data["model"])
            if "language" in data and data["language"] in LANGUAGES:
                self.lang_var.set(data["language"])
            if "translate" in data:
                self.translate_var.set(bool(data["translate"]))
            if "source" in data and data["source"] in INPUT_SOURCES:
                self.source_var.set(data["source"])
            if "chunk_seconds" in data and data["chunk_seconds"] in CHUNK_OPTIONS:
                self.chunk_var.set(f"{data['chunk_seconds']}s")
            if "mic_device" in data and isinstance(data["mic_device"], str):
                self.mic_var.set(data["mic_device"] or MIC_DEFAULT_LABEL)
            if "diarize" in data:
                self.diarize_var.set(bool(data["diarize"]))
                if data.get("diarize"):
                    self._on_diarize_toggle()
            if "hf_token" in data and isinstance(data["hf_token"], str):
                self.hf_token_var.set(data["hf_token"])
        except Exception:
            pass

    def _save_settings(self):
        try:
            path = _config_path()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            mic = self.mic_var.get()
            data = {
                "model": self.model_var.get(),
                "language": self.lang_var.get(),
                "translate": self.translate_var.get(),
                "source": self.source_var.get(),
                "chunk_seconds": self._chunk_seconds(),
                "mic_device": mic if mic != MIC_DEFAULT_LABEL else "",
                "diarize": self.diarize_var.get(),
                "hf_token": self.hf_token_var.get(),
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _chunk_seconds(self):
        """Return chunk size as integer seconds from the UI var (e.g. '8s' → 8)."""
        raw = self.chunk_var.get().rstrip("s")
        try:
            return int(raw)
        except ValueError:
            return CHUNK_DEFAULT

    def _on_close(self):
        self._save_settings()
        self.destroy()

    # --------------------------------------------------------------- Browse --

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[
                ("Audio files", "*.mp3 *.wav *.m4a *.ogg *.flac *.aac *.wma *.opus"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.file_var.set(path)

    # --------------------------------------------------------------- Start / Stop --

    def _start(self):
        src = self.source_var.get()
        if src == "Audio file":
            self._start_file_transcribe()
        else:
            self._start_live(src)

    def _stop(self):
        self._recording = False
        self.stop_btn.config(state=tk.DISABLED)
        self.status_var.set("Stopping…")

    # --------------------------------------------------------------- File transcription --

    def _start_file_transcribe(self):
        path = self.file_var.get().strip()
        if not path:
            messagebox.showwarning("No file", "Please select an audio file first.")
            return
        if not os.path.isfile(path):
            messagebox.showerror("File not found", f"Cannot find:\n{path}")
            return
        self.btn.config(state=tk.DISABLED)
        self.copy_btn.config(state=tk.DISABLED)
        self._set_text("")
        self.status_var.set("Loading model…")
        threading.Thread(target=self._transcribe_file, args=(path,), daemon=True).start()

    def _transcribe_file(self, path):
        try:
            from faster_whisper import WhisperModel
            model_name = self.model_var.get()
            lang = self.lang_var.get()
            if lang == "auto":
                lang = None
            task = "translate" if self.translate_var.get() else "transcribe"
            cache = _model_cache_dir()
            diarize = self.diarize_var.get()
            hf_token = self.hf_token_var.get().strip() or os.environ.get("HF_TOKEN", "")

            self.after(0, self.status_var.set,
                       f"Loading '{model_name}' model — cache: {cache}")
            model = WhisperModel(model_name, compute_type="float16", download_root=cache)
            self.after(0, self.status_var.set, "Transcribing…")
            segments, info = model.transcribe(
                path, language=lang, beam_size=5, task=task, vad_filter=True)
            segments = list(segments)  # materialize generator before diarization

            if diarize:
                self.after(0, self.status_var.set,
                           "Loading diarization model… (first run downloads ~1–2 GB)")
                pipeline = self._get_diar_pipeline(hf_token)
                self.after(0, self.status_var.set, "Running speaker diarization…")
                diarization = pipeline(path)
                labeled = _assign_speakers(segments, diarization)
                result = _format_labeled_segments(labeled)
            else:
                result = " ".join(seg.text.strip() for seg in segments)

            self.after(0, self._on_file_done, result, info.language)
        except Exception as exc:
            self.after(0, self._on_error, str(exc))

    def _on_file_done(self, text, lang):
        self._set_text(text)
        self.status_var.set(f"Done — detected language: {lang}")
        self.btn.config(state=tk.NORMAL)
        self.copy_btn.config(state=tk.NORMAL)

    # --------------------------------------------------------------- Live capture --

    def _start_live(self, source):
        self._recording = True
        self.btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.copy_btn.config(state=tk.DISABLED)
        self._set_text("")
        self.status_var.set("Starting…")
        chunk_secs = self._chunk_seconds()
        threading.Thread(
            target=self._live_loop, args=(source, chunk_secs), daemon=True).start()

    def _live_loop(self, source, chunk_secs):
        try:
            import numpy as np
            from faster_whisper import WhisperModel

            model_name = self.model_var.get()
            lang = self.lang_var.get() if self.lang_var.get() != "auto" else None
            task = "translate" if self.translate_var.get() else "transcribe"
            cache = _model_cache_dir()
            diarize = self.diarize_var.get()
            hf_token = self.hf_token_var.get().strip() or os.environ.get("HF_TOKEN", "")

            model_sizes = {"tiny": "~75 MB", "base": "~150 MB", "small": "~250 MB",
                           "medium": "~800 MB", "large-v3": "~3 GB"}
            model_size_hint = model_sizes.get(model_name, "")
            diar_note = "  |  diarization: ON" if diarize else ""
            self.after(0, self._log_info, (
                f"Source:     {source}\n"
                f"Model:      {model_name} {model_size_hint}  |  task: {task}  |  lang: {lang or 'auto'}{diar_note}\n"
                f"Model cache: {cache}\n"
                f"Chunk size: {chunk_secs}s  |  sample rate: {SAMPLE_RATE} Hz\n"
                "─────────────────────────────────────────\n"
                f"Loading model… (first run downloads {model_size_hint}, please wait)\n"
            ))
            self.after(0, self.status_var.set,
                       f"Loading '{model_name}' model… (cache: {cache})")

            model = WhisperModel(model_name, compute_type="float16", download_root=cache)

            diar_pipeline = None
            if diarize:
                self.after(0, self._log_info,
                           "Loading diarization model… (first run downloads ~1–2 GB)\n")
                self.after(0, self.status_var.set, "Loading diarization model…")
                diar_pipeline = self._get_diar_pipeline(hf_token)
                self.after(0, self._log_info, "Diarization model loaded.\n")

            self.after(0, self._log_info, "Model loaded. Opening audio device…\n")
            self.after(0, self.status_var.set, "Model loaded — opening audio device…")

            if source == "System audio (loopback)":
                self._capture_loopback(model, lang, task, chunk_secs, diarize, diar_pipeline)
            else:
                self._capture_microphone(model, lang, task, chunk_secs, diarize, diar_pipeline)

        except ImportError as exc:
            msg = (
                f"Missing dependency: {exc}\n\n"
                "The bundled EXE should include soundcard. If you are running from source:\n"
                "    pip install soundcard numpy"
            )
            self.after(0, self._on_error, msg)
        except Exception as exc:
            self.after(0, self._on_error, str(exc))
        finally:
            self._recording = False
            self.after(0, self._on_live_stopped)

    def _capture_loopback(self, model, lang, task, chunk_secs,
                          diarize=False, diar_pipeline=None):
        import soundcard as sc

        try:
            default_speaker = sc.default_speaker()
        except Exception as exc:
            raise RuntimeError(
                f"Cannot find a default audio output device.\n\n"
                f"soundcard error: {exc}\n\n"
                "Make sure Windows has a default playback device set:\n"
                "  Start → Settings → Sound → Output"
            ) from exc

        speaker_name = str(default_speaker.name)
        try:
            loopback = sc.get_microphone(id=speaker_name, include_loopback=True)
        except Exception as exc:
            raise RuntimeError(
                f"Cannot open WASAPI loopback on '{speaker_name}'.\n\n"
                f"soundcard error: {exc}\n\n"
                "Possible fixes:\n"
                "  • Update your audio driver (Realtek, IDT, etc.)\n"
                "  • Control Panel → Sound → Recording → enable 'Stereo Mix'\n"
                "  • Some USB headsets block WASAPI loopback — try built-in speakers"
            ) from exc

        self.after(0, self._log_info,
                   f"Loopback device: {speaker_name}\nRecording… (speak during your call)\n")
        self.after(0, self.status_var.set,
                   f"Recording loopback: {speaker_name}")
        self._run_capture(loopback, model, lang, task, chunk_secs, diarize, diar_pipeline)

    def _capture_microphone(self, model, lang, task, chunk_secs,
                            diarize=False, diar_pipeline=None):
        import soundcard as sc

        selected = self.mic_var.get()
        try:
            if not selected or selected == MIC_DEFAULT_LABEL:
                mic = sc.default_microphone()
            else:
                mic = sc.get_microphone(id=selected)
        except Exception as exc:
            device_label = selected if selected and selected != MIC_DEFAULT_LABEL else "default microphone"
            raise RuntimeError(
                f"Cannot open {device_label}.\n\n"
                f"soundcard error: {exc}\n\n"
                "Make sure a microphone is connected and set as the default recording device:\n"
                "  Start → Settings → Sound → Input\n\n"
                "If the issue persists, try selecting a specific device from the Microphone dropdown."
            ) from exc

        self.after(0, self._log_info,
                   f"Microphone: {mic.name}\nRecording…\n")
        self.after(0, self.status_var.set, f"Recording microphone: {mic.name}")
        self._run_capture(mic, model, lang, task, chunk_secs, diarize, diar_pipeline)

    def _run_capture(self, device, model, lang, task, chunk_secs,
                     diarize=False, diar_pipeline=None):
        import numpy as np

        chunk_frames = SAMPLE_RATE * chunk_secs
        chunk_num = 0
        overlap = np.zeros(0, dtype="float32")

        with device.recorder(samplerate=SAMPLE_RATE, channels=1) as recorder:
            buffer = []
            while self._recording:
                data = recorder.record(numframes=SAMPLE_RATE)
                if data.ndim > 1:
                    data = data.mean(axis=1)
                data = data.astype("float32")
                buffer.append(data)

                total = sum(len(d) for d in buffer)
                secs_buffered = total / SAMPLE_RATE
                self.after(0, self.status_var.set,
                           f"Recording… {secs_buffered:.0f}/{chunk_secs}s buffered")

                if total >= chunk_frames:
                    chunk_num += 1
                    new_audio = np.concatenate(buffer)
                    buffer = []
                    chunk = np.concatenate([overlap, new_audio]) if len(overlap) else new_audio
                    overlap_secs = len(overlap) / SAMPLE_RATE
                    overlap = new_audio[-int(SAMPLE_RATE * OVERLAP_SECS):]
                    # Normalize: boost quiet audio and prevent clipping
                    peak = float(np.max(np.abs(chunk)))
                    if peak > 1e-6:
                        chunk = chunk / peak * 0.95
                    self.after(0, self.status_var.set,
                               f"Processing chunk #{chunk_num}…")
                    self._process_chunk(model, chunk, lang, task, chunk_num, overlap_secs,
                                        diarize, diar_pipeline)

        # Flush remaining audio when stopped
        if buffer:
            remaining = np.concatenate(buffer)
            if len(remaining) > SAMPLE_RATE // 2:
                chunk_num += 1
                chunk = np.concatenate([overlap, remaining]) if len(overlap) else remaining
                overlap_secs = len(overlap) / SAMPLE_RATE
                peak = float(np.max(np.abs(chunk)))
                if peak > 1e-6:
                    chunk = chunk / peak * 0.95
                self.after(0, self.status_var.set, "Processing final chunk…")
                self._process_chunk(model, chunk, lang, task, chunk_num, overlap_secs,
                                    diarize, diar_pipeline)

    def _process_chunk(self, model, audio, lang, task, chunk_num=0, overlap_secs=0.0,
                       diarize=False, diar_pipeline=None):
        try:
            segments, info = model.transcribe(
                audio, language=lang, beam_size=5, task=task,
                vad_filter=True, vad_parameters=_LIVE_VAD_PARAMS)
            # Skip segments that START within the overlap zone — they were already
            # emitted by the previous chunk.  Segments starting at or after the
            # overlap boundary are genuinely new content.
            valid_segs = [seg for seg in segments if seg.start >= overlap_secs]

            if diarize and diar_pipeline is not None and valid_segs:
                import torch
                audio_tensor = torch.from_numpy(audio[None, :])
                diarization = diar_pipeline({
                    "waveform": audio_tensor,
                    "sample_rate": SAMPLE_RATE,
                })
                labeled = _assign_speakers(valid_segs, diarization)
                text = _format_labeled_segments(labeled)
                if text:
                    self.after(0, lambda t=text: self._append_text(t, as_block=True))
                    self.after(0, self.status_var.set,
                               f"Chunk #{chunk_num} done — detected: {info.language} | recording…")
                else:
                    self.after(0, self.status_var.set,
                               f"Chunk #{chunk_num}: no speech detected | recording…")
            else:
                text = " ".join(seg.text.strip() for seg in valid_segs)
                if text:
                    self.after(0, self._append_text, text)
                    self.after(0, self.status_var.set,
                               f"Chunk #{chunk_num} done — detected: {info.language} | recording…")
                else:
                    self.after(0, self.status_var.set,
                               f"Chunk #{chunk_num}: no speech detected | recording…")
        except Exception as exc:
            self.after(0, self.status_var.set, f"Chunk #{chunk_num} error: {exc}")

    def _on_live_stopped(self):
        self.btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.copy_btn.config(state=tk.NORMAL if self._has_text() else tk.DISABLED)
        self.status_var.set("Stopped")
        self._log_info("─────────────────────────────────────────\nStopped.\n")

    # --------------------------------------------------------------- Error --

    def _on_error(self, msg):
        self.status_var.set("Error — see output area")
        self._log_error(f"ERROR:\n{msg}\n")
        messagebox.showerror("Error", msg)
        self.btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    # --------------------------------------------------------------- Text helpers --

    def _log_info(self, text):
        self.text.config(state=tk.NORMAL)
        self.text.insert(tk.END, text, "info")
        self.text.see(tk.END)
        self.text.config(state=tk.DISABLED)

    def _log_error(self, text):
        self.text.config(state=tk.NORMAL)
        self.text.insert(tk.END, text, "error_tag")
        self.text.see(tk.END)
        self.text.config(state=tk.DISABLED)

    def _set_text(self, text):
        self.text.config(state=tk.NORMAL)
        self.text.delete("1.0", tk.END)
        if text:
            self.text.insert("1.0", text)
        self.text.config(state=tk.DISABLED)

    def _append_text(self, text, as_block=False):
        self.text.config(state=tk.NORMAL)
        if as_block:
            # Speaker-labeled output: separate from previous content with a blank line
            existing = self.text.get("1.0", "end-1c")
            if existing.strip():
                self.text.insert(tk.END, "\n\n")
            self.text.insert(tk.END, text)
        else:
            # Flowing prose: space-separate consecutive chunks
            tail = self.text.get("end-2c", "end-1c")
            if tail and tail not in ("\n", " "):
                self.text.insert(tk.END, " ")
            self.text.insert(tk.END, text)
        self.text.see(tk.END)
        self.text.config(state=tk.DISABLED)
        self.copy_btn.config(state=tk.NORMAL)

    def _has_text(self):
        return bool(self.text.get("1.0", tk.END).strip())

    def _clear(self):
        self._set_text("")
        self.copy_btn.config(state=tk.DISABLED)
        self.status_var.set("Ready")

    def _copy(self):
        text = self.text.get("1.0", tk.END).strip()
        self.clipboard_clear()
        self.clipboard_append(text)
        self.status_var.set("Copied to clipboard")


if __name__ == "__main__":
    app = Audio2TextApp()
    app.mainloop()
