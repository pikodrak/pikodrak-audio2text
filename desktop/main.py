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

SAMPLE_RATE = 16000
CHUNK_DEFAULT = 8  # seconds — default buffer size before each Whisper pass


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
        self.geometry("720x600")
        self.resizable(True, True)
        self._recording = False
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
        if src == "Audio file":
            self._file_row.pack(fill=tk.X, after=self._src_frame)
            self.btn.config(text="Transcribe")
        else:
            self._file_row.pack_forget()
            self.btn.config(text="Start")

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
        except Exception:
            pass

    def _save_settings(self):
        try:
            path = _config_path()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            data = {
                "model": self.model_var.get(),
                "language": self.lang_var.get(),
                "translate": self.translate_var.get(),
                "source": self.source_var.get(),
                "chunk_seconds": self._chunk_seconds(),
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
            self.after(0, self.status_var.set,
                       f"Loading '{model_name}' model — cache: {cache}")
            model = WhisperModel(model_name, compute_type="float16", download_root=cache)
            self.after(0, self.status_var.set, "Transcribing…")
            segments, info = model.transcribe(
                path, language=lang, beam_size=5, task=task, vad_filter=True)
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

            model_sizes = {"tiny": "~75 MB", "base": "~150 MB", "small": "~250 MB",
                           "medium": "~800 MB", "large-v3": "~3 GB"}
            model_size_hint = model_sizes.get(model_name, "")
            self.after(0, self._log_info, (
                f"Source:     {source}\n"
                f"Model:      {model_name} {model_size_hint}  |  task: {task}  |  lang: {lang or 'auto'}\n"
                f"Model cache: {cache}\n"
                f"Chunk size: {chunk_secs}s  |  sample rate: {SAMPLE_RATE} Hz\n"
                "─────────────────────────────────────────\n"
                f"Loading model… (first run downloads {model_size_hint}, please wait)\n"
            ))
            self.after(0, self.status_var.set,
                       f"Loading '{model_name}' model… (cache: {cache})")

            model = WhisperModel(model_name, compute_type="float16", download_root=cache)

            self.after(0, self._log_info, "Model loaded. Opening audio device…\n")
            self.after(0, self.status_var.set, "Model loaded — opening audio device…")

            if source == "System audio (loopback)":
                self._capture_loopback(model, lang, task, chunk_secs)
            else:
                self._capture_microphone(model, lang, task, chunk_secs)

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

    def _capture_loopback(self, model, lang, task, chunk_secs):
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
        self._run_capture(loopback, model, lang, task, chunk_secs)

    def _capture_microphone(self, model, lang, task, chunk_secs):
        import soundcard as sc

        try:
            mic = sc.default_microphone()
        except Exception as exc:
            raise RuntimeError(
                f"Cannot find a default microphone.\n\n"
                f"soundcard error: {exc}\n\n"
                "Make sure a microphone is connected and set as the default recording device:\n"
                "  Start → Settings → Sound → Input"
            ) from exc

        self.after(0, self._log_info,
                   f"Microphone: {mic.name}\nRecording…\n")
        self.after(0, self.status_var.set, f"Recording microphone: {mic.name}")
        self._run_capture(mic, model, lang, task, chunk_secs)

    def _run_capture(self, device, model, lang, task, chunk_secs):
        import numpy as np

        chunk_frames = SAMPLE_RATE * chunk_secs
        chunk_num = 0

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
                    chunk = np.concatenate(buffer)
                    buffer = []
                    # Normalize: boost quiet audio and prevent clipping
                    peak = float(np.max(np.abs(chunk)))
                    if peak > 1e-6:
                        chunk = chunk / peak * 0.95
                    self.after(0, self.status_var.set,
                               f"Processing chunk #{chunk_num}…")
                    self._process_chunk(model, chunk, lang, task, chunk_num)

        # Flush remaining audio when stopped
        if buffer:
            remaining = np.concatenate(buffer)
            if len(remaining) > SAMPLE_RATE // 2:
                chunk_num += 1
                peak = float(np.max(np.abs(remaining)))
                if peak > 1e-6:
                    remaining = remaining / peak * 0.95
                self.after(0, self.status_var.set, "Processing final chunk…")
                self._process_chunk(model, remaining, lang, task, chunk_num)

    def _process_chunk(self, model, audio, lang, task, chunk_num=0):
        try:
            segments, info = model.transcribe(
                audio, language=lang, beam_size=5, task=task, vad_filter=True)
            text = " ".join(seg.text.strip() for seg in segments)
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

    def _append_text(self, text):
        self.text.config(state=tk.NORMAL)
        self.text.insert(tk.END, text + "\n")
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
