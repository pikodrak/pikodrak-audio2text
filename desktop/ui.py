import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

import config
import diarization as diar


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
                  relief="solid", borderwidth=1, wraplength=300,
                  padding=(6, 4)).pack()

    def _hide(self, _event=None):
        if self._tip:
            self._tip.destroy()
            self._tip = None


class SettingsDialog(tk.Toplevel):
    """Modal dialog for advanced settings: beam size, output format, diarization config."""

    def __init__(self, parent):
        super().__init__(parent)
        self.title("Settings")
        self.resizable(False, False)
        self.grab_set()
        self._parent = parent
        self._saved = False
        self._build()
        self._load_from_parent()
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.transient(parent)
        self.wait_visibility()
        self.update_idletasks()
        # Center over parent
        pw, ph = parent.winfo_width(), parent.winfo_height()
        px, py = parent.winfo_rootx(), parent.winfo_rooty()
        sw, sh = self.winfo_width(), self.winfo_height()
        self.geometry(f"+{px + (pw - sw)//2}+{py + (ph - sh)//2}")

    def _build(self):
        pad = {"padx": 10, "pady": 4}

        # ── Transcription ──────────────────────────────────────────────────────
        xc = ttk.LabelFrame(self, text="Transcription", padding=8)
        xc.pack(fill=tk.X, padx=10, pady=(10, 4))

        row = ttk.Frame(xc)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Beam size:", width=14, anchor=tk.W).pack(side=tk.LEFT)
        self.beam_var = tk.IntVar()
        beam_cb = ttk.Combobox(row, textvariable=self.beam_var,
                               values=config.BEAM_SIZES, width=4, state="readonly")
        beam_cb.pack(side=tk.LEFT, padx=(0, 8))
        self._beam_hint = ttk.Label(row, text="", foreground="#555555")
        self._beam_hint.pack(side=tk.LEFT)
        beam_cb.bind("<<ComboboxSelected>>", self._on_beam_change)

        row2 = ttk.Frame(xc)
        row2.pack(fill=tk.X, pady=2)
        self.translate_var = tk.BooleanVar()
        ttk.Checkbutton(row2, text="Translate output to English",
                        variable=self.translate_var).pack(side=tk.LEFT)

        # ── Output format ──────────────────────────────────────────────────────
        of = ttk.LabelFrame(self, text="Output Format", padding=8)
        of.pack(fill=tk.X, padx=10, pady=4)
        self.format_var = tk.StringVar()
        for fmt in config.OUTPUT_FORMATS:
            ttk.Radiobutton(of, text=fmt, variable=self.format_var, value=fmt).pack(
                anchor=tk.W)
        ttk.Label(of, text="(Timestamped shows [HH:MM:SS] per segment; "
                  "ignored when diarization is on)",
                  foreground="#777777", font=("Segoe UI", 8)).pack(anchor=tk.W, pady=(2, 0))

        # ── Speaker diarization ─────────────────────────────────────────────────
        df = ttk.LabelFrame(self, text="Speaker Diarization", padding=8)
        df.pack(fill=tk.X, padx=10, pady=4)

        self.diarize_var = tk.BooleanVar()
        ttk.Checkbutton(df, text="Enable speaker diarization",
                        variable=self.diarize_var,
                        command=self._on_diarize_toggle).pack(anchor=tk.W)
        _ToolTip(df,
                 "Labels each segment with a speaker ID.\n\n"
                 "Requires:\n"
                 "  • pip install pyannote.audio\n"
                 "  • A free HuggingFace token\n"
                 "  • Accept license at huggingface.co/pyannote/speaker-diarization-3.1\n\n"
                 "First run downloads ~1–2 GB. Not available in the bundled EXE.")

        # HF Token row
        self._hf_row = ttk.Frame(df)
        self._hf_row.pack(fill=tk.X, pady=(6, 2))
        ttk.Label(self._hf_row, text="HF Token:", width=12, anchor=tk.W).pack(side=tk.LEFT)
        self.hf_token_var = tk.StringVar()
        ttk.Entry(self._hf_row, textvariable=self.hf_token_var,
                  width=34, show="*").pack(side=tk.LEFT, padx=(0, 6))
        self._save_token_btn = ttk.Button(self._hf_row, text="Save to keyring",
                                          command=self._save_token_to_keyring)
        self._save_token_btn.pack(side=tk.LEFT)
        ttk.Label(df, text="Token is stored in your OS keyring, not in the settings file.",
                  foreground="#777777", font=("Segoe UI", 8)).pack(anchor=tk.W)

        # Speaker mode row
        self._mode_row = ttk.Frame(df)
        self._mode_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(self._mode_row, text="Speaker mode:", width=14, anchor=tk.W).pack(side=tk.LEFT)
        self.speaker_mode_var = tk.StringVar()
        mode_cb = ttk.Combobox(self._mode_row, textvariable=self.speaker_mode_var,
                               values=config.SPEAKER_MODES, width=12, state="readonly")
        mode_cb.pack(side=tk.LEFT, padx=(0, 8))
        mode_cb.bind("<<ComboboxSelected>>", self._on_speaker_mode_change)

        # Custom count (visible when Custom)
        self._custom_row = ttk.Frame(df)
        ttk.Label(self._custom_row, text="Speaker count:", width=14, anchor=tk.W).pack(side=tk.LEFT)
        self.custom_count_var = tk.IntVar()
        ttk.Spinbox(self._custom_row, textvariable=self.custom_count_var,
                    from_=2, to=10, width=4).pack(side=tk.LEFT)
        ttk.Label(self._custom_row, text="(forces pyannote to use exactly this many speakers)",
                  foreground="#555555").pack(side=tk.LEFT, padx=(8, 0))

        # Two-person names (visible when Two persons)
        self._names_row = ttk.Frame(df)
        ttk.Label(self._names_row, text="Speaker A name:", width=16,
                  anchor=tk.W).pack(side=tk.LEFT)
        self.speaker_a_var = tk.StringVar()
        ttk.Entry(self._names_row, textvariable=self.speaker_a_var, width=14).pack(
            side=tk.LEFT, padx=(0, 12))
        ttk.Label(self._names_row, text="Speaker B name:", anchor=tk.W).pack(side=tk.LEFT)
        self.speaker_b_var = tk.StringVar()
        ttk.Entry(self._names_row, textvariable=self.speaker_b_var, width=14).pack(
            side=tk.LEFT)

        # ── Buttons ────────────────────────────────────────────────────────────
        btn_row = ttk.Frame(self)
        btn_row.pack(fill=tk.X, padx=10, pady=(8, 10))
        ttk.Button(btn_row, text="OK", command=self._ok, width=10).pack(side=tk.RIGHT)
        ttk.Button(btn_row, text="Cancel", command=self._cancel, width=10).pack(
            side=tk.RIGHT, padx=(0, 6))

    def _load_from_parent(self):
        p = self._parent
        self.beam_var.set(p.beam_size_var.get())
        self._refresh_beam_hint()
        self.translate_var.set(p.translate_var.get())
        self.format_var.set(p.output_format_var.get())
        self.diarize_var.set(p.diarize_var.get())
        self.hf_token_var.set(p.hf_token_var.get())
        self.speaker_mode_var.set(p.speaker_mode_var.get())
        self.custom_count_var.set(p.custom_speaker_count_var.get())
        self.speaker_a_var.set(p.speaker_a_name_var.get())
        self.speaker_b_var.set(p.speaker_b_name_var.get())
        self._on_diarize_toggle()
        self._on_speaker_mode_change()

    def _refresh_beam_hint(self):
        hints = {1: "greedy / fastest", 2: "faster", 3: "balanced",
                 5: "accurate (default)", 8: "best accuracy, slowest"}
        self._beam_hint.config(text=hints.get(self.beam_var.get(), ""))

    def _on_beam_change(self, _event=None):
        self._refresh_beam_hint()

    def _on_diarize_toggle(self):
        state = "normal" if self.diarize_var.get() else "disabled"
        for child in self._hf_row.winfo_children():
            try:
                child.config(state=state)
            except tk.TclError:
                pass
        for child in self._mode_row.winfo_children():
            try:
                child.config(state=state)
            except tk.TclError:
                pass
        self._on_speaker_mode_change()

    def _on_speaker_mode_change(self, _event=None):
        mode = self.speaker_mode_var.get()
        enabled = self.diarize_var.get()

        self._custom_row.pack_forget()
        self._names_row.pack_forget()

        if not enabled:
            return
        if mode == "Custom":
            self._custom_row.pack(fill=tk.X, pady=(4, 0))
        elif mode == "Two persons":
            self._names_row.pack(fill=tk.X, pady=(4, 0))

    def _save_token_to_keyring(self):
        token = self.hf_token_var.get().strip()
        ok = config.save_hf_token(token)
        if ok:
            messagebox.showinfo("Saved", "HF token saved to OS keyring.",
                                parent=self)
        else:
            messagebox.showwarning(
                "Keyring unavailable",
                "Could not save to keyring.\n\n"
                "Install the keyring package:\n"
                "    pip install keyring\n\n"
                "The token will be active for this session only.",
                parent=self,
            )

    def _ok(self):
        p = self._parent
        p.beam_size_var.set(self.beam_var.get())
        p.translate_var.set(self.translate_var.get())
        p.output_format_var.set(self.format_var.get())
        p.diarize_var.set(self.diarize_var.get())
        p.hf_token_var.set(self.hf_token_var.get().strip())
        p.speaker_mode_var.set(self.speaker_mode_var.get())
        p.custom_speaker_count_var.set(self.custom_count_var.get())
        p.speaker_a_name_var.set(self.speaker_a_var.get().strip() or "Person A")
        p.speaker_b_name_var.set(self.speaker_b_var.get().strip() or "Person B")
        # Sync diarization toggle visibility in main window
        p._on_diarize_toggle()
        self._saved = True
        # Invalidate cached pipeline when token changes
        diar.reset_cache()
        self.destroy()

    def _cancel(self):
        self.destroy()


class Audio2TextApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Audio2Text")
        self.geometry("740x660")
        self.resizable(True, True)
        self._recording = False
        # Streaming preview state — shared between capture and preview threads
        self._whisper_lock = threading.Lock()
        self._buffer_lock = threading.Lock()
        self._live_buffer = []
        self._current_overlap = None
        self._preview_start_mark = None
        self._preview_gen = 0
        # Per-session speaker label mapping for cross-chunk consistency
        self._speaker_label_map = {}
        self._build_ui()
        self._apply_settings(config.load_settings())
        self._on_source_change()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ──────────────────────────────────────────────────── UI construction ──

    def _build_ui(self):
        # Input source
        self._src_frame = ttk.LabelFrame(self, text="Input Source", padding=8)
        self._src_frame.pack(fill=tk.X, padx=10, pady=(10, 0))

        self.source_var = tk.StringVar(value="Audio file")
        for src in config.INPUT_SOURCES:
            ttk.Radiobutton(
                self._src_frame, text=src,
                variable=self.source_var, value=src,
                command=self._on_source_change,
            ).pack(side=tk.LEFT, padx=8)

        # File picker row
        self._file_row = ttk.Frame(self, padding=(10, 5, 10, 0))
        self._file_row.pack(fill=tk.X)
        ttk.Label(self._file_row, text="Audio file:").pack(side=tk.LEFT)
        self.file_var = tk.StringVar()
        ttk.Entry(self._file_row, textvariable=self.file_var, width=52).pack(
            side=tk.LEFT, padx=5)
        ttk.Button(self._file_row, text="Browse…", command=self._browse).pack(
            side=tk.LEFT)

        # Microphone device row
        self._device_row = ttk.Frame(self, padding=(10, 5, 10, 0))
        ttk.Label(self._device_row, text="Microphone:").pack(side=tk.LEFT)
        self.mic_var = tk.StringVar(value=config.MIC_DEFAULT_LABEL)
        self._mic_cb = ttk.Combobox(
            self._device_row, textvariable=self.mic_var, width=44, state="readonly")
        self._mic_cb["values"] = [config.MIC_DEFAULT_LABEL]
        self._mic_cb.pack(side=tk.LEFT, padx=(5, 3))
        ttk.Button(self._device_row, text="↺", width=2,
                   command=self._refresh_mics).pack(side=tk.LEFT)

        # Options row: model, language, chunk, settings button
        opts = ttk.Frame(self, padding=(10, 6, 10, 4))
        opts.pack(fill=tk.X)

        ttk.Label(opts, text="Model:").pack(side=tk.LEFT)
        self.model_var = tk.StringVar(value="small")
        model_cb = ttk.Combobox(opts, textvariable=self.model_var, values=config.MODELS,
                                width=9, state="readonly")
        model_cb.pack(side=tk.LEFT, padx=(5, 2))
        self._model_hint_lbl = ttk.Label(opts, text="", foreground="#555555",
                                         font=("Segoe UI", 8))
        self._model_hint_lbl.pack(side=tk.LEFT, padx=(0, 10))
        model_cb.bind("<<ComboboxSelected>>", self._on_model_change)

        ttk.Label(opts, text="Language:").pack(side=tk.LEFT)
        self.lang_var = tk.StringVar(value="auto")
        ttk.Combobox(opts, textvariable=self.lang_var, values=config.LANGUAGES,
                     width=8, state="readonly").pack(side=tk.LEFT, padx=5)

        ttk.Label(opts, text="Chunk:").pack(side=tk.LEFT, padx=(12, 0))
        self.chunk_var = tk.StringVar(value=f"{config.CHUNK_DEFAULT}s")
        chunk_cb = ttk.Combobox(
            opts, textvariable=self.chunk_var,
            values=[f"{s}s" for s in config.CHUNK_OPTIONS],
            width=5, state="readonly",
        )
        chunk_cb.pack(side=tk.LEFT, padx=(3, 8))
        _ToolTip(chunk_cb,
                 "Buffer duration before each Whisper pass.\n"
                 "Smaller → lower latency, less context (may cut words).\n"
                 "Larger → higher latency, better accuracy.\n"
                 "Default: 8s  |  Two-person mode: 4–6s recommended")

        ttk.Button(opts, text="Settings…", command=self._open_settings).pack(side=tk.LEFT)

        # Diarization row: toggle + speaker mode (shown when diarization on)
        diar_row = ttk.Frame(self, padding=(10, 0, 10, 4))
        diar_row.pack(fill=tk.X)

        self.diarize_var = tk.BooleanVar(value=False)
        diar_cb = ttk.Checkbutton(
            diar_row, text="Speaker diarization",
            variable=self.diarize_var,
            command=self._on_diarize_toggle,
        )
        diar_cb.pack(side=tk.LEFT)
        _ToolTip(diar_cb,
                 "Label each segment with the speaker ID.\n\n"
                 "Requires:\n"
                 "  • pip install pyannote.audio\n"
                 "  • A free HuggingFace token (Settings → Diarization)\n"
                 "  • Accept license at huggingface.co/pyannote/speaker-diarization-3.1\n\n"
                 "First run downloads ~1–2 GB. Not available in the bundled EXE.")

        # Speaker mode selector — visible when diarization is enabled
        self._speaker_mode_frame = ttk.Frame(diar_row)
        ttk.Label(self._speaker_mode_frame, text="  Mode:").pack(side=tk.LEFT)
        self.speaker_mode_var = tk.StringVar(value="Auto")
        self._speaker_mode_cb = ttk.Combobox(
            self._speaker_mode_frame, textvariable=self.speaker_mode_var,
            values=config.SPEAKER_MODES, width=11, state="readonly")
        self._speaker_mode_cb.pack(side=tk.LEFT, padx=(3, 0))
        _ToolTip(self._speaker_mode_cb,
                 "Auto: detect any number of speakers.\n"
                 "Two persons: constrain to exactly 2 speakers (ideal for dialogues).\n"
                 "  → Use 4–6s chunks for best dialogue accuracy.\n"
                 "Custom: force a specific speaker count (set in Settings).")

        # Hidden vars — set via SettingsDialog, used during transcription
        self.hf_token_var = tk.StringVar()
        self.beam_size_var = tk.IntVar(value=5)
        self.translate_var = tk.BooleanVar(value=False)
        self.output_format_var = tk.StringVar(value="Plain text")
        self.custom_speaker_count_var = tk.IntVar(value=2)
        self.speaker_a_name_var = tk.StringVar(value="Person A")
        self.speaker_b_name_var = tk.StringVar(value="Person B")

        # Action buttons
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

        # Output
        out_frame = ttk.LabelFrame(self, text="Transcription / Log", padding=8)
        out_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 5))
        self.text = scrolledtext.ScrolledText(
            out_frame, wrap=tk.WORD, state=tk.DISABLED, font=("Segoe UI", 10))
        self.text.pack(fill=tk.BOTH, expand=True)
        self.text.tag_config("info", foreground="#888888")
        self.text.tag_config("error_tag", foreground="#cc0000")
        self.text.tag_config("pending", foreground="#999999",
                             font=("Segoe UI", 10, "italic"))

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN,
                  anchor=tk.W, padding=(6, 2)).pack(fill=tk.X, side=tk.BOTTOM)

    # ──────────────────────────────────────────────────── Settings ──

    def _apply_settings(self, s):
        """Apply a settings dict to all UI variables."""
        if s.get("model") in config.MODELS:
            self.model_var.set(s["model"])
        if s.get("language") in config.LANGUAGES:
            self.lang_var.set(s["language"])
        if s.get("source") in config.INPUT_SOURCES:
            self.source_var.set(s["source"])
        if s.get("chunk_seconds") in config.CHUNK_OPTIONS:
            self.chunk_var.set(f"{s['chunk_seconds']}s")
        if isinstance(s.get("mic_device"), str):
            self.mic_var.set(s["mic_device"] or config.MIC_DEFAULT_LABEL)
        if isinstance(s.get("diarize"), bool):
            self.diarize_var.set(s["diarize"])
        if s.get("beam_size") in config.BEAM_SIZES:
            self.beam_size_var.set(s["beam_size"])
        if s.get("output_format") in config.OUTPUT_FORMATS:
            self.output_format_var.set(s["output_format"])
        if s.get("speaker_mode") in config.SPEAKER_MODES:
            self.speaker_mode_var.set(s["speaker_mode"])
        if isinstance(s.get("custom_speaker_count"), int):
            self.custom_speaker_count_var.set(s["custom_speaker_count"])
        if isinstance(s.get("translate"), bool):
            self.translate_var.set(s["translate"])
        if isinstance(s.get("speaker_a_name"), str):
            self.speaker_a_name_var.set(s["speaker_a_name"])
        if isinstance(s.get("speaker_b_name"), str):
            self.speaker_b_name_var.set(s["speaker_b_name"])
        self.hf_token_var.set(config.load_hf_token())
        self._on_model_change()

    def _collect_settings(self):
        mic = self.mic_var.get()
        return {
            "model": self.model_var.get(),
            "language": self.lang_var.get(),
            "translate": self.translate_var.get(),
            "source": self.source_var.get(),
            "chunk_seconds": self._chunk_seconds(),
            "mic_device": mic if mic != config.MIC_DEFAULT_LABEL else "",
            "diarize": self.diarize_var.get(),
            "beam_size": self.beam_size_var.get(),
            "output_format": self.output_format_var.get(),
            "speaker_mode": self.speaker_mode_var.get(),
            "custom_speaker_count": self.custom_speaker_count_var.get(),
            "speaker_a_name": self.speaker_a_name_var.get(),
            "speaker_b_name": self.speaker_b_name_var.get(),
        }

    def _save_settings(self):
        config.save_settings(self._collect_settings())
        config.save_hf_token(self.hf_token_var.get().strip())

    def _chunk_seconds(self):
        raw = self.chunk_var.get().rstrip("s")
        try:
            return int(raw)
        except ValueError:
            return config.CHUNK_DEFAULT

    def _on_close(self):
        self._save_settings()
        self.destroy()

    def _open_settings(self):
        SettingsDialog(self)

    def _on_model_change(self, _event=None):
        hint = config.MODEL_HINTS.get(self.model_var.get(), "")
        self._model_hint_lbl.config(text=hint)

    # ──────────────────────────────────────────────────── Source / device ──

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
            self._speaker_mode_frame.pack(side=tk.LEFT)
        else:
            self._speaker_mode_frame.pack_forget()
        diar.reset_cache()

    def _refresh_mics(self):
        try:
            import soundcard as sc
            names = [config.MIC_DEFAULT_LABEL] + [m.name for m in sc.all_microphones()]
        except Exception:
            names = [config.MIC_DEFAULT_LABEL]
        current = self.mic_var.get()
        self._mic_cb["values"] = names
        if current not in names:
            self.mic_var.set(config.MIC_DEFAULT_LABEL)

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

    # ──────────────────────────────────────────────────── Start / Stop ──

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

    # ──────────────────────────────────────────────────── File transcription ──

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
            cache = config.model_cache_dir()
            do_diarize = self.diarize_var.get()
            hf_token = self.hf_token_var.get().strip() or os.environ.get("HF_TOKEN", "")
            beam_size = self.beam_size_var.get()
            output_fmt = self.output_format_var.get()
            speaker_mode = self.speaker_mode_var.get()
            two_person_names = (
                (self.speaker_a_name_var.get(), self.speaker_b_name_var.get())
                if speaker_mode == "Two persons" else None
            )

            self.after(0, self.status_var.set,
                       f"Loading '{model_name}' model — cache: {cache}")
            model = WhisperModel(model_name, compute_type="float16", download_root=cache)
            self.after(0, self.status_var.set, "Transcribing…")
            segments, info = model.transcribe(
                path, language=lang, beam_size=beam_size, task=task, vad_filter=True)
            segments = list(segments)

            if do_diarize:
                self.after(0, self.status_var.set,
                           "Loading diarization model… (first run downloads ~1–2 GB)")
                min_s, max_s = diar.speaker_constraints(
                    speaker_mode, self.custom_speaker_count_var.get())
                pipeline = diar.get_pipeline(hf_token)
                self.after(0, self.status_var.set, "Running speaker diarization…")
                diar_kwargs = {}
                if min_s is not None:
                    diar_kwargs["min_speakers"] = min_s
                if max_s is not None:
                    diar_kwargs["max_speakers"] = max_s
                diarization = pipeline(path, **diar_kwargs)
                label_map = {}
                labeled = diar.assign_speakers(
                    segments, diarization,
                    speaker_mode=speaker_mode,
                    two_person_names=two_person_names,
                    label_map=label_map,
                )
                result = diar.format_labeled_segments(labeled)
            elif output_fmt == "Timestamped":
                result = diar.format_timestamped(segments)
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

    # ──────────────────────────────────────────────────── Live capture ──

    def _start_live(self, source):
        self._recording = True
        self._speaker_label_map = {}
        self.btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.copy_btn.config(state=tk.DISABLED)
        self._set_text("")
        self._preview_start_mark = None
        self._preview_gen = 0
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
            cache = config.model_cache_dir()
            do_diarize = self.diarize_var.get()
            hf_token = self.hf_token_var.get().strip() or os.environ.get("HF_TOKEN", "")
            beam_size = self.beam_size_var.get()
            speaker_mode = self.speaker_mode_var.get()

            model_sizes = {"tiny": "~75 MB", "base": "~150 MB", "small": "~250 MB",
                           "medium": "~800 MB", "large-v3": "~3 GB"}
            hint = model_sizes.get(model_name, "")
            diar_note = f"  |  diarization: ON ({speaker_mode})" if do_diarize else ""
            self.after(0, self._log_info, (
                f"Source:      {source}\n"
                f"Model:       {model_name} {hint}  |  task: {task}"
                f"  |  lang: {lang or 'auto'}  |  beam: {beam_size}{diar_note}\n"
                f"Model cache: {cache}\n"
                f"Chunk size:  {chunk_secs}s  |  sample rate: {config.SAMPLE_RATE} Hz\n"
                "─────────────────────────────────────────\n"
                f"Loading model… (first run downloads {hint}, please wait)\n"
            ))
            self.after(0, self.status_var.set,
                       f"Loading '{model_name}' model… (cache: {cache})")

            model = WhisperModel(model_name, compute_type="float16", download_root=cache)

            diar_pipeline = None
            if do_diarize:
                self.after(0, self._log_info,
                           "Loading diarization model… (first run downloads ~1–2 GB)\n")
                self.after(0, self.status_var.set, "Loading diarization model…")
                diar_pipeline = diar.get_pipeline(hf_token)
                self.after(0, self._log_info, "Diarization model loaded.\n")

            self.after(0, self._log_info, "Model loaded. Opening audio device…\n")
            self.after(0, self.status_var.set, "Model loaded — opening audio device…")

            if source == "System audio (loopback)":
                self._capture_loopback(
                    model, lang, task, chunk_secs, beam_size,
                    do_diarize, diar_pipeline, speaker_mode)
            else:
                self._capture_microphone(
                    model, lang, task, chunk_secs, beam_size,
                    do_diarize, diar_pipeline, speaker_mode)

        except ImportError as exc:
            msg = (
                f"Missing dependency: {exc}\n\n"
                "The bundled EXE should include soundcard. If running from source:\n"
                "    pip install soundcard numpy"
            )
            self.after(0, self._on_error, msg)
        except Exception as exc:
            self.after(0, self._on_error, str(exc))
        finally:
            self._recording = False
            self.after(0, self._on_live_stopped)

    def _capture_loopback(self, model, lang, task, chunk_secs, beam_size,
                          diarize, diar_pipeline, speaker_mode):
        import soundcard as sc

        try:
            default_speaker = sc.default_speaker()
        except Exception as exc:
            raise RuntimeError(
                f"Cannot find a default audio output device.\n\nsoundcard error: {exc}\n\n"
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
        self.after(0, self.status_var.set, f"Recording loopback: {speaker_name}")
        self._run_capture(loopback, model, lang, task, chunk_secs, beam_size,
                          diarize, diar_pipeline, speaker_mode)

    def _capture_microphone(self, model, lang, task, chunk_secs, beam_size,
                            diarize, diar_pipeline, speaker_mode):
        import soundcard as sc

        selected = self.mic_var.get()
        try:
            if not selected or selected == config.MIC_DEFAULT_LABEL:
                mic = sc.default_microphone()
            else:
                mic = sc.get_microphone(id=selected)
        except Exception as exc:
            label = selected if selected and selected != config.MIC_DEFAULT_LABEL \
                else "default microphone"
            raise RuntimeError(
                f"Cannot open {label}.\n\nsoundcard error: {exc}\n\n"
                "Make sure a microphone is connected and set as the default recording device:\n"
                "  Start → Settings → Sound → Input\n\n"
                "If the issue persists, try selecting a specific device from the dropdown."
            ) from exc

        self.after(0, self._log_info, f"Microphone: {mic.name}\nRecording…\n")
        self.after(0, self.status_var.set, f"Recording microphone: {mic.name}")
        self._run_capture(mic, model, lang, task, chunk_secs, beam_size,
                          diarize, diar_pipeline, speaker_mode)

    def _run_capture(self, device, model, lang, task, chunk_secs, beam_size,
                     do_diarize, diar_pipeline, speaker_mode):
        import numpy as np

        chunk_frames = config.SAMPLE_RATE * chunk_secs
        chunk_num = 0
        overlap = np.zeros(0, dtype="float32")

        with self._buffer_lock:
            self._live_buffer = []
            self._current_overlap = overlap

        preview_thread = threading.Thread(
            target=self._preview_loop, args=(model, lang, task, beam_size), daemon=True)
        preview_thread.start()

        with device.recorder(samplerate=config.SAMPLE_RATE, channels=1) as recorder:
            buffer = []
            while self._recording:
                data = recorder.record(numframes=config.SAMPLE_RATE)
                if data.ndim > 1:
                    data = data.mean(axis=1)
                data = data.astype("float32")
                buffer.append(data)

                with self._buffer_lock:
                    self._live_buffer = list(buffer)

                total = sum(len(d) for d in buffer)
                secs_buffered = total / config.SAMPLE_RATE
                self.after(0, self.status_var.set,
                           f"Recording… {secs_buffered:.0f}/{chunk_secs}s buffered")

                if total >= chunk_frames:
                    chunk_num += 1
                    new_audio = np.concatenate(buffer)
                    buffer = []
                    self._preview_gen += 1
                    with self._buffer_lock:
                        self._live_buffer = []
                    chunk = np.concatenate([overlap, new_audio]) if len(overlap) else new_audio
                    overlap_secs = len(overlap) / config.SAMPLE_RATE
                    overlap = new_audio[-int(config.SAMPLE_RATE * config.OVERLAP_SECS):]
                    with self._buffer_lock:
                        self._current_overlap = overlap
                    peak = float(np.max(np.abs(chunk)))
                    if peak > 1e-6:
                        chunk = chunk / peak * 0.95
                    self.after(0, self.status_var.set, f"Processing chunk #{chunk_num}…")
                    self._process_chunk(model, chunk, lang, task, beam_size, chunk_num,
                                        overlap_secs, do_diarize, diar_pipeline, speaker_mode)

        # Flush remaining audio
        if buffer:
            remaining = np.concatenate(buffer)
            if len(remaining) > config.SAMPLE_RATE // 2:
                chunk_num += 1
                self._preview_gen += 1
                with self._buffer_lock:
                    self._live_buffer = []
                chunk = np.concatenate([overlap, remaining]) if len(overlap) else remaining
                overlap_secs = len(overlap) / config.SAMPLE_RATE
                peak = float(np.max(np.abs(chunk)))
                if peak > 1e-6:
                    chunk = chunk / peak * 0.95
                self.after(0, self.status_var.set, "Processing final chunk…")
                self._process_chunk(model, chunk, lang, task, beam_size, chunk_num,
                                    overlap_secs, do_diarize, diar_pipeline, speaker_mode)

        with self._buffer_lock:
            self._live_buffer = []

    def _preview_loop(self, model, lang, task, beam_size):
        """Greedy Whisper every PREVIEW_INTERVAL_SECS on accumulating buffer → grey italic."""
        import time
        import numpy as np

        while self._recording:
            time.sleep(config.PREVIEW_INTERVAL_SECS)
            if not self._recording:
                break

            gen = self._preview_gen

            with self._buffer_lock:
                buf = list(self._live_buffer)
                ovlp = (self._current_overlap.copy()
                        if self._current_overlap is not None and len(self._current_overlap)
                        else np.zeros(0, dtype="float32"))

            if not buf:
                continue

            snapshot = np.concatenate(buf)
            if len(snapshot) < config.SAMPLE_RATE // 2:
                continue

            audio = np.concatenate([ovlp, snapshot]) if len(ovlp) else snapshot
            overlap_secs = len(ovlp) / config.SAMPLE_RATE
            peak = float(np.max(np.abs(audio)))
            if peak < 1e-6:
                continue
            audio = audio / peak * 0.95

            with self._whisper_lock:
                if self._preview_gen != gen:
                    continue
                try:
                    segs_gen, _ = model.transcribe(
                        audio, language=lang, beam_size=1, task=task,
                        vad_filter=True, vad_parameters=config.LIVE_VAD_PARAMS,
                    )
                    text = " ".join(
                        seg.text.strip() for seg in segs_gen
                        if seg.start >= overlap_secs
                    )
                except Exception:
                    continue

            if text and self._recording and self._preview_gen == gen:
                self.after(0, self._update_preview, text, gen)

    def _process_chunk(self, model, audio, lang, task, beam_size, chunk_num=0,
                       overlap_secs=0.0, do_diarize=False, diar_pipeline=None,
                       speaker_mode="Auto"):
        try:
            with self._whisper_lock:
                segs_gen, info = model.transcribe(
                    audio, language=lang, beam_size=beam_size, task=task,
                    vad_filter=True, vad_parameters=config.LIVE_VAD_PARAMS)
                valid_segs = [seg for seg in segs_gen if seg.start >= overlap_secs]
            detected_lang = info.language

            if do_diarize and diar_pipeline is not None and valid_segs:
                import torch
                two_person_names = (
                    (self.speaker_a_name_var.get(), self.speaker_b_name_var.get())
                    if speaker_mode == "Two persons" else None
                )
                audio_tensor = torch.from_numpy(audio[None, :])
                min_s, max_s = diar.speaker_constraints(
                    speaker_mode, self.custom_speaker_count_var.get())
                diar_kwargs = {}
                if min_s is not None:
                    diar_kwargs["min_speakers"] = min_s
                if max_s is not None:
                    diar_kwargs["max_speakers"] = max_s
                diarization = diar_pipeline(
                    {"waveform": audio_tensor, "sample_rate": config.SAMPLE_RATE},
                    **diar_kwargs)
                labeled = diar.assign_speakers(
                    valid_segs, diarization,
                    speaker_mode=speaker_mode,
                    two_person_names=two_person_names,
                    label_map=self._speaker_label_map,
                )
                text = diar.format_labeled_segments(labeled)
                if text:
                    self.after(0, self._clear_preview)
                    self.after(0, lambda t=text: self._append_text(t, as_block=True))
                    self.after(0, self.status_var.set,
                               f"Chunk #{chunk_num} done — {detected_lang} | recording…")
                else:
                    self.after(0, self._clear_preview)
                    self.after(0, self.status_var.set,
                               f"Chunk #{chunk_num}: no speech | recording…")
            else:
                text = " ".join(seg.text.strip() for seg in valid_segs)
                if text:
                    self.after(0, self._finalize_preview, text)
                    self.after(0, self.status_var.set,
                               f"Chunk #{chunk_num} done — {detected_lang} | recording…")
                else:
                    self.after(0, self._clear_preview)
                    self.after(0, self.status_var.set,
                               f"Chunk #{chunk_num}: no speech | recording…")
        except Exception as exc:
            self.after(0, self.status_var.set, f"Chunk #{chunk_num} error: {exc}")

    def _on_live_stopped(self):
        self.btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.copy_btn.config(state=tk.NORMAL if self._has_text() else tk.DISABLED)
        self.status_var.set("Stopped")
        self._log_info("─────────────────────────────────────────\nStopped.\n")

    # ──────────────────────────────────────────────────── Error / text helpers ──

    def _on_error(self, msg):
        self.status_var.set("Error — see output area")
        self._log_error(f"ERROR:\n{msg}\n")
        messagebox.showerror("Error", msg)
        self.btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

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
            existing = self.text.get("1.0", "end-1c")
            if existing.strip():
                self.text.insert(tk.END, "\n\n")
            self.text.insert(tk.END, text)
        else:
            tail = self.text.get("end-2c", "end-1c")
            if tail and tail not in ("\n", " "):
                self.text.insert(tk.END, " ")
            self.text.insert(tk.END, text)
        self.text.see(tk.END)
        self.text.config(state=tk.DISABLED)
        self.copy_btn.config(state=tk.NORMAL)

    def _update_preview(self, text, gen):
        if gen != self._preview_gen:
            return
        self.text.config(state=tk.NORMAL)
        if self._preview_start_mark is not None:
            self.text.delete(self._preview_start_mark, tk.END)
        else:
            tail = self.text.get("end-2c", "end-1c")
            if tail and tail not in ("\n", " "):
                self.text.insert(tk.END, " ")
            self.text.mark_set("pending_preview", tk.END)
            self.text.mark_gravity("pending_preview", tk.LEFT)
            self._preview_start_mark = "pending_preview"
        self.text.insert(tk.END, text, "pending")
        self.text.see(tk.END)
        self.text.config(state=tk.DISABLED)

    def _finalize_preview(self, text):
        self.text.config(state=tk.NORMAL)
        if self._preview_start_mark is not None:
            self.text.delete(self._preview_start_mark, tk.END)
            self._preview_start_mark = None
        else:
            tail = self.text.get("end-2c", "end-1c")
            if tail and tail not in ("\n", " "):
                self.text.insert(tk.END, " ")
        self.text.insert(tk.END, text)
        self.text.see(tk.END)
        self.text.config(state=tk.DISABLED)
        self.copy_btn.config(state=tk.NORMAL)

    def _clear_preview(self):
        if self._preview_start_mark is not None:
            self.text.config(state=tk.NORMAL)
            self.text.delete(self._preview_start_mark, tk.END)
            self.text.config(state=tk.DISABLED)
            self._preview_start_mark = None

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
