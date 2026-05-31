import os
import sys
import time
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
                  relief="solid", borderwidth=1, wraplength=320,
                  padding=(6, 4)).pack()

    def _hide(self, _event=None):
        if self._tip:
            self._tip.destroy()
            self._tip = None


class _ConsoleRedirector:
    """Intercepts sys.stdout/stderr so model-download progress bars show in the GUI.

    tqdm rewrites its current line with a carriage return ('\\r'). We anchor the
    start of the current line with a Tk *mark* (which moves with the text) and,
    on each '\\r', delete only from that mark to the end — so a progress bar
    updates in place instead of progressively eating earlier lines.
    """
    _MARK = "console_line_start"

    def __init__(self, text_widget):
        self.text_widget = text_widget
        self._anchored = False

    def write(self, msg):
        self.text_widget.after(0, self._write, msg)

    def reset(self):
        """Drop the line anchor (call when the widget is cleared elsewhere)."""
        self._anchored = False

    def _write(self, msg):
        try:
            w = self.text_widget
            w.config(state=tk.NORMAL)
            if not self._anchored:
                w.mark_set(self._MARK, tk.END)
                w.mark_gravity(self._MARK, "left")
                self._anchored = True
            for ch in msg:
                if ch == "\r":
                    w.delete(self._MARK, tk.END)        # return to line start
                elif ch == "\n":
                    w.insert(tk.END, "\n")
                    w.mark_set(self._MARK, tk.END)      # next line starts here
                else:
                    w.insert(tk.END, ch)
            w.see(tk.END)
            w.config(state=tk.DISABLED)
        except Exception:
            pass

    def flush(self):
        pass


class AdvancedDialog(tk.Toplevel):
    """Out-of-the-way settings: model, language, VAD, translate, HF token.

    The main window deliberately keeps only the one control that matters for
    everyday use (number of speakers); everything else has a sane default and
    lives here.
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.title("Advanced settings")
        self.resizable(False, False)
        self.grab_set()
        self._parent = parent
        self._build()
        self._load_from_parent()
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.transient(parent)
        self.wait_visibility()
        self.update_idletasks()
        pw, ph = parent.winfo_width(), parent.winfo_height()
        px, py = parent.winfo_rootx(), parent.winfo_rooty()
        sw, sh = self.winfo_width(), self.winfo_height()
        self.geometry(f"+{px + (pw - sw)//2}+{py + (ph - sh)//2}")

    def _build(self):
        # ── Engine ──
        ef = ttk.LabelFrame(self, text="Transcription engine", padding=8)
        ef.pack(fill=tk.X, padx=10, pady=(10, 4))
        self.engine_var = tk.StringVar()
        for eng in config.ENGINES:
            ttk.Radiobutton(ef, text=eng, variable=self.engine_var, value=eng,
                            command=self._on_engine_change).pack(anchor=tk.W)
        ttk.Label(ef, text="Cloud transcribes the whole recording at once → far better "
                  "speaker separation for interviews. Needs internet + an API key.",
                  foreground="#777777", font=("Segoe UI", 8), wraplength=380).pack(
            anchor=tk.W, pady=(2, 0))

        # ── Model ──
        mf = ttk.LabelFrame(self, text="Speech model", padding=8)
        mf.pack(fill=tk.X, padx=10, pady=(10, 4))
        row = ttk.Frame(mf)
        row.pack(fill=tk.X)
        ttk.Label(row, text="Model:", width=10, anchor=tk.W).pack(side=tk.LEFT)
        self.model_var = tk.StringVar()
        cb = ttk.Combobox(row, textvariable=self.model_var, values=config.MODELS,
                          width=10, state="readonly")
        cb.pack(side=tk.LEFT)
        cb.bind("<<ComboboxSelected>>", lambda e: self._refresh_model_hint())
        self._model_hint = ttk.Label(mf, text="", foreground="#555555",
                                     font=("Segoe UI", 8), wraplength=360)
        self._model_hint.pack(anchor=tk.W, pady=(2, 0))

        row2 = ttk.Frame(mf)
        row2.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(row2, text="Language:", width=10, anchor=tk.W).pack(side=tk.LEFT)
        self.lang_var = tk.StringVar()
        ttk.Combobox(row2, textvariable=self.lang_var, values=config.LANGUAGES,
                     width=8, state="readonly").pack(side=tk.LEFT)
        ttk.Label(row2, text="(auto detects on the first words, then locks)",
                  foreground="#777777", font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(8, 0))

        prow = ttk.Frame(mf)
        prow.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(prow, text="Prompt:", width=10, anchor=tk.W).pack(side=tk.LEFT)
        self.initial_prompt_var = tk.StringVar()
        ttk.Entry(prow, textvariable=self.initial_prompt_var, width=34).pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(mf, text="Optional: names / terms with correct diacritics to bias "
                  "Whisper (e.g. \"Čestmír, Lankmajer, herohero\"). Empty = Czech default.",
                  foreground="#777777", font=("Segoe UI", 8), wraplength=380).pack(
            anchor=tk.W, pady=(2, 0))

        # ── Transcription options ──
        of = ttk.LabelFrame(self, text="Transcription", padding=8)
        of.pack(fill=tk.X, padx=10, pady=4)
        self.use_vad_var = tk.BooleanVar()
        ttk.Checkbutton(of, text="Voice-activity filter (skips silence; recommended)",
                        variable=self.use_vad_var).pack(anchor=tk.W)
        self.translate_var = tk.BooleanVar()
        ttk.Checkbutton(of, text="Translate output to English",
                        variable=self.translate_var).pack(anchor=tk.W)
        brow = ttk.Frame(of)
        brow.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(brow, text="Beam size (file mode):").pack(side=tk.LEFT)
        self.beam_var = tk.IntVar()
        ttk.Combobox(brow, textvariable=self.beam_var, values=config.BEAM_SIZES,
                     width=4, state="readonly").pack(side=tk.LEFT, padx=(6, 0))

        # ── HF token ──
        hf = ttk.LabelFrame(self, text="Speaker diarization — HuggingFace token", padding=8)
        hf.pack(fill=tk.X, padx=10, pady=4)
        ttk.Label(hf, text="Needed only when separating speakers (2+).",
                  foreground="#555555", font=("Segoe UI", 8)).pack(anchor=tk.W)
        trow = ttk.Frame(hf)
        trow.pack(fill=tk.X, pady=(4, 2))
        ttk.Label(trow, text="Token:", width=8, anchor=tk.W).pack(side=tk.LEFT)
        self.hf_token_var = tk.StringVar()
        ttk.Entry(trow, textvariable=self.hf_token_var, width=34, show="*").pack(
            side=tk.LEFT, padx=(0, 6))
        ttk.Button(trow, text="Save to keyring", command=self._save_token).pack(side=tk.LEFT)
        ttk.Label(hf, text="Free token at huggingface.co/settings/tokens — first accept the "
                  "license at huggingface.co/pyannote/speaker-diarization-3.1. "
                  "Stored in the OS keyring, never in the settings file.",
                  foreground="#777777", font=("Segoe UI", 8), wraplength=380).pack(anchor=tk.W)

        # ── Gemini (cloud) ──
        gf = ttk.LabelFrame(self, text="Google Gemini (cloud engine)", padding=8)
        gf.pack(fill=tk.X, padx=10, pady=4)
        grow = ttk.Frame(gf)
        grow.pack(fill=tk.X)
        ttk.Label(grow, text="Model:", width=8, anchor=tk.W).pack(side=tk.LEFT)
        self.gemini_model_var = tk.StringVar()
        ttk.Combobox(grow, textvariable=self.gemini_model_var,
                     values=config.GEMINI_MODELS, width=18, state="readonly").pack(side=tk.LEFT)
        krow = ttk.Frame(gf)
        krow.pack(fill=tk.X, pady=(4, 2))
        ttk.Label(krow, text="API key:", width=8, anchor=tk.W).pack(side=tk.LEFT)
        self.gemini_key_var = tk.StringVar()
        ttk.Entry(krow, textvariable=self.gemini_key_var, width=34, show="*").pack(
            side=tk.LEFT, padx=(0, 6))
        ttk.Button(krow, text="Save to keyring", command=self._save_gemini_key).pack(
            side=tk.LEFT)
        ttk.Label(gf, text="Free key at https://aistudio.google.com/apikey . "
                  "Stored in the OS keyring, never in the settings file.",
                  foreground="#777777", font=("Segoe UI", 8), wraplength=380).pack(anchor=tk.W)

        # ── Deepgram (real-time cloud) ──
        df2 = ttk.LabelFrame(self, text="Deepgram (real-time cloud engine)", padding=8)
        df2.pack(fill=tk.X, padx=10, pady=4)
        drow = ttk.Frame(df2)
        drow.pack(fill=tk.X, pady=(0, 2))
        ttk.Label(drow, text="API key:", width=8, anchor=tk.W).pack(side=tk.LEFT)
        self.deepgram_key_var = tk.StringVar()
        ttk.Entry(drow, textvariable=self.deepgram_key_var, width=34, show="*").pack(
            side=tk.LEFT, padx=(0, 6))
        ttk.Button(drow, text="Save to keyring", command=self._save_deepgram_key).pack(
            side=tk.LEFT)
        self.deepgram_eu_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(df2, text="Use EU servers — audio stays in the EU (mip_opt_out: "
                        "no storage, no model training)", variable=self.deepgram_eu_var).pack(
            anchor=tk.W, pady=(4, 0))
        ttk.Label(df2, text="Live speakers in the window even on a weak PC. Free key "
                  "(~133h) at https://console.deepgram.com . Stored in the OS keyring.",
                  foreground="#777777", font=("Segoe UI", 8), wraplength=380).pack(anchor=tk.W)

        # ── Buttons ──
        btn = ttk.Frame(self)
        btn.pack(fill=tk.X, padx=10, pady=(8, 10))
        ttk.Button(btn, text="OK", command=self._ok, width=10).pack(side=tk.RIGHT)
        ttk.Button(btn, text="Cancel", command=self.destroy, width=10).pack(
            side=tk.RIGHT, padx=(0, 6))

    def _refresh_model_hint(self):
        self._model_hint.config(text=config.MODEL_HINTS.get(self.model_var.get(), ""))

    def _on_engine_change(self):
        pass  # both engines stay configurable; selection just decides what's used

    def _load_from_parent(self):
        p = self._parent
        self.engine_var.set(p.engine_var.get())
        self.model_var.set(p.model_var.get())
        self.lang_var.set(p.lang_var.get())
        self.use_vad_var.set(p.use_vad_var.get())
        self.translate_var.set(p.translate_var.get())
        self.beam_var.set(p.beam_size_var.get())
        self.hf_token_var.set(p.hf_token_var.get())
        self.gemini_model_var.set(p.gemini_model_var.get())
        self.gemini_key_var.set(p.gemini_key_var.get())
        self.deepgram_key_var.set(p.deepgram_key_var.get())
        self.deepgram_eu_var.set(p.deepgram_eu_var.get())
        self.initial_prompt_var.set(p.initial_prompt_var.get())
        self._refresh_model_hint()

    def _save_token(self):
        token = self.hf_token_var.get().strip()
        if config.save_hf_token(token):
            messagebox.showinfo("Saved", "HF token saved to the OS keyring.", parent=self)
        else:
            messagebox.showwarning(
                "Keyring unavailable",
                "Could not save to keyring.\nThe token will be active for this session only.",
                parent=self)

    def _save_gemini_key(self):
        key = self.gemini_key_var.get().strip()
        if config.save_gemini_key(key):
            messagebox.showinfo("Saved", "Gemini API key saved to the OS keyring.",
                                parent=self)
        else:
            messagebox.showwarning(
                "Keyring unavailable",
                "Could not save to keyring.\nThe key will be active for this session only.",
                parent=self)

    def _save_deepgram_key(self):
        key = self.deepgram_key_var.get().strip()
        if config.save_deepgram_key(key):
            messagebox.showinfo("Saved", "Deepgram API key saved to the OS keyring.",
                                parent=self)
        else:
            messagebox.showwarning(
                "Keyring unavailable",
                "Could not save to keyring.\nThe key will be active for this session only.",
                parent=self)

    def _ok(self):
        p = self._parent
        p.engine_var.set(self.engine_var.get())
        p.model_var.set(self.model_var.get())
        p.lang_var.set(self.lang_var.get())
        p.use_vad_var.set(self.use_vad_var.get())
        p.translate_var.set(self.translate_var.get())
        p.beam_size_var.set(self.beam_var.get())
        p.hf_token_var.set(self.hf_token_var.get().strip())
        p.gemini_model_var.set(self.gemini_model_var.get())
        p.gemini_key_var.set(self.gemini_key_var.get().strip())
        p.deepgram_key_var.set(self.deepgram_key_var.get().strip())
        p.deepgram_eu_var.set(self.deepgram_eu_var.get())
        p.initial_prompt_var.set(self.initial_prompt_var.get())
        config.save_gemini_key(self.gemini_key_var.get().strip())
        config.save_deepgram_key(self.deepgram_key_var.get().strip())
        p._on_model_change()
        p._update_speaker_hint()
        self.destroy()


class Audio2TextApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Audio2Text — real-time transcription")
        self.geometry("760x640")
        self.resizable(True, True)

        # ── Live engine shared state ──
        self._recording = False
        self._words_lock = threading.Lock()   # guards _words / _tentative
        self._cap_lock = threading.Lock()      # guards _pending_audio
        self._diar_lock = threading.Lock()     # guards _captured / _diar_chunks
        self._words = []          # committed streaming.Word objects (shared)
        self._tentative = []      # tentative tail Word objects
        self._pending_audio = []  # raw blocks captured, awaiting the ASR thread
        self._captured = 0        # total samples captured (absolute clock)
        # Full session audio kept as a list of blocks (joined only when needed),
        # so capturing stays O(1) per block instead of re-concatenating a growing
        # array every 0.25 s — that was O(n^2) over a long recording.
        self._diar_chunks = []
        self._diar_samples = 0
        self._diar_on = False
        self._keep_audio = False  # keep full session audio (local diar OR cloud engine)
        self._asr = None
        self._last_render = None
        self._refresh_scheduled = False
        self._live_thread = None
        self._speaker_tags = {}   # speaker name -> Text tag (assigned a color)
        self._diar_error_shown = False
        # Whisper model is cached and reused across Stop/Start. Recreating a CUDA
        # model every session and letting the GC free the old one at an arbitrary
        # time can crash ctranslate2/cuDNN with an access violation.
        self._whisper_model = None
        self._whisper_model_key = None

        self._diar_downloading = False

        self._build_ui()

        self._redir = _ConsoleRedirector(self.text)
        sys.stdout = self._redir
        sys.stderr = self._redir

        raw = config.load_settings()
        settings = config.sanitize_settings(raw, diarization_available=diar.DIARIZATION_AVAILABLE)
        self._apply_settings(settings)
        if settings is not raw and raw.get("num_speakers", 1) != settings.get("num_speakers", 1):
            self._save_settings()
        self._on_source_change()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(200, self._maybe_auto_download)

    # ──────────────────────────────────────────────────── UI construction ──

    def _build_ui(self):
        # Input source
        self._src_frame = ttk.LabelFrame(self, text="Input source", padding=8)
        self._src_frame.pack(fill=tk.X, padx=10, pady=(10, 0))
        self.source_var = tk.StringVar(value="Microphone")
        for src in config.INPUT_SOURCES:
            ttk.Radiobutton(self._src_frame, text=src, variable=self.source_var,
                            value=src, command=self._on_source_change).pack(
                side=tk.LEFT, padx=8)

        # Microphone device row
        self._device_row = ttk.Frame(self, padding=(10, 5, 10, 0))
        ttk.Label(self._device_row, text="Microphone:").pack(side=tk.LEFT)
        self.mic_var = tk.StringVar(value=config.MIC_DEFAULT_LABEL)
        self._mic_cb = ttk.Combobox(self._device_row, textvariable=self.mic_var,
                                    width=44, state="readonly")
        self._mic_cb["values"] = [config.MIC_DEFAULT_LABEL]
        self._mic_cb.pack(side=tk.LEFT, padx=(5, 3))
        ttk.Button(self._device_row, text="↺", width=2,
                   command=self._refresh_mics).pack(side=tk.LEFT)

        # File picker row
        self._file_row = ttk.Frame(self, padding=(10, 5, 10, 0))
        ttk.Label(self._file_row, text="Audio file:").pack(side=tk.LEFT)
        self.file_var = tk.StringVar()
        ttk.Entry(self._file_row, textvariable=self.file_var, width=52).pack(
            side=tk.LEFT, padx=5)
        ttk.Button(self._file_row, text="Browse…", command=self._browse).pack(side=tk.LEFT)

        # The one prominent setting: number of speakers
        spk = ttk.LabelFrame(self, text="Speakers", padding=8)
        spk.pack(fill=tk.X, padx=10, pady=(8, 0))
        ttk.Label(spk, text="Number of speakers:").pack(side=tk.LEFT)
        self.num_speakers_var = tk.IntVar(value=2)
        self._spk_spin = ttk.Spinbox(
            spk, from_=1, to=config.MAX_SPEAKERS, width=4,
            textvariable=self.num_speakers_var, command=self._on_speakers_change)
        self._spk_spin.pack(side=tk.LEFT, padx=(6, 8))
        self._spk_spin.bind("<KeyRelease>", lambda e: self._on_speakers_change())
        self._spk_hint = ttk.Label(spk, text="", foreground="#555555")
        self._spk_hint.pack(side=tk.LEFT)
        _ToolTip(self._spk_spin,
                 "How many people are talking.\n"
                 "1 = plain transcript, no speaker labels.\n"
                 "2+ = label each line with the speaker (needs the diarization\n"
                 "model + a HuggingFace token, see Advanced).")

        # Hidden / advanced vars (defaults; edited via Advanced dialog)
        self.model_var = tk.StringVar(value="small")
        self.lang_var = tk.StringVar(value="cs")
        self.translate_var = tk.BooleanVar(value=False)
        self.beam_size_var = tk.IntVar(value=5)
        self.use_vad_var = tk.BooleanVar(value=True)
        self.hf_token_var = tk.StringVar()
        self.engine_var = tk.StringVar(value=config.ENGINE_LOCAL)
        self.gemini_model_var = tk.StringVar(value=config.GEMINI_MODELS[0])
        self.gemini_key_var = tk.StringVar()
        self.deepgram_key_var = tk.StringVar()
        self.deepgram_eu_var = tk.BooleanVar(value=True)
        self.initial_prompt_var = tk.StringVar()

        # Action buttons
        btn = ttk.Frame(self, padding=(10, 8, 10, 4))
        btn.pack(fill=tk.X)
        self.btn = ttk.Button(btn, text="Start", command=self._start)
        self.btn.pack(side=tk.LEFT)
        self.stop_btn = ttk.Button(btn, text="Stop", command=self._stop, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        self.copy_btn = ttk.Button(btn, text="Copy", command=self._copy, state=tk.DISABLED)
        self.copy_btn.pack(side=tk.LEFT, padx=5)
        ttk.Button(btn, text="Clear", command=self._clear).pack(side=tk.LEFT)
        ttk.Button(btn, text="Advanced…", command=self._open_advanced).pack(side=tk.LEFT, padx=5)
        self._model_lbl = ttk.Label(btn, text="", foreground="#777777",
                                    font=("Segoe UI", 8))
        self._model_lbl.pack(side=tk.RIGHT)

        # Output
        out = ttk.LabelFrame(self, text="Transcript", padding=8)
        out.pack(fill=tk.BOTH, expand=True, padx=10, pady=(4, 5))
        self.text = scrolledtext.ScrolledText(out, wrap=tk.WORD, state=tk.DISABLED,
                                              font=("Segoe UI", 11))
        self.text.pack(fill=tk.BOTH, expand=True)
        self.text.tag_config("info", foreground="#888888")
        self.text.tag_config("error_tag", foreground="#cc0000")
        self.text.tag_config("pending", foreground="#999999",
                             font=("Segoe UI", 11, "italic"))

        # Status bar
        self._status_frame = ttk.Frame(self)
        self._status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self._status_frame, textvariable=self.status_var, relief=tk.SUNKEN,
                  anchor=tk.W, padding=(6, 2)).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._retry_btn = ttk.Button(self._status_frame, text="Retry",
                                     command=self._start_diar_download)

        self._update_speaker_hint()

    # ──────────────────────────────────────────────────── Settings ──

    def _apply_settings(self, s):
        if s.get("model") in config.MODELS:
            self.model_var.set(s["model"])
        if s.get("language") in config.LANGUAGES:
            self.lang_var.set(s["language"])
        if s.get("source") in config.INPUT_SOURCES:
            self.source_var.set(s["source"])
        if isinstance(s.get("mic_device"), str):
            self.mic_var.set(s["mic_device"] or config.MIC_DEFAULT_LABEL)
        if s.get("engine") in config.ENGINES:
            self.engine_var.set(s["engine"])
        if s.get("gemini_model") in config.GEMINI_MODELS:
            self.gemini_model_var.set(s["gemini_model"])
        if isinstance(s.get("initial_prompt"), str):
            self.initial_prompt_var.set(s["initial_prompt"])
        n = s.get("num_speakers")
        if isinstance(n, int):
            n = max(1, min(config.MAX_SPEAKERS, n))
            # Only the LOCAL engine needs pyannote; the cloud engine keeps 2+.
            if (n >= 2 and not diar.DIARIZATION_AVAILABLE
                    and self.engine_var.get() not in config.CLOUD_ENGINES):
                n = 1
            self.num_speakers_var.set(n)
        if s.get("beam_size") in config.BEAM_SIZES:
            self.beam_size_var.set(s["beam_size"])
        if isinstance(s.get("translate"), bool):
            self.translate_var.set(s["translate"])
        if isinstance(s.get("use_vad"), bool):
            self.use_vad_var.set(s["use_vad"])
        self.hf_token_var.set(config.load_hf_token())
        self.gemini_key_var.set(config.load_gemini_key())
        self.deepgram_key_var.set(config.load_deepgram_key())
        if isinstance(s.get("deepgram_eu"), bool):
            self.deepgram_eu_var.set(s["deepgram_eu"])
        self._on_model_change()
        self._update_speaker_hint()

    def _num_speakers(self):
        try:
            return max(1, min(config.MAX_SPEAKERS, int(self.num_speakers_var.get())))
        except (tk.TclError, ValueError):
            return 1

    def _snapshot_config(self, source=None):
        """Read every GUI value on the MAIN thread into a plain dict.

        tkinter is not thread-safe, so worker threads must never touch *_var.get()
        — they read from this snapshot instead.
        """
        lang = self.lang_var.get()
        return {
            "source": source,
            "mic": self.mic_var.get(),
            "model_name": self.model_var.get(),
            "language": None if lang == "auto" else lang,    # for Whisper
            "cloud_language": "cs" if lang == "auto" else lang,  # for Gemini
            "task": "translate" if self.translate_var.get() else "transcribe",
            "use_vad": self.use_vad_var.get(),
            "beam_size": self.beam_size_var.get(),
            "num_speakers": self._num_speakers(),
            "hf_token": self.hf_token_var.get().strip() or os.environ.get("HF_TOKEN", ""),
            "initial_prompt": self.initial_prompt_var.get(),
            "engine": self.engine_var.get(),
            "use_cloud": self.engine_var.get() == config.ENGINE_GEMINI,
            "use_deepgram": self.engine_var.get() == config.ENGINE_DEEPGRAM,
            "gemini_key": self.gemini_key_var.get().strip() or config.load_gemini_key(),
            "gemini_model": self.gemini_model_var.get(),
            "deepgram_key": self.deepgram_key_var.get().strip() or config.load_deepgram_key(),
            "deepgram_eu": bool(self.deepgram_eu_var.get()),
        }

    def _collect_settings(self):
        mic = self.mic_var.get()
        return {
            "model": self.model_var.get(),
            "language": self.lang_var.get(),
            "translate": self.translate_var.get(),
            "source": self.source_var.get(),
            "mic_device": mic if mic != config.MIC_DEFAULT_LABEL else "",
            "num_speakers": self._num_speakers(),
            "beam_size": self.beam_size_var.get(),
            "use_vad": self.use_vad_var.get(),
            "engine": self.engine_var.get(),
            "gemini_model": self.gemini_model_var.get(),
            "initial_prompt": self.initial_prompt_var.get(),
            "deepgram_eu": bool(self.deepgram_eu_var.get()),
        }

    def _save_settings(self):
        config.save_settings(self._collect_settings())
        config.save_hf_token(self.hf_token_var.get().strip())
        config.save_gemini_key(self.gemini_key_var.get().strip())
        config.save_deepgram_key(self.deepgram_key_var.get().strip())

    def _on_close(self):
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        self._recording = False
        self._save_settings()
        self.destroy()

    def _open_advanced(self):
        AdvancedDialog(self)

    def _on_model_change(self, _event=None):
        self._model_lbl.config(text=f"model: {self.model_var.get()}  ·  "
                               f"lang: {self.lang_var.get()}")

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

    def _update_speaker_hint(self):
        n = self._num_speakers()
        eng = self.engine_var.get()
        if eng == config.ENGINE_DEEPGRAM:
            self._spk_hint.config(text="(Deepgram separates speakers live)")
            return
        cloud = eng == config.ENGINE_GEMINI
        if n <= 1:
            self._spk_hint.config(text="(plain transcript, no speaker labels)")
        elif cloud:
            self._spk_hint.config(text=f"(cloud separates {n} speakers on stop)")
        elif not diar.DIARIZATION_AVAILABLE:
            self._spk_hint.config(text=f"(separate {n} voices — diarization not installed yet)")
        else:
            self._spk_hint.config(text=f"(label each line with one of {n} speakers)")

    def _on_speakers_change(self):
        self._update_speaker_hint()
        # Cloud engine does its own diarization — no local pyannote needed.
        if self.engine_var.get() in config.CLOUD_ENGINES:
            return
        if self._num_speakers() >= 2 and not diar.DIARIZATION_AVAILABLE:
            if getattr(sys, "frozen", False):
                self._start_diar_download()
            else:
                messagebox.showinfo(
                    "Install pyannote.audio",
                    "Separating speakers needs pyannote.audio.\n\n"
                    "Install it in this Python environment:\n"
                    "    pip install pyannote.audio\n\n"
                    "Then restart Audio2Text.")

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
            filetypes=[("Audio files", "*.mp3 *.wav *.m4a *.ogg *.flac *.aac *.wma *.opus"),
                       ("All files", "*.*")])
        if path:
            self.file_var.set(path)

    # ──────────────────────────────────────────────────── Diarization download ──

    def _maybe_auto_download(self):
        if diar.DIARIZATION_AVAILABLE or self._diar_downloading:
            return
        if not getattr(sys, "frozen", False):
            return
        if self.engine_var.get() in config.CLOUD_ENGINES:
            return  # cloud engine needs no local diarization model
        if self._num_speakers() >= 2:
            self._start_diar_download()

    def _start_diar_download(self):
        if self._diar_downloading:
            return
        self._diar_downloading = True
        self._retry_btn.pack_forget()
        self.status_var.set("Downloading diarization (PyTorch + pyannote.audio ~2 GB)…")
        diar.setup_venv(progress_callback=self._on_diar_progress,
                        done_callback=self._on_diar_done)

    def _on_diar_progress(self, message):
        self.after(0, self.status_var.set, f"Diarization: {message}")

    def _on_diar_done(self, success, error):
        self.after(0, self._finish_diar_download, success, error)

    def _finish_diar_download(self, success, error):
        self._diar_downloading = False
        if success:
            self.status_var.set("Diarization ready — set the number of speakers and Start")
            self._update_speaker_hint()
        else:
            short = (error or "Unknown error")[:120]
            if "Python 3 not found" in (error or ""):
                self.status_var.set("Install Python from python.org to enable diarization")
            else:
                self.status_var.set(f"Diarization download failed — {short}")
            self._retry_btn.pack(side=tk.RIGHT, padx=(4, 4))

    # ──────────────────────────────────────────────────── Start / Stop ──

    def _start(self):
        if self.source_var.get() == "Audio file":
            self._start_file_transcribe()
        else:
            self._start_live(self.source_var.get())

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
        cfg = self._snapshot_config()
        threading.Thread(target=self._transcribe_file, args=(path, cfg),
                         daemon=True).start()

    def _transcribe_file(self, path, cfg):
        # Cloud engine: send the file straight to Gemini for a diarized transcript.
        if cfg["use_cloud"]:
            self._transcribe_file_gemini(path, cfg)
            return
        try:
            from faster_whisper import WhisperModel
            model_name = cfg["model_name"]
            lang = cfg["language"]
            task = cfg["task"]
            cache = config.model_cache_dir()
            n = cfg["num_speakers"]
            do_diarize = n >= 2 and diar.DIARIZATION_AVAILABLE
            hf_token = cfg["hf_token"]

            device, compute = config.whisper_device_and_compute_type()
            self.after(0, self.status_var.set,
                       f"Loading '{model_name}' ({device}/{compute}) — cache: {cache}")
            model = WhisperModel(model_name, device=device, compute_type=compute,
                                 download_root=cache)
            # Decode once into a numpy array so both Whisper and (WAV-based)
            # diarization work for any input format — feeding a file path to the
            # diarizer would hit the broken torchcodec backend for non-WAV files.
            from faster_whisper.audio import decode_audio
            audio = decode_audio(path, sampling_rate=config.SAMPLE_RATE)

            self.after(0, self.status_var.set, "Transcribing…")
            use_vad = cfg["use_vad"]
            tkwargs = dict(
                language=lang, beam_size=cfg["beam_size"], task=task,
                vad_filter=use_vad, word_timestamps=True,
                hallucination_silence_threshold=config.HALLUCINATION_SILENCE_SEC)
            init_prompt = config.whisper_initial_prompt(lang, cfg["initial_prompt"])
            if init_prompt:
                tkwargs["initial_prompt"] = init_prompt
            if use_vad:
                tkwargs["vad_parameters"] = config.FILE_VAD_PARAMS
            segments, info = model.transcribe(audio, **tkwargs)
            segments = list(segments)

            if do_diarize:
                self.after(0, self.status_var.set,
                           "Diarizing… (first run downloads ~1–2 GB)")
                result_turns = diar.run_diarize_audio(
                    audio, config.SAMPLE_RATE, hf_token, n, n)
                labeled = diar.assign_speakers(segments, result_turns,
                                               speaker_mode="Custom", label_map={})
                result = diar.format_labeled_segments(labeled)
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

    # ──────────────────────────────────────────────────── Cloud (Gemini) ──

    def _cloud_progress(self, msg):
        self.after(0, self.status_var.set, msg)

    def _cloud_transcribe(self, audio, cfg):
        """Blocking call to Gemini on a mono float32 array; returns transcript."""
        import cloud
        return cloud.transcribe_gemini(
            cfg["gemini_key"], cfg["gemini_model"], audio, config.SAMPLE_RATE,
            language=cfg["cloud_language"], num_speakers=cfg["num_speakers"],
            progress=self._cloud_progress)

    def _transcribe_file_gemini(self, path, cfg):
        try:
            self.after(0, self.status_var.set, "Decoding audio…")
            from faster_whisper.audio import decode_audio
            audio = decode_audio(path, sampling_rate=config.SAMPLE_RATE)
            self.after(0, self.status_var.set, "Sending to Google Gemini…")
            text = self._cloud_transcribe(audio, cfg)
            self.after(0, self._on_file_done, text, "cloud")
        except Exception as exc:
            self.after(0, self._on_error, str(exc))

    # ──────────────────────────────────────────────────── Live: orchestration ──

    def _start_live(self, source):
        if self._live_thread is not None and self._live_thread.is_alive():
            return  # a previous session is still shutting down
        self._recording = True
        self._diar_error_shown = False
        self.btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.copy_btn.config(state=tk.DISABLED)
        self._set_text("")
        self._last_render = None
        self.status_var.set("Starting…")
        cfg = self._snapshot_config(source)
        target = self._live_loop_deepgram if cfg["use_deepgram"] else self._live_loop
        self._live_thread = threading.Thread(target=target, args=(cfg,), daemon=True)
        self._live_thread.start()

    # ──────────────────────────────────────────────────── Live: Deepgram ──

    @staticmethod
    def _dg_speaker_name(n):
        return f"Speaker {int(n) + 1}"

    def _live_loop_deepgram(self, cfg):
        """Stream audio to Deepgram and render live diarized words.

        No local model — the cloud does transcription AND live speaker labels,
        so this works even on a weak PC with no GPU.
        """
        import deepgram
        stream = None
        self._diar_on = True           # Deepgram words carry speaker labels
        with self._words_lock:
            self._words = []
            self._tentative = []
        self._last_render = None
        try:
            self.after(0, self._log_info,
                       f"Source:   {cfg['source']}\n"
                       "Engine:   Deepgram (real-time cloud)\n"
                       f"Language: {cfg['cloud_language']}  |  live diarization\n"
                       "─────────────────────────────────────────\n"
                       "Connecting to Deepgram…\n")
            self.after(0, self.status_var.set, "Connecting to Deepgram…")
            stream = deepgram.DeepgramStream(
                cfg["deepgram_key"], language=cfg["cloud_language"],
                sample_rate=config.SAMPLE_RATE, eu=cfg["deepgram_eu"],
                on_words=self._on_deepgram_words, on_error=self._on_deepgram_error)
            stream.start()

            device_obj = self._open_capture_device(cfg["source"], cfg["mic"])
            self.after(0, self.status_var.set, "Recording… (Deepgram, live speakers)")
            self.after(0, self._log_info, "Recording — speak now.\n")
            self._deepgram_capture(device_obj, stream)
        except Exception as exc:
            self.after(0, self._on_error, str(exc))
        finally:
            self._recording = False
            if stream:
                try:
                    stream.finish()
                except Exception:
                    pass
            self.after(0, self._on_live_stopped)

    def _deepgram_capture(self, device, stream):
        import numpy as np
        import warnings
        warnings.filterwarnings("ignore", message="data discontinuity in recording")
        block = int(config.SAMPLE_RATE * config.CAPTURE_BLOCK_SECS)
        with device.recorder(samplerate=config.SAMPLE_RATE, channels=1) as rec:
            while self._recording:
                data = rec.record(numframes=block)
                if data.ndim > 1:
                    data = data.mean(axis=1)
                pcm = (np.clip(data, -1.0, 1.0) * 32767).astype("<i2").tobytes()
                stream.send_audio(pcm)

    def _on_deepgram_words(self, words, is_final):
        """Called from the Deepgram WS thread with diarized words."""
        import streaming
        objs = [streaming.Word(w["start"], w["end"], " " + w["text"],
                               speaker=self._dg_speaker_name(w["speaker"]))
                for w in words if w["text"]]
        with self._words_lock:
            if is_final:
                self._words.extend(objs)
                self._tentative = []
            else:
                self._tentative = objs   # interim segment replaces the tail
        self._schedule_refresh()

    def _on_deepgram_error(self, msg):
        self._recording = False
        self.after(0, self.status_var.set, "Deepgram error — see dialog")
        self.after(0, lambda m=msg: messagebox.showerror("Deepgram", m))

    def _get_whisper_model(self, WhisperModel, name, device, compute, cache):
        """Return a cached WhisperModel, loading (and replacing) only when needed."""
        key = (name, device, compute)
        if self._whisper_model is not None and self._whisper_model_key == key:
            self.after(0, self.status_var.set, f"Using '{name}' ({device}/{compute})")
            return self._whisper_model
        # Free the previous model now, while no inference is running, so its
        # (possibly CUDA) resources are released at a safe point.
        self._whisper_model = None
        self._whisper_model_key = None
        import gc
        gc.collect()
        self.after(0, self.status_var.set, f"Loading '{name}' ({device}/{compute})…")
        model = WhisperModel(name, device=device, compute_type=compute,
                             download_root=cache)
        self._whisper_model = model
        self._whisper_model_key = key
        return model

    def _live_loop(self, cfg):
        import numpy as np
        import streaming
        diar_worker = None
        asr_thread = diar_thread = None
        source = cfg["source"]
        hf_token = cfg["hf_token"]
        n_speakers = cfg["num_speakers"]
        try:
            from faster_whisper import WhisperModel

            model_name = cfg["model_name"]
            lang = cfg["language"]
            task = cfg["task"]
            use_vad = cfg["use_vad"]
            cache = config.model_cache_dir()
            use_cloud = cfg["use_cloud"]
            # Local diarization only when NOT using the cloud (Gemini does its own).
            diar_on = (not use_cloud) and n_speakers >= 2 and diar.DIARIZATION_AVAILABLE
            self._diar_on = diar_on
            # Keep the whole recording when either local diar or the cloud needs it.
            self._keep_audio = diar_on or use_cloud
            self._use_cloud = use_cloud

            engine_note = ("Google Gemini (cloud, on stop)" if use_cloud
                           else f"local{' + diarization' if diar_on else ''}")
            self.after(0, self._log_info,
                       f"Source:   {source}\n"
                       f"Engine:   {engine_note}\n"
                       f"Model:    {model_name}  |  lang: {lang or 'auto'}  |  task: {task}\n"
                       f"Speakers: {n_speakers}\n"
                       f"Cache:    {cache}\n"
                       "─────────────────────────────────────────\n"
                       "Loading model…\n")
            device, compute = config.whisper_device_and_compute_type()
            model = self._get_whisper_model(WhisperModel, model_name, device, compute, cache)

            if diar_on:
                if getattr(sys, "frozen", False):
                    self.after(0, self.status_var.set, "Starting diarization worker…")
                    diar_worker = diar.DiarizeWorker()
                    diar_worker.start(hf_token)
                else:
                    self.after(0, self.status_var.set, "Loading diarization model…")
                    diar.preload_pipeline(hf_token)

            # Beam search is more accurate but re-decodes the buffer twice per
            # LocalAgreement pass — affordable on a GPU, too slow on CPU.
            stream_beam = 5 if device == "cuda" else 1
            init_prompt = config.whisper_initial_prompt(lang, cfg["initial_prompt"])
            transcribe_fn = streaming.WhisperTranscribeFn(
                model, language=lang, task=task, vad_filter=use_vad,
                vad_parameters=config.LIVE_VAD_PARAMS, beam_size=stream_beam,
                initial_prompt=init_prompt,
                hallucination_silence_threshold=config.HALLUCINATION_SILENCE_SEC)
            self._asr = streaming.OnlineASR(
                transcribe_fn, sample_rate=config.SAMPLE_RATE,
                buffer_trim_sec=config.BUFFER_TRIM_SEC)

            with self._words_lock:
                self._words = []
                self._tentative = []
            with self._diar_lock:
                self._captured = 0
                self._diar_chunks = []
                self._diar_samples = 0
            with self._cap_lock:
                self._pending_audio = []

            device_obj = self._open_capture_device(source, cfg["mic"])
            self.after(0, self.status_var.set, "Recording…")
            self.after(0, self._log_info, "Recording — speak now.\n")

            asr_thread = threading.Thread(target=self._asr_worker, daemon=True)
            asr_thread.start()
            if diar_on:
                diar_thread = threading.Thread(
                    target=self._diar_worker, args=(hf_token, n_speakers, diar_worker),
                    daemon=True)
                diar_thread.start()

            self._capture_loop(device_obj)

        except ImportError as exc:
            self.after(0, self._on_error,
                       f"Missing dependency: {exc}\n\nIf running from source:\n"
                       "    pip install soundcard numpy faster-whisper")
        except Exception as exc:
            self.after(0, self._on_error, str(exc))
        finally:
            self._recording = False
            if asr_thread:
                asr_thread.join(timeout=60)
            if diar_thread:
                diar_thread.join(timeout=10)
            if getattr(self, "_use_cloud", False):
                # Replace the live local preview with Gemini's accurate diarized
                # transcript of the whole recording.
                self._finalize_with_cloud(cfg)
            elif self._diar_on and not self._diar_error_shown:
                # Final, accurate diarization over the COMPLETE recording, once
                # all words are committed — authoritative "who said what".
                try:
                    self.after(0, self.status_var.set, "Finalizing speaker labels…")
                    self._run_full_diarization(hf_token, n_speakers, diar_worker)
                except Exception as exc:
                    self._report_diar_error(str(exc))
            if diar_worker:
                diar_worker.stop()
            self._asr = None   # model itself stays cached in self._whisper_model
            self.after(0, self._on_live_stopped)

    def _snapshot_diar_audio(self):
        """Join the kept audio blocks once. Returns (audio, base_time_seconds)."""
        import numpy as np
        with self._diar_lock:
            if not self._diar_chunks:
                return None, 0.0
            audio = np.concatenate(self._diar_chunks)
            base = (self._captured - len(audio)) / config.SAMPLE_RATE
        return audio, base

    def _finalize_with_cloud(self, cfg):
        """On stop, send the whole recording to Gemini and show its transcript."""
        audio, _ = self._snapshot_diar_audio()
        if audio is None or len(audio) < int(1.0 * config.SAMPLE_RATE):
            return  # nothing worth sending
        try:
            self.after(0, self.status_var.set, "Sending recording to Google Gemini…")
            text = self._cloud_transcribe(audio, cfg)
            self.after(0, self._show_cloud_result, text)
        except Exception as exc:
            # Bind the message now: Python clears `exc` when the except block
            # exits, so a later after()-callback would NameError on it.
            msg = f"{exc}\n\nThe local live transcript is kept."
            self.after(0, self.status_var.set, "Cloud transcription failed — see dialog")
            self.after(0, lambda m=msg: messagebox.showwarning(
                "Cloud transcription failed", m))

    def _show_cloud_result(self, text):
        self._set_text(text)
        self.status_var.set("Done — transcribed by Google Gemini")
        self.copy_btn.config(state=tk.NORMAL)

    # ──────────────────────────────────────────────────── Live: capture ──

    def _open_capture_device(self, source, mic):
        """Return a soundcard device whose .recorder() we can open."""
        import soundcard as sc
        if source == "System audio (loopback)":
            try:
                spk = sc.default_speaker()
                return sc.get_microphone(id=str(spk.name), include_loopback=True)
            except Exception as exc:
                if sys.platform == "darwin":
                    raise RuntimeError(
                        f"Cannot open system-audio loopback.\n\nsoundcard error: {exc}\n\n"
                        "macOS has no native loopback. Install BlackHole, create a "
                        "Multi-Output Device, set it as output, then restart.\n"
                        "  https://github.com/ExistentialAudio/BlackHole") from exc
                raise RuntimeError(
                    f"Cannot open WASAPI loopback.\n\nsoundcard error: {exc}\n\n"
                    "Make sure Windows has a default playback device set "
                    "(Settings → Sound → Output).") from exc
        # Microphone
        selected = mic
        try:
            if not selected or selected == config.MIC_DEFAULT_LABEL:
                return sc.default_microphone()
            return sc.get_microphone(id=selected)
        except Exception as exc:
            raise RuntimeError(
                f"Cannot open the microphone.\n\nsoundcard error: {exc}\n\n"
                "Make sure a microphone is connected and allowed for this app.") from exc

    def _capture_loop(self, device):
        import numpy as np
        import warnings
        # soundcard warns on every minor buffer gap; it recovers on its own and
        # the noise would otherwise spam the transcript via the stderr redirect.
        warnings.filterwarnings("ignore", message="data discontinuity in recording")
        block = int(config.SAMPLE_RATE * config.CAPTURE_BLOCK_SECS)
        maxlen = int(config.DIAR_MAX_SESSION_SEC * config.SAMPLE_RATE)
        with device.recorder(samplerate=config.SAMPLE_RATE, channels=1) as rec:
            while self._recording:
                data = rec.record(numframes=block)
                if data.ndim > 1:
                    data = data.mean(axis=1)
                data = data.astype("float32")
                with self._cap_lock:
                    self._pending_audio.append(data)
                # Dedicated lock so the capture thread never blocks behind the
                # (heavier) ASR/diarization work — a stalled capture loop makes
                # soundcard drop samples ("data discontinuity").
                with self._diar_lock:
                    self._captured += len(data)
                    if self._keep_audio:
                        # Keep the WHOLE session audio so diarization sees full
                        # context (capped to bound memory on long sessions).
                        self._diar_chunks.append(data)
                        self._diar_samples += len(data)
                        while self._diar_samples > maxlen and len(self._diar_chunks) > 1:
                            self._diar_samples -= len(self._diar_chunks.pop(0))

    def _drain_pending(self):
        with self._cap_lock:
            blocks = self._pending_audio
            self._pending_audio = []
        return blocks

    def _has_pending(self):
        with self._cap_lock:
            return bool(self._pending_audio)

    # ──────────────────────────────────────────────────── Live: ASR worker ──

    def _asr_worker(self):
        import numpy as np
        interval = int(config.PROCESS_INTERVAL_SECS * config.SAMPLE_RATE)
        inserted = 0
        last_processed = 0
        while self._recording or self._has_pending():
            blocks = self._drain_pending()
            if blocks:
                chunk = np.concatenate(blocks)
                self._asr.insert_audio(chunk)
                inserted += len(chunk)
            if inserted - last_processed >= interval:
                self._run_one_pass()
                last_processed = inserted
            else:
                time.sleep(0.05)

        # Final pass on whatever is buffered, then flush the tentative tail.
        blocks = self._drain_pending()
        if blocks:
            self._asr.insert_audio(np.concatenate(blocks))
        self._run_one_pass()
        try:
            remaining = self._asr.finish()
        except Exception:
            remaining = []
        with self._words_lock:
            self._words.extend(remaining)
            self._tentative = []
        self._schedule_refresh()

    def _run_one_pass(self):
        try:
            committed, tail = self._asr.process()
        except Exception as exc:
            self.after(0, self.status_var.set, f"Transcription error: {exc}")
            return
        with self._words_lock:
            self._words.extend(committed)
            self._tentative = tail
        self._schedule_refresh()
        if self._recording:
            self.after(0, self.status_var.set,
                       f"Recording… {self._asr.buffered_seconds():.0f}s buffered")

    # ──────────────────────────────────────────────────── Live: diarization worker ──

    def _diar_worker(self, hf_token, n_speakers, worker):
        """Periodically re-diarize the WHOLE recording and relabel every word."""
        last = 0.0
        min_samples = int(config.DIAR_MIN_SEC * config.SAMPLE_RATE)
        while self._recording:
            time.sleep(0.3)
            if time.monotonic() - last < config.DIAR_INTERVAL_SECS:
                continue
            with self._diar_lock:
                have = self._diar_samples >= min_samples
            if not have:
                continue
            try:
                self._run_full_diarization(hf_token, n_speakers, worker)
            except Exception as exc:
                fatal = self._report_diar_error(str(exc))
                if fatal:
                    return  # needs user action + restart; stop retrying
            last = time.monotonic()

    def _run_full_diarization(self, hf_token, n_speakers, worker):
        """Diarize the entire session audio at once and relabel all committed words.

        One diarization over the full audio keeps pyannote's speaker labels
        self-consistent, so we just map them by first-appearance and assign each
        word to its best-overlapping turn (see streaming.assign_by_turns).
        """
        import streaming
        audio, base = self._snapshot_diar_audio()
        if audio is None or len(audio) < int(config.DIAR_MIN_SEC * config.SAMPLE_RATE):
            return
        turns = diar.run_diarize_audio(
            audio, config.SAMPLE_RATE, hf_token, n_speakers, n_speakers, worker=worker)
        # Turns are relative to the kept audio; shift to the absolute session clock.
        turns_abs = [{"start": t["start"] + base, "end": t["end"] + base,
                      "speaker": t["speaker"]} for t in turns]
        with self._words_lock:
            streaming.assign_by_turns(self._words, turns_abs)
        self._schedule_refresh()

    def _report_diar_error(self, msg):
        """Show a clear, actionable note for a diarization failure (once per run).

        Returns True for fatal errors (gated model / bad token) that won't fix
        themselves on retry, so the diarization worker can stop hammering.
        """
        low = msg.lower()
        gated = any(k in low for k in (
            "gated", "403", "authorized list", "awaiting", "access to model",
            "is restricted"))
        bad_token = ("401" in low or "unauthorized" in low
                     or ("token" in low and "invalid" in low))
        if not self._diar_error_shown:
            self._diar_error_shown = True
            if gated:
                friendly = (
                    "Speaker separation is OFF — your HuggingFace account hasn't been "
                    "granted access to the diarization model.\n\n"
                    "Fix (one-time):\n"
                    "1. Sign in at huggingface.co, then open BOTH pages and click "
                    "\"Agree and access\":\n"
                    "     https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                    "     https://huggingface.co/pyannote/segmentation-3.0\n"
                    "2. Make sure your token is set in Advanced.\n"
                    "3. Stop and Start again.\n\n"
                    "Transcription keeps working without speaker labels.")
            elif bad_token:
                friendly = (
                    "Speaker separation is OFF — the HuggingFace token is missing or "
                    "invalid.\n\nSet a valid token in Advanced, then Start again.\n\n"
                    "Transcription keeps working without speaker labels.")
            else:
                friendly = ("Speaker separation is OFF — diarization error:\n\n"
                            + msg[:400]
                            + "\n\nTranscription keeps working without speaker labels.")
            self.after(0, lambda f=friendly: messagebox.showwarning(
                "Speaker separation unavailable", f))
        self.after(0, self.status_var.set,
                   "Speaker separation unavailable — transcribing without labels")
        return gated or bad_token

    # ──────────────────────────────────────────────────── Live: rendering ──

    def _schedule_refresh(self):
        if not self._refresh_scheduled:
            self._refresh_scheduled = True
            self.after(0, self._refresh_live)

    def _build_lines(self, words, tentative, diar_on):
        """Group committed words into (speaker, text) lines + the tentative tail.

        A new line begins whenever the speaker changes, so every contiguous turn
        is its own labelled line. Not-yet-diarized words (speaker None) attach to
        the current line and get relabelled on the next diarization pass.
        """
        lines = []
        if diar_on:
            cur, buf = None, []
            for w in words:
                spk = w.speaker
                if spk is not None and spk != cur:
                    if buf:
                        lines.append((cur, "".join(buf).strip()))
                    cur, buf = spk, [w.text]
                else:
                    buf.append(w.text)
            if buf:
                lines.append((cur, "".join(buf).strip()))
        else:
            lines.append((None, "".join(w.text for w in words).strip()))
        tail = "".join(w.text for w in tentative).strip()
        return lines, tail

    def _speaker_tag(self, speaker):
        """Return (label_tag, text_tag) for a speaker, assigning a color once."""
        pair = self._speaker_tags.get(speaker)
        if pair is None:
            idx = len(self._speaker_tags)
            color = config.SPEAKER_COLORS[idx % len(config.SPEAKER_COLORS)]
            label_tag, text_tag = f"spk{idx}_l", f"spk{idx}_t"
            self.text.tag_config(label_tag, foreground=color,
                                 font=("Segoe UI", 11, "bold"))
            self.text.tag_config(text_tag, foreground=color)
            self._speaker_tags[speaker] = (label_tag, text_tag)
            pair = (label_tag, text_tag)
        return pair

    def _refresh_live(self):
        self._refresh_scheduled = False
        with self._words_lock:
            words = list(self._words)
            tentative = list(self._tentative)
        lines, tail = self._build_lines(words, tentative, self._diar_on)
        has_body = any(text for _, text in lines)
        if not has_body and not tail:
            return  # keep the loading/log header until real text arrives
        sig = (tuple(lines), tail, self._diar_on)
        if sig == self._last_render:
            return
        self._last_render = sig

        self.text.config(state=tk.NORMAL)
        self.text.delete("1.0", tk.END)
        self._redir.reset()  # transcript replaced the console anchor's text

        first = True
        for speaker, text in lines:
            if not text:
                continue
            if not first:
                self.text.insert(tk.END, "\n")
            first = False
            if self._diar_on and speaker:
                label_tag, text_tag = self._speaker_tag(speaker)
                self.text.insert(tk.END, f"{speaker}: ", label_tag)
                self.text.insert(tk.END, text, text_tag)
            else:
                self.text.insert(tk.END, text)
        if tail:
            if not first:
                self.text.insert(tk.END, "\n" if self._diar_on else " ")
            self.text.insert(tk.END, tail, "pending")

        self.text.see(tk.END)
        self.text.config(state=tk.DISABLED)
        self.copy_btn.config(state=tk.NORMAL)

    def _on_live_stopped(self):
        self.btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.copy_btn.config(state=tk.NORMAL if self._has_text() else tk.DISABLED)
        self.status_var.set("Stopped")

    # ──────────────────────────────────────────────────── Text helpers ──

    def _on_error(self, msg):
        self.status_var.set("Error — see transcript area")
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
        self._last_render = None
        if getattr(self, "_redir", None):
            self._redir.reset()  # stale line anchor would delete the wrong range

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
