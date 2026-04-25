import sys
import os
import threading
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext


MODELS = ["tiny", "base", "small", "medium"]
LANGUAGES = ["auto", "cs", "en", "de", "fr", "es", "it", "pl", "sk"]
INPUT_SOURCES = ["Audio file", "Microphone", "System audio (loopback)"]

SAMPLE_RATE = 16000
CHUNK_SECONDS = 5  # seconds buffered before each whisper pass


class Audio2TextApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Audio2Text")
        self.geometry("720x580")
        self.resizable(True, True)
        self._recording = False
        self._build_ui()

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
        self.model_var = tk.StringVar(value="tiny")
        ttk.Combobox(opts, textvariable=self.model_var, values=MODELS,
                     width=8, state="readonly").pack(side=tk.LEFT, padx=(5, 15))

        ttk.Label(opts, text="Language:").pack(side=tk.LEFT)
        self.lang_var = tk.StringVar(value="auto")
        ttk.Combobox(opts, textvariable=self.lang_var, values=LANGUAGES,
                     width=8, state="readonly").pack(side=tk.LEFT, padx=5)

        self.translate_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="Translate to English",
                        variable=self.translate_var).pack(side=tk.LEFT, padx=(15, 0))

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
        out_frame = ttk.LabelFrame(self, text="Transcription", padding=8)
        out_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 5))
        self.text = scrolledtext.ScrolledText(
            out_frame, wrap=tk.WORD, state=tk.DISABLED, font=("Segoe UI", 10))
        self.text.pack(fill=tk.BOTH, expand=True)

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
            self.after(0, self.status_var.set,
                       f"Loading '{model_name}' model (first run downloads ~75 MB)…")
            model = WhisperModel(model_name, compute_type="int8")
            self.after(0, self.status_var.set, "Transcribing…")
            segments, info = model.transcribe(path, language=lang, beam_size=5, task=task)
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
        self.status_var.set("Loading model…")
        threading.Thread(
            target=self._live_loop, args=(source,), daemon=True).start()

    def _live_loop(self, source):
        try:
            import numpy as np
            from faster_whisper import WhisperModel

            model_name = self.model_var.get()
            lang = self.lang_var.get() if self.lang_var.get() != "auto" else None
            task = "translate" if self.translate_var.get() else "transcribe"

            self.after(0, self.status_var.set, f"Loading '{model_name}' model…")
            model = WhisperModel(model_name, compute_type="int8")

            if source == "System audio (loopback)":
                self._capture_loopback(model, lang, task)
            else:
                self._capture_microphone(model, lang, task)
        except ImportError as exc:
            self.after(
                0, self._on_error,
                f"Missing dependency: {exc}\n\nInstall with: pip install soundcard numpy",
            )
        except Exception as exc:
            self.after(0, self._on_error, str(exc))
        finally:
            self._recording = False
            self.after(0, self._on_live_stopped)

    def _capture_loopback(self, model, lang, task):
        import numpy as np
        import soundcard as sc

        default_speaker = sc.default_speaker()
        loopback = sc.get_microphone(
            id=str(default_speaker.name), include_loopback=True)

        self.after(
            0, self.status_var.set,
            f"Capturing system audio from: {default_speaker.name}",
        )
        self._run_capture(loopback, model, lang, task)

    def _capture_microphone(self, model, lang, task):
        import soundcard as sc

        mic = sc.default_microphone()
        self.after(0, self.status_var.set, f"Capturing microphone: {mic.name}")
        self._run_capture(mic, model, lang, task)

    def _run_capture(self, device, model, lang, task):
        import numpy as np

        chunk_frames = SAMPLE_RATE * CHUNK_SECONDS

        with device.recorder(samplerate=SAMPLE_RATE, channels=1) as recorder:
            buffer = []
            while self._recording:
                # Record 1 second at a time so we can react to _stop quickly
                data = recorder.record(numframes=SAMPLE_RATE)
                if data.ndim > 1:
                    data = data.mean(axis=1)
                data = data.astype("float32")
                buffer.append(data)

                total = sum(len(d) for d in buffer)
                if total >= chunk_frames:
                    chunk = np.concatenate(buffer)
                    buffer = []
                    self._process_chunk(model, chunk, lang, task)

        # Flush remaining audio when stopped
        if buffer:
            import numpy as np
            remaining = np.concatenate(buffer)
            if len(remaining) > SAMPLE_RATE // 2:  # skip very short tail
                self._process_chunk(model, remaining, lang, task)

    def _process_chunk(self, model, audio, lang, task):
        try:
            segments, info = model.transcribe(
                audio, language=lang, beam_size=5, task=task)
            text = " ".join(seg.text.strip() for seg in segments)
            if text:
                self.after(0, self._append_text, text)
                self.after(
                    0, self.status_var.set,
                    f"Live — detected: {info.language} | processing…",
                )
        except Exception as exc:
            self.after(0, self.status_var.set, f"Chunk error: {exc}")

    def _on_live_stopped(self):
        self.btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.copy_btn.config(state=tk.NORMAL if self._has_text() else tk.DISABLED)
        self.status_var.set("Stopped")

    # --------------------------------------------------------------- Error --

    def _on_error(self, msg):
        self.status_var.set(f"Error: {msg}")
        messagebox.showerror("Error", msg)
        self.btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    # --------------------------------------------------------------- Text helpers --

    def _set_text(self, text):
        self.text.config(state=tk.NORMAL)
        self.text.delete("1.0", tk.END)
        if text:
            self.text.insert("1.0", text)
        self.text.config(state=tk.DISABLED)

    def _append_text(self, text):
        self.text.config(state=tk.NORMAL)
        if self._has_text():
            self.text.insert(tk.END, "\n" + text)
        else:
            self.text.insert("1.0", text)
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
