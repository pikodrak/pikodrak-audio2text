import sys
import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext


MODELS = ["tiny", "base", "small", "medium"]
LANGUAGES = ["auto", "cs", "en", "de", "fr", "es", "it", "pl", "sk"]


class Audio2TextApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Audio2Text")
        self.geometry("700x520")
        self.resizable(True, True)
        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self, padding=10)
        top.pack(fill=tk.X)
        ttk.Label(top, text="Audio file:").pack(side=tk.LEFT)
        self.file_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.file_var, width=52).pack(side=tk.LEFT, padx=5)
        ttk.Button(top, text="Browse…", command=self._browse).pack(side=tk.LEFT)

        opts = ttk.Frame(self, padding=(10, 0, 10, 10))
        opts.pack(fill=tk.X)
        ttk.Label(opts, text="Model:").pack(side=tk.LEFT)
        self.model_var = tk.StringVar(value="tiny")
        ttk.Combobox(opts, textvariable=self.model_var, values=MODELS, width=8,
                     state="readonly").pack(side=tk.LEFT, padx=(5, 15))
        ttk.Label(opts, text="Language:").pack(side=tk.LEFT)
        self.lang_var = tk.StringVar(value="auto")
        ttk.Combobox(opts, textvariable=self.lang_var, values=LANGUAGES, width=8,
                     state="readonly").pack(side=tk.LEFT, padx=5)

        btn_frame = ttk.Frame(self, padding=(10, 0, 10, 5))
        btn_frame.pack(fill=tk.X)
        self.btn = ttk.Button(btn_frame, text="Transcribe", command=self._start_transcribe)
        self.btn.pack(side=tk.LEFT)
        self.copy_btn = ttk.Button(btn_frame, text="Copy Text", command=self._copy,
                                   state=tk.DISABLED)
        self.copy_btn.pack(side=tk.LEFT, padx=5)

        out_frame = ttk.LabelFrame(self, text="Transcription", padding=8)
        out_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 5))
        self.text = scrolledtext.ScrolledText(out_frame, wrap=tk.WORD, state=tk.DISABLED,
                                              font=("Segoe UI", 10))
        self.text.pack(fill=tk.BOTH, expand=True)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN,
                  anchor=tk.W, padding=(6, 2)).pack(fill=tk.X, side=tk.BOTTOM)

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

    def _start_transcribe(self):
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
        threading.Thread(target=self._transcribe, args=(path,), daemon=True).start()

    def _transcribe(self, path):
        try:
            from faster_whisper import WhisperModel
            model_name = self.model_var.get()
            lang = self.lang_var.get()
            if lang == "auto":
                lang = None
            self.after(0, self.status_var.set,
                       f"Loading '{model_name}' model (first run downloads ~75 MB)…")
            model = WhisperModel(model_name, compute_type="int8")
            self.after(0, self.status_var.set, "Transcribing…")
            segments, info = model.transcribe(path, language=lang, beam_size=5)
            result = " ".join(seg.text.strip() for seg in segments)
            self.after(0, self._on_done, result, info.language)
        except Exception as exc:
            self.after(0, self._on_error, str(exc))

    def _on_done(self, text, lang):
        self._set_text(text)
        self.status_var.set(f"Done — detected language: {lang}")
        self.btn.config(state=tk.NORMAL)
        self.copy_btn.config(state=tk.NORMAL)

    def _on_error(self, msg):
        self.status_var.set(f"Error: {msg}")
        messagebox.showerror("Transcription failed", msg)
        self.btn.config(state=tk.NORMAL)

    def _set_text(self, text):
        self.text.config(state=tk.NORMAL)
        self.text.delete("1.0", tk.END)
        self.text.insert("1.0", text)
        self.text.config(state=tk.DISABLED)

    def _copy(self):
        text = self.text.get("1.0", tk.END).strip()
        self.clipboard_clear()
        self.clipboard_append(text)
        self.status_var.set("Copied to clipboard")


if __name__ == "__main__":
    app = Audio2TextApp()
    app.mainloop()
