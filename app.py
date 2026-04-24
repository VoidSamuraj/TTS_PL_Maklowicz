# SpeechT5 TTS Studio
# Tkinter GUI for text-to-speech synthesis with voice cloning.
#
# Pipeline overview:
#   text -> tokenizer -> SpeechT5 model -> mel spectrogram
#        -> HifiGAN vocoder -> PCM audio (16kHz float32)
#   voice cloning is done via a 512-dim x-vector extracted from a reference .wav
#
# Dependencies:
#   torch, torchaudio   - tensor ops and audio I/O
#   transformers        - SpeechT5 model and processor from HuggingFace
#   speechbrain         - x-vector speaker encoder
#   noisereduce         - spectral noise gating on generated audio
#   matplotlib          - waveform plot embedded in the window
#   sounddevice         - system audio playback
#   soundfile           - writing .wav files

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import numpy as np
import os
import warnings
import traceback

os.environ["SPEECHBRAIN_FETCH_FROM_HUB"] = "1"
os.environ["K2_DISABLE"] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain")

# ---------------------------------------------------------------------------
# lazy_import()
#   Imports all heavy ML libraries in one shot.
#   Called on the first Generate press, not at app startup — keeps the
#   window opening instantly instead of waiting for PyTorch to load.
# ---------------------------------------------------------------------------
def lazy_import():
    global torch, torchaudio, sf, nr, textwrap
    global SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
    global EncoderClassifier

    import torch
    import torchaudio
    import soundfile as sf
    import noisereduce as nr
    import textwrap
    from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
    from speechbrain.pretrained import EncoderClassifier
    import matplotlib
    matplotlib.use("TkAgg")   # must be set before any other matplotlib import
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# PRESET_MODELS
#   Maps dropdown label -> HuggingFace model ID.
#   "__custom__" is special: shows local path fields instead of downloading.
#   Your fine-tuned model is first so it is selected by default on startup.
# ---------------------------------------------------------------------------
PRESET_MODELS = {
    "[ Your fine-tuned model ]":
        "__custom__",
    "d190305/speecht5_finetuned_pl_inteligentne_full  (PL ⭐)":
        "d190305/speecht5_finetuned_pl_inteligentne_full",
    "d190305/speecht5_finetuned_voxpopuli_pl_full_dataset  (PL)":
        "d190305/speecht5_finetuned_voxpopuli_pl_full_dataset",
    "Sagicc/speecht5_finetuned_multilingual_librispeech_pl  (PL, baza)":
        "Sagicc/speecht5_finetuned_multilingual_librispeech_pl",
    "microsoft/speecht5_tts  (EN, base)":
        "microsoft/speecht5_tts",
}

# ---------------------------------------------------------------------------
# Default paths for the custom model — relative to the script's own folder.
# Assumes the folder layout:
#   app.py
#   results\
#     checkpoint-12600\    <- model weights
#     outputSave\          <- processor / tokenizer
# ---------------------------------------------------------------------------
_SCRIPT_DIR        = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHECKPOINT = os.path.join(_SCRIPT_DIR, "results", "checkpoint-12600")
DEFAULT_PROCESSOR  = os.path.join(_SCRIPT_DIR, "results", "outputSave")

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
DARK_BG   = "#0f0f13"
PANEL_BG  = "#16161e"
ACCENT    = "#7c6af7"
ACCENT2   = "#c084fc"
TEXT_MAIN = "#e8e6f0"
TEXT_DIM  = "#6b6880"
BORDER    = "#2a2840"
SUCCESS   = "#4ade80"
WARN      = "#f59e0b"
ERROR     = "#f87171"

FONT_TITLE = ("Courier New", 22, "bold")
FONT_LABEL = ("Courier New", 10)
FONT_MONO  = ("Courier New", 9)
FONT_BTN   = ("Courier New", 11, "bold")


# ===========================================================================
# TTSApp
# ===========================================================================
class TTSApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SpeechT5 TTS Studio")
        self.configure(bg=DARK_BG)
        self.geometry("1100x1000")
        self.minsize(900, 800)
        self.resizable(True, True)

        self._audio_data       = None   # generated audio, float32 numpy array [N]
        self._sample_rate      = 16000  # SpeechT5 fixed output rate
        self._model            = None
        self._processor        = None
        self._vocoder          = None   # HifiGAN: mel spectrogram -> waveform
        self._spk_model        = None   # SpeechBrain x-vector extractor
        self._loaded_model_key = None   # which model is currently in RAM

        # Ustaw kolory listy rozwijanej PRZED zbudowaniem UI
        # (option_add musi być wywołane przed stworzeniem widgetów)
        self._apply_combobox_listbox_style()

        self._build_ui()
        self._log("Ready. Configure paths and press Generate.", "info")

    # =========================================================================
    # Combobox dropdown list styling
    # =========================================================================

    def _apply_combobox_listbox_style(self):
        """
        Tkinter's ttk.Style nie kontroluje listy rozwijanej Combobox —
        jest ona zwykłym Listbox. Jedynym sposobem na jej ostylowanie
        jest option_add(), które musi być wywołane przed utworzeniem widgetu.
        """
        self.option_add("*TCombobox*Listbox.background",       PANEL_BG)
        self.option_add("*TCombobox*Listbox.foreground",       TEXT_MAIN)
        self.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
        self.option_add("*TCombobox*Listbox.selectForeground", "white")
        self.option_add("*TCombobox*Listbox.font",             FONT_MONO)
        self.option_add("*TCombobox*Listbox.relief",           "flat")

    # =========================================================================
    # UI construction
    # =========================================================================

    def _build_ui(self):
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        # ── Scrollable canvas wrapping all content ─────────────────────────
        canvas  = tk.Canvas(self, bg=DARK_BG, highlightthickness=0)
        vscroll = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vscroll.set)
        vscroll.grid(row=0, column=1, sticky="ns")
        canvas.grid(row=0, column=0, sticky="nsew")

        inner  = tk.Frame(canvas, bg=DARK_BG)
        win_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(win_id, width=e.width))
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

        inner.rowconfigure(0, weight=0)
        inner.rowconfigure(1, weight=0)
        inner.rowconfigure(2, weight=0)
        inner.rowconfigure(3, weight=0)
        inner.columnconfigure(0, weight=1)

        # ── Row 0: header ──────────────────────────────────────────────────
        hdr = tk.Frame(inner, bg=DARK_BG)
        hdr.grid(row=0, column=0, sticky="ew", padx=28, pady=(22, 0))
        tk.Label(hdr, text="SpeechT5",   font=FONT_TITLE, fg=ACCENT,    bg=DARK_BG).pack(side="left")
        tk.Label(hdr, text=" TTS Studio",font=FONT_TITLE, fg=TEXT_MAIN, bg=DARK_BG).pack(side="left")
        self._status_dot = tk.Label(hdr, text="●", font=("Courier New", 18), fg=TEXT_DIM, bg=DARK_BG)
        self._status_dot.pack(side="right", padx=4)
        self._status_lbl = tk.Label(hdr, text="idle", font=FONT_MONO, fg=TEXT_DIM, bg=DARK_BG)
        self._status_lbl.pack(side="right")

        tk.Frame(inner, height=1, bg=BORDER).grid(row=0, column=0, sticky="ew",
                                                   padx=28, pady=(64, 0))

        # ── Row 1: two-column body ─────────────────────────────────────────
        body = tk.Frame(inner, bg=DARK_BG)
        body.grid(row=1, column=0, sticky="ew", padx=28, pady=(10, 6))
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)

        left  = tk.Frame(body, bg=DARK_BG)
        right = tk.Frame(body, bg=DARK_BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        right.grid(row=0, column=1, sticky="nsew")

        self._build_left(left)
        self._build_right(right)

        # ── Row 2: waveform ────────────────────────────────────────────────
        self._build_waveform(inner)

        # ── Row 3: log ────────────────────────────────────────────────────
        self._build_log(inner)

    def _build_left(self, parent):
        self._section(parent, "MODEL")

        self._model_row = tk.Frame(parent, bg=DARK_BG)
        self._model_row.pack(fill="x", pady=(0, 4))
        self._model_var = tk.StringVar(value=list(PRESET_MODELS.keys())[0])
        cb = ttk.Combobox(self._model_row, textvariable=self._model_var,
                          values=list(PRESET_MODELS.keys()),
                          state="readonly", font=FONT_MONO)
        cb.pack(side="left", fill="x", expand=True)
        cb.bind("<<ComboboxSelected>>", self._on_model_select)
        self._style_combo(cb)

        self._custom_model_frame = tk.Frame(parent, bg=DARK_BG)
        self._path_entry("Checkpoint (folder):", "checkpoint_path",
                         self._custom_model_frame, is_dir=True)
        self._path_entry("Processor (folder):",  "processor_path",
                         self._custom_model_frame, is_dir=True)
        self._checkpoint_path_var.set(DEFAULT_CHECKPOINT)
        self._processor_path_var.set(DEFAULT_PROCESSOR)

        self._toggle_custom_frame()

        self._section(parent, "VOICE EMBEDDING")
        self._path_entry("Reference .wav:", "embedding_path", parent, is_dir=False)

        self._section(parent, "TEXT")
        self._text_box = tk.Text(parent, height=6, font=FONT_MONO,
                                 bg=PANEL_BG, fg=TEXT_MAIN,
                                 insertbackground=ACCENT, relief="flat", bd=0,
                                 wrap="word", highlightthickness=1,
                                 highlightbackground=BORDER, highlightcolor=ACCENT)
        self._text_box.pack(fill="x", pady=(0, 6))
        self._text_box.insert("1.0",
            "Proszę Państwa, oto zupa instant, danie które łączy aromat "
            "kartonu z teksturą wilgotnej szmaty.")

        self._section(parent, "OUTPUT")
        self._path_entry("Save as (.wav):", "output_path", parent, is_dir=False, save=True)

    def _build_right(self, parent):
        self._section(parent, "PARAMETERS")

        params = [
            ("Chunk size (chars):",    "chunk_size", "60"),
            ("Pause between chunks:",  "pause_dur",  "0.2"),
            ("Fade in/out (s):",       "fade_dur",   "0.05"),
            ("Max audio cap (s):",     "max_cap",    "6.0"),
        ]
        self._param_vars = {}
        for label, key, default in params:
            fr = tk.Frame(parent, bg=DARK_BG)
            fr.pack(fill="x", pady=3)
            tk.Label(fr, text=label, width=24, anchor="w",
                     font=FONT_LABEL, fg=TEXT_DIM, bg=DARK_BG).pack(side="left")
            var = tk.StringVar(value=default)
            tk.Entry(fr, textvariable=var, width=8,
                     font=FONT_MONO, bg=PANEL_BG, fg=TEXT_MAIN,
                     insertbackground=ACCENT, relief="flat", bd=4,
                     highlightthickness=1, highlightbackground=BORDER,
                     highlightcolor=ACCENT).pack(side="left")
            self._param_vars[key] = var

        fr2 = tk.Frame(parent, bg=DARK_BG)
        fr2.pack(fill="x", pady=6)
        self._noise_var = tk.BooleanVar(value=True)
        tk.Checkbutton(fr2, text="Noise reduction", variable=self._noise_var,
                       font=FONT_LABEL, fg=TEXT_DIM, bg=DARK_BG,
                       activebackground=DARK_BG, selectcolor=PANEL_BG,
                       activeforeground=TEXT_MAIN).pack(side="left")

        self._section(parent, "DEVICE")
        self._device_var = tk.StringVar(value="auto")
        for txt, val in [("Auto (CUDA -> CPU)", "auto"), ("CPU", "cpu"), ("CUDA", "cuda")]:
            tk.Radiobutton(parent, text=txt, variable=self._device_var, value=val,
                           font=FONT_LABEL, fg=TEXT_DIM, bg=DARK_BG,
                           activebackground=DARK_BG, selectcolor=PANEL_BG,
                           activeforeground=TEXT_MAIN).pack(anchor="w")

        tk.Frame(parent, height=1, bg=BORDER).pack(fill="x", pady=12)

        self._gen_btn  = self._make_btn(parent, "▶  Generate", self._start_generation, bg=ACCENT,    fg="white")
        self._play_btn = self._make_btn(parent, "♪  Play",     self._play_audio,        bg=PANEL_BG, fg=ACCENT2)
        self._save_btn = self._make_btn(parent, "💾  Save",    self._save_audio,        bg=PANEL_BG, fg=SUCCESS)
        for btn in (self._gen_btn, self._play_btn, self._save_btn):
            btn.pack(fill="x", pady=3)

    def _build_waveform(self, parent):
        self._section_raw(parent, "WAVEFORM")
        wave_frame = tk.Frame(parent, bg=PANEL_BG,
                              highlightthickness=1, highlightbackground=BORDER)
        wave_frame.grid(row=2, column=0, sticky="ew", padx=28, pady=(0, 4))

        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        self._fig = Figure(figsize=(9, 1.6), dpi=96, facecolor=PANEL_BG)
        self._ax  = self._fig.add_subplot(111)
        self._ax.set_facecolor(PANEL_BG)
        self._ax.tick_params(colors=TEXT_DIM, labelsize=7)
        for spine in self._ax.spines.values():
            spine.set_edgecolor(BORDER)
        self._fig.tight_layout(pad=0.4)

        self._canvas_widget = FigureCanvasTkAgg(self._fig, master=wave_frame)
        self._canvas_widget.get_tk_widget().pack(fill="x")
        self._draw_empty_waveform()

    def _build_log(self, parent):
        wrapper = tk.Frame(parent, bg=DARK_BG)
        wrapper.grid(row=3, column=0, sticky="nsew", padx=28, pady=(6, 12))

        hdr = tk.Frame(wrapper, bg=DARK_BG)
        hdr.pack(fill="x", pady=(0, 4))
        tk.Label(hdr, text="LOG", font=("Courier New", 8, "bold"),
                 fg=ACCENT, bg=DARK_BG).pack(side="left")
        tk.Frame(hdr, height=1, bg=BORDER).pack(side="left", fill="x",
                                                  expand=True, padx=(6, 0), pady=6)
        tk.Button(hdr, text="💾 save log", font=FONT_MONO,
                  bg=PANEL_BG, fg=TEXT_DIM,
                  activebackground=ACCENT, activeforeground="white",
                  relief="flat", bd=0, padx=8, cursor="hand2",
                  command=self._save_log).pack(side="right", padx=(4, 0))
        tk.Button(hdr, text="✕ clear log", font=FONT_MONO,
                  bg=PANEL_BG, fg=TEXT_DIM,
                  activebackground=ACCENT, activeforeground="white",
                  relief="flat", bd=0, padx=8, cursor="hand2",
                  command=self._clear_log).pack(side="right", padx=(4, 0))

        log_frame = tk.Frame(wrapper, bg=PANEL_BG,
                             highlightthickness=1, highlightbackground=BORDER)
        log_frame.pack(fill="both", expand=True)

        self._log_text = tk.Text(log_frame, font=FONT_MONO,
                                 bg=PANEL_BG, fg=TEXT_DIM,
                                 relief="flat", bd=6, height=14,
                                 state="disabled", wrap="word")
        scroll = ttk.Scrollbar(log_frame, command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self._log_text.pack(fill="both", expand=True)

        self._log_text.tag_config("info",   foreground=TEXT_DIM)
        self._log_text.tag_config("ok",     foreground=SUCCESS)
        self._log_text.tag_config("warn",   foreground=WARN)
        self._log_text.tag_config("error",  foreground=ERROR)
        self._log_text.tag_config("accent", foreground=ACCENT2)

    # =========================================================================
    # UI helpers
    # =========================================================================

    def _section(self, parent, title):
        fr = tk.Frame(parent, bg=DARK_BG)
        fr.pack(fill="x", pady=(10, 2))
        tk.Label(fr, text=title, font=("Courier New", 8, "bold"),
                 fg=ACCENT, bg=DARK_BG).pack(side="left")
        tk.Frame(fr, height=1, bg=BORDER).pack(side="left", fill="x",
                                                expand=True, padx=(6, 0), pady=6)

    def _section_raw(self, parent, title):
        fr = tk.Frame(parent, bg=DARK_BG)
        fr.grid(row=2, column=0, sticky="ew", padx=28, pady=(8, 0))
        tk.Label(fr, text=title, font=("Courier New", 8, "bold"),
                 fg=ACCENT, bg=DARK_BG).pack(side="left")
        tk.Frame(fr, height=1, bg=BORDER).pack(side="left", fill="x",
                                                expand=True, padx=(6, 0), pady=6)

    def _path_entry(self, label, key, parent, is_dir=False, save=False):
        fr = tk.Frame(parent, bg=DARK_BG)
        fr.pack(fill="x", pady=2)
        tk.Label(fr, text=label, width=22, anchor="w",
                 font=FONT_LABEL, fg=TEXT_DIM, bg=DARK_BG).pack(side="left")
        var = tk.StringVar()
        setattr(self, f"_{key}_var", var)
        tk.Entry(fr, textvariable=var, font=FONT_MONO, bg=PANEL_BG, fg=TEXT_MAIN,
                 insertbackground=ACCENT, relief="flat", bd=4,
                 highlightthickness=1, highlightbackground=BORDER,
                 highlightcolor=ACCENT).pack(side="left", fill="x", expand=True)

        def browse():
            if save:
                path = filedialog.asksaveasfilename(defaultextension=".wav",
                                                    filetypes=[("WAV files", "*.wav")])
            elif is_dir:
                path = filedialog.askdirectory()
            else:
                path = filedialog.askopenfilename(
                    filetypes=[("WAV files", "*.wav"), ("All", "*.*")])
            if path:
                var.set(path)

        tk.Button(fr, text="…", font=FONT_MONO, bg=PANEL_BG, fg=ACCENT2,
                  activebackground=ACCENT, activeforeground="white",
                  relief="flat", bd=0, padx=8, cursor="hand2",
                  command=browse).pack(side="left", padx=(4, 0))

        if key == "output_path":
            var.set("output_tts.wav")

    def _make_btn(self, parent, text, cmd, bg=PANEL_BG, fg=TEXT_MAIN):
        return tk.Button(parent, text=text, command=cmd, font=FONT_BTN,
                         bg=bg, fg=fg, activebackground=ACCENT,
                         activeforeground="white", relief="flat", bd=0,
                         pady=8, cursor="hand2")

    def _style_combo(self, cb):
        """
        Stylizuje pole Combobox (górna część z wybraną wartością).
        Lista rozwijana (Listbox) jest stylizowana przez _apply_combobox_listbox_style()
        wywoływane w __init__ przed budowaniem UI.
        """
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TCombobox",
                        fieldbackground=PANEL_BG,
                        background=PANEL_BG,
                        foreground=TEXT_MAIN,
                        selectbackground=ACCENT,
                        selectforeground="white",
                        arrowcolor=ACCENT,
                        bordercolor=BORDER,
                        lightcolor=BORDER,
                        darkcolor=BORDER)
        style.map("TCombobox",
                  fieldbackground=[("readonly", PANEL_BG), ("disabled", PANEL_BG)],
                  foreground=[("readonly", TEXT_MAIN),     ("disabled", TEXT_DIM)],
                  background=[("readonly", PANEL_BG),      ("disabled", PANEL_BG)],
                  selectbackground=[("readonly", ACCENT)],
                  selectforeground=[("readonly", "white")])

    def _on_model_select(self, _=None):
        self._toggle_custom_frame()

    def _toggle_custom_frame(self):
        if PRESET_MODELS.get(self._model_var.get()) == "__custom__":
            self._custom_model_frame.pack(fill="x", pady=(0, 6),
                                          after=self._model_row)
        else:
            self._custom_model_frame.pack_forget()

    # =========================================================================
    # Log and status
    # =========================================================================

    def _log(self, msg, tag="info"):
        self._log_text.configure(state="normal")
        self._log_text.insert("end", f"» {msg}\n", tag)
        self._log_text.see("end")
        self._log_text.configure(state="disabled")

    def _clear_log(self):
        self._log_text.configure(state="normal")
        self._log_text.delete("1.0", "end")
        self._log_text.configure(state="disabled")

    def _save_log(self):
        content = self._log_text.get("1.0", "end").strip()
        if not content:
            messagebox.showinfo("Empty log", "Nothing to save.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".txt",
                                            filetypes=[("Text files", "*.txt")])
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        self._log(f"Log saved: {path}", "ok")

    def _set_status(self, text, color=TEXT_DIM):
        self._status_lbl.config(text=text)
        self._status_dot.config(fg=color)

    # =========================================================================
    # Waveform
    # =========================================================================

    def _draw_empty_waveform(self):
        self._ax.clear()
        self._ax.set_facecolor(PANEL_BG)
        self._ax.axhline(0, color=BORDER, linewidth=0.8)
        self._ax.set_xlim(0, 1)
        self._ax.set_ylim(-1, 1)
        self._ax.tick_params(colors=TEXT_DIM, labelsize=7)
        for spine in self._ax.spines.values():
            spine.set_edgecolor(BORDER)
        self._ax.text(0.5, 0, "no audio", ha="center", va="center",
                      color=TEXT_DIM, fontsize=9, fontfamily="Courier New")
        self._canvas_widget.draw()

    def _draw_waveform(self, audio_np, sr):
        self._ax.clear()
        self._ax.set_facecolor(PANEL_BG)
        t    = np.linspace(0, len(audio_np) / sr, num=len(audio_np))
        step = max(1, len(audio_np) // 4000)
        t_ds, a_ds = t[::step], audio_np[::step]

        self._ax.fill_between(t_ds, a_ds, alpha=0.35, color=ACCENT)
        self._ax.plot(t_ds, a_ds, color=ACCENT2, linewidth=0.6)
        self._ax.axhline(0, color=BORDER, linewidth=0.5)
        self._ax.set_xlim(0, t[-1])
        self._ax.set_ylim(-1, 1)
        self._ax.tick_params(colors=TEXT_DIM, labelsize=7)
        self._ax.set_xlabel("time [s]", color=TEXT_DIM, fontsize=7,
                             fontfamily="Courier New")
        for spine in self._ax.spines.values():
            spine.set_edgecolor(BORDER)
        self._fig.tight_layout(pad=0.4)
        self._canvas_widget.draw()

    # =========================================================================
    # Generation
    # =========================================================================

    def _get_device(self):
        import torch
        d = self._device_var.get()
        return ("cuda" if torch.cuda.is_available() else "cpu") if d == "auto" else d

    def _start_generation(self):
        text = self._text_box.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("No text", "Enter text to synthesize.")
            return
        self._gen_btn.config(state="disabled")
        self._set_status("generating…", WARN)
        threading.Thread(target=self._generate_worker,
                         args=(text,), daemon=True).start()

    def _generate_worker(self, text):
        try:
            lazy_import()
            self._log("Imports OK.", "ok")
            device = self._get_device()
            self._log(f"Device: {device}", "accent")
            
            if self._model is not None:
                del self._model
                del self._vocoder
                del self._spk_model
                if device == "cuda":
                    torch.cuda.empty_cache()
                self._model = self._vocoder = self._spk_model = None

            # ── Model loading ──────────────────────────────────────────────
            model_key = self._model_var.get()
            hf_path   = PRESET_MODELS[model_key]

            if self._loaded_model_key != model_key or self._spk_model is None:
                self._log("Loading model…", "warn")

                if hf_path == "__custom__":
                    ckpt      = self._checkpoint_path_var.get()
                    proc_path = self._processor_path_var.get()
                    if not ckpt or not proc_path:
                        raise ValueError("Provide checkpoint and processor paths.")
                    self._processor = SpeechT5Processor.from_pretrained(proc_path)
                    self._model     = SpeechT5ForTextToSpeech.from_pretrained(ckpt).to(device)
                else:
                    self._processor = SpeechT5Processor.from_pretrained(hf_path)
                    self._model     = SpeechT5ForTextToSpeech.from_pretrained(hf_path).to(device)

                self._vocoder   = SpeechT5HifiGan.from_pretrained(
                    "microsoft/speecht5_hifigan").to(device)
                self._spk_model = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-xvect-voxceleb",
                    run_opts={"device": device})
                self._loaded_model_key = model_key
                self._log("Model loaded ✓", "ok")

            # ── Speaker embedding ──────────────────────────────────────────
            emb_path = self._embedding_path_var.get()
            if not emb_path or not os.path.exists(emb_path):
                raise FileNotFoundError("Embedding .wav not found.")

            waveform, sr = torchaudio.load(emb_path)
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
            waveform = waveform.mean(dim=0, keepdim=True)

            with torch.no_grad():
                spk_emb = self._spk_model.encode_batch(waveform.to(device))
                spk_emb = torch.nn.functional.normalize(spk_emb, dim=2)
                spk_emb = spk_emb.squeeze().unsqueeze(0)
            self._log("Speaker embedding ready ✓", "ok")

            # ── Parameters ────────────────────────────────────────────────
            chunk_size = int(self._param_vars["chunk_size"].get())
            pause_dur  = float(self._param_vars["pause_dur"].get())
            fade_dur   = float(self._param_vars["fade_dur"].get())
            max_cap    = float(self._param_vars["max_cap"].get())
            do_noise   = self._noise_var.get()

            chunks  = textwrap.wrap(text, width=chunk_size)
            self._log(f"Split into {len(chunks)} chunk(s).", "info")

            all_segments = []
            pause_t      = torch.zeros((1, int(16000 * pause_dur)))

            # ── Synthesis loop ─────────────────────────────────────────────
            for i, chunk in enumerate(chunks):
                self._log(f"[{i+1}/{len(chunks)}] {chunk[:50]}…", "accent")
                max_samples = int(16000 * min(len(chunk) / 1.0 + 0.5, max_cap))

                inputs = self._processor(text=chunk, return_tensors="pt").to(device)
                with torch.no_grad():
                    spec   = self._model.generate_speech(inputs["input_ids"], spk_emb)
                    speech = self._vocoder(spec)

                audio_np = speech.squeeze().cpu().numpy()

                if do_noise:
                    import noisereduce as nr
                    audio_np = nr.reduce_noise(
                        y=audio_np,
                        y_noise=audio_np[:int(0.2 * 16000)],
                        sr=16000)

                speech = torch.tensor(audio_np).unsqueeze(0)
                if speech.shape[1] > max_samples:
                    speech = speech[:, :max_samples]

                n   = speech.shape[1]
                fs  = int(16000 * fade_dur)
                env = np.concatenate([
                    np.linspace(0, 1, fs),
                    np.ones(max(0, n - 2 * fs)),
                    np.linspace(1, 0, fs)
                ])[:n]
                speech = speech * torch.tensor(env).unsqueeze(0)

                all_segments.append(speech.cpu())
                all_segments.append(pause_t)

            # ── Assemble and output ────────────────────────────────────────
            final = torch.cat(all_segments, dim=1)
            self._audio_data  = final.squeeze().numpy()
            self._sample_rate = 16000

            self.after(0, lambda: self._draw_waveform(self._audio_data, 16000))
            self._log("Generation complete ✓", "ok")

            out = self._output_path_var.get()
            if out:
                torchaudio.save(out, final.cpu(), sample_rate=16000)
                self._log(f"Saved: {out}", "ok")

            self.after(0, lambda: self._set_status("ready", SUCCESS))

        except Exception as e:
            self._log(f"ERROR: {e}", "error")
            self._log(traceback.format_exc(), "error")
            self.after(0, lambda: self._set_status("error", ERROR))
        finally:
            self.after(0, lambda: self._gen_btn.config(state="normal"))

    # =========================================================================
    # Playback
    # =========================================================================

    def _play_audio(self):
        if self._audio_data is None:
            messagebox.showinfo("No audio", "Generate speech first.")
            return
        threading.Thread(target=self._play_worker, daemon=True).start()

    def _play_worker(self):
        try:
            import sounddevice as sd
            self._log("Playing…", "accent")
            sd.play(self._audio_data, self._sample_rate)
            sd.wait()
            self._log("Playback finished.", "info")
        except ImportError:
            self._log("Install 'sounddevice' to enable playback.", "warn")
        except Exception as e:
            self._log(f"Playback error: {e}", "error")
            self._log(traceback.format_exc(), "error")

    # =========================================================================
    # Save
    # =========================================================================

    def _save_audio(self):
        if self._audio_data is None:
            messagebox.showinfo("No audio", "Generate speech first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".wav",
                                            filetypes=[("WAV files", "*.wav")])
        if not path:
            return
        import soundfile as sf
        sf.write(path, self._audio_data, self._sample_rate)
        self._log(f"Saved to: {path}", "ok")


if __name__ == "__main__":
    app = TTSApp()
    app.mainloop()