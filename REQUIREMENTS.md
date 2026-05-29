## ⚙️ Requirements

### 🐍 Python
**Python 3.10.11** is required. Other versions are not guaranteed to work due to strict dependency compatibility (especially `speechbrain` and `torch`).

> Download: https://www.python.org/downloads/release/python-31011/

---

### 🎙️ Running the App (`app.py`)

| Package | Version |
|---|---|
| `torch` | 2.2.2+cpu *(or CUDA build)* |
| `torchaudio` | 2.2.2+cpu *(must match torch)* |
| `transformers` | 4.37.2 |
| `speechbrain` | 0.5.16 |
| `noisereduce` | 3.0.3 |
| `matplotlib` | 3.10.8 |
| `sounddevice` | 0.5.5 |
| `soundfile` | 0.13.1 |
| `numpy` | 1.26.4 |

```bash
pip install torch==2.2.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.37.2 speechbrain==0.5.16 noisereduce==3.0.3 matplotlib==3.10.8 sounddevice==0.5.5 soundfile==0.13.1 numpy==1.26.4
```

> For GPU support replace `cpu` with `cu118` or `cu121` in the PyTorch index URL.

---

### 🏋️ Dataset Preparation & Training

In addition to the packages above:

| Package | Used in |
|---|---|
| `datasets` | `SpeechT5_TTS_Fine_tuning.ipynb` |
| `pandas` | `generate_csv_file.py` |
| `yt-dlp` | `download.py` |
| `srt` | `trim_audio_and_update_sub.py` |
| `pysrt` | `cut_files.py` |
| `tqdm` | multiple scripts |

```bash
pip install datasets pandas yt-dlp srt pysrt tqdm
```

> Jupyter and IPython (used in notebooks for inline audio playback) can be installed with: `pip install jupyter`

---

### 🪟 Windows – Additional Requirement

**Microsoft Visual C++ Redistributable (x64)** is required by PyTorch on Windows.

> Download: https://aka.ms/vs/17/release/vc_redist.x64.exe

### 🔗 Symlinks (Windows only)

`speechbrain` requires symlink creation. On Windows, either:
- Enable **Developer Mode**: *Settings → System → For developers → Developer Mode → ON*
- Or run the app **as Administrator**
