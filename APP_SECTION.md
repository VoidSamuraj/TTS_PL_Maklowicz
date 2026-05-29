---

## 🖥️ 4. TTS Studio App

`app.py` is a desktop GUI application for generating speech using any SpeechT5-based model — including your own fine-tuned checkpoint or any model from HuggingFace.

![App Preview](app_preview.png)

### ✨ Features

- 🎛️ **Multiple model support** — switch between your local fine-tuned model or any preset HuggingFace model from a dropdown
- 🎤 **Voice cloning** — provide a reference `.wav` file to extract a speaker embedding and clone the voice
- 🔇 **Noise reduction** — optional spectral noise gate applied to generated audio
- 📊 **Waveform preview** — live waveform plot of the generated audio
- 💾 **Export** — save generated audio as `.wav`
- ⚙️ **Adjustable parameters** — chunk size, pause between chunks, fade in/out, max audio length

### 🚀 How to run

```bash
python app.py
```

### 📋 Available models (preset dropdown)

| Label | HuggingFace ID |
|---|---|
| Your fine-tuned model | *(local path)* |
| d190305/speecht5_finetuned_pl_inteligentne_full | Polish ⭐ |
| d190305/speecht5_finetuned_voxpopuli_pl_full_dataset | Polish |
| Sagicc/speecht5_finetuned_multilingual_librispeech_pl | Polish (base) |
| microsoft/speecht5_tts | English (base) |

You can also point the app to **any local checkpoint folder** by selecting *"Your fine-tuned model"* and providing the paths manually.

### ⚠️ Tips & Limitations

> **Keep phrases short.** SpeechT5 is prone to hallucinations on long inputs — it may skip words, repeat syllables, or produce garbled output. Aim for **sentences under ~60 characters**. The app automatically splits text into chunks (configurable via *Chunk size* parameter).

- Shorter, natural sentences produce the most stable results
- The model outputs audio at a fixed **16 kHz** sample rate
- First generation after launch takes longer — models are loaded lazily on first use
