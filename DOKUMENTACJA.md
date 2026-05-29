# Dokumentacja projektu MaklowiczTTS

---

## Spis treści

1. [Ogólny zarys projektu](#1-ogólny-zarys-projektu)
2. [Aplikacja – app.py](#2-aplikacja--apppy)
   - [Stałe i konfiguracja globalna](#21-stałe-i-konfiguracja-globalna)
   - [lazy_import()](#22-lazy_import)
   - [Klasa TTSApp – init](#23-klasa-ttsapp--__init__)
   - [Budowanie interfejsu](#24-budowanie-interfejsu)
   - [Pomocniki UI](#25-pomocniki-ui)
   - [Log i status](#26-log-i-status)
   - [Waveform](#27-waveform)
   - [Generowanie mowy](#28-generowanie-mowy)
   - [Odtwarzanie i zapis](#29-odtwarzanie-i-zapis)
3. [Pipeline treningu](#3-pipeline-treningu)
   - [download.py](#31-downloadpy)
   - [cut_files.py](#32-cut_filespy)
   - [remove_noise.py](#33-remove_noisepy)
   - [generate_csv_file.py](#34-generate_csv_filepy)
   - [SpeechT5_TTS_Fine_tuning.ipynb](#35-speecht5_tts_fine_tuningipynb)
4. [Kluczowe pojęcia](#4-kluczowe-pojęcia)

---

## 1. Ogólny zarys projektu

Projekt składa się z dwóch niezależnych części:

- **Pipeline treningu** — zestaw skryptów przygotowujący dataset z filmów YouTube i trenujący model TTS (text-to-speech) na polskim głosie.
- **Aplikacja app.py** — desktopowe GUI pozwalające generować mowę z wytrenowanego (lub dowolnego innego) modelu SpeechT5.

Architektura modelu bazuje na **SpeechT5** Microsoftu — transformerze obsługującym zarówno rozumienie jak i syntezę mowy. Do klonowania głosu używane są **x-wektory** (512-wymiarowe embeddingi mówcy) wyciągane przez SpeechBrain. Dźwięk z mel-spektrogramu generuje **HiFi-GAN** — sieć neuronowa pełniąca rolę vocodera.

Cały pipeline wygląda tak:

```
tekst → tokenizer → SpeechT5 → mel-spektrogram → HiFi-GAN → audio PCM 16kHz
                          ↑
               x-wektor mówcy (512-dim)
```

---

## 2. Aplikacja – app.py

### 2.1 Stałe i konfiguracja globalna

Na początku pliku zdefiniowane są:

**Zmienne środowiskowe** (ustawiane przed importami):
- `SPEECHBRAIN_FETCH_FROM_HUB = "1"` — pozwala speechbrain pobierać modele z HuggingFace Hub zamiast szukać lokalnie.
- `K2_DISABLE = "1"` — wyłącza opcjonalną zależność `k2` (biblioteka do dekodowania sekwencji), której instalacja na Windows jest bardzo skomplikowana i do działania projektu niepotrzebna.

**PRESET_MODELS** — słownik mapujący etykietę z dropdownu na identyfikator modelu HuggingFace. Wartość `"__custom__"` jest specjalna — powoduje pokazanie pól do wpisania lokalnych ścieżek zamiast pobierania z internetu.

**DEFAULT_CHECKPOINT / DEFAULT_PROCESSOR** — domyślne ścieżki do lokalnego modelu (folder `results/checkpoint-12600` i `results/outputSave`), budowane względem lokalizacji pliku `app.py`.

**Paleta kolorów** (`DARK_BG`, `PANEL_BG`, `ACCENT` itd.) — ciemny motyw całego interfejsu, zdefiniowany jako stałe hex. Dzięki temu zmiana koloru w jednym miejscu aktualizuje cały wygląd.

**Czcionki** (`FONT_TITLE`, `FONT_LABEL`, `FONT_MONO`, `FONT_BTN`) — wszystkie oparte na Courier New, żeby zachować spójny, terminalowy styl.

---

### 2.2 `lazy_import()`

```python
def lazy_import():
    global torch, torchaudio, sf, nr, textwrap
    ...
```

Funkcja importuje wszystkie ciężkie biblioteki ML — PyTorch, torchaudio, transformers, speechbrain, noisereduce, matplotlib. Celowo nie są importowane na starcie pliku.

**Dlaczego lazy?** PyTorch przy imporcie ładuje do pamięci kilkaset MB. Gdyby import był na górze pliku, okno aplikacji otwierałoby się z kilkusekundowym opóźnieniem. Dzięki lazy_import okno pojawia się natychmiast, a biblioteki ładują się dopiero przy pierwszym naciśnięciu przycisku Generate.

Funkcja używa `global` żeby zmienne były dostępne w reszcie modułu po załadowaniu.

---

### 2.3 Klasa TTSApp – `__init__`

```python
class TTSApp(tk.Tk):
    def __init__(self):
```

TTSApp dziedziczy po `tk.Tk` — jest bezpośrednio głównym oknem aplikacji, nie kontenerem w osobnym oknie.

Inicjalizowane są zmienne stanu:
- `_audio_data` — wygenerowane audio jako numpy array float32, używane do odtwarzania i zapisu.
- `_sample_rate` — zawsze 16000 Hz (SpeechT5 ma stałą częstotliwość wyjściową).
- `_model`, `_processor`, `_vocoder` — załadowane obiekty modeli trzymane w pamięci między generacjami, żeby nie ładować ich przy każdym naciśnięciu.
- `_spk_model` — model SpeechBrain do wyciągania x-wektorów mówcy.
- `_loaded_model_key` — zapamiętuje który model jest aktualnie w RAM, żeby unikać zbędnego przeładowywania.

`_apply_combobox_listbox_style()` musi być wywołane **przed** `_build_ui()` — tkinter wymaga ustawienia opcji listy rozwijanej zanim widżet zostanie stworzony.

---

### 2.4 Budowanie interfejsu

#### `_build_ui()`

Główna metoda konstruująca cały układ. Tworzy **scrollowalny canvas** owijający całą zawartość — jest to konieczne, bo przy małych rozdzielczościach ekranu dolne sekcje (waveform, log) wychodziły poza okno.

Mechanizm scrolla:
1. `tk.Canvas` wypełnia całe okno.
2. `ttk.Scrollbar` po prawej stronie steruje canvasem.
3. Wewnątrz canvasa tworzony jest `inner` — zwykły Frame, w którym siedzi cała zawartość.
4. Binding `<Configure>` na `inner` aktualizuje `scrollregion` canvasa gdy zmienia się rozmiar zawartości.
5. Binding `<Configure>` na canvasie dopasowuje szerokość `inner` do szerokości okna.
6. `<MouseWheel>` obsługuje kółko myszy.

Układ wierszy w `inner`:
- **Row 0** — nagłówek (tytuł + wskaźnik statusu + pozioma linia dekoracyjna)
- **Row 1** — główne ciało: dwie kolumny (lewa 3/5 szerokości, prawa 2/5)
- **Row 2** — wykres waveformu
- **Row 3** — panel logu

#### `_build_left(parent)`

Buduje lewą kolumnę z sekcjami:

- **MODEL** — dropdown z listą modeli. Jeśli wybrano `"[ Your fine-tuned model ]"`, pojawia się `_custom_model_frame` z polami do wpisania ścieżek lokalnych.
- **VOICE EMBEDDING** — pole do wskazania pliku `.wav` referencyjnego głosu. Ten plik służy do wyciągnięcia x-wektora mówcy.
- **TEXT** — pole tekstowe do wpisania tekstu do syntezy. Domyślnie wstawione jest przykładowe zdanie po polsku.
- **OUTPUT** — pole ze ścieżką docelowego pliku `.wav`.

#### `_build_right(parent)`

Buduje prawą kolumnę z sekcjami:

- **PARAMETERS** — cztery parametry generacji:
  - `Chunk size` — maksymalna długość kawałka tekstu w znakach. SpeechT5 traci jakość na długich wejściach, dlatego tekst jest dzielony.
  - `Pause between chunks` — długość ciszy wstawianej między kawałkami (sekundy).
  - `Fade in/out` — czas wygaszania dźwięku na początku i końcu każdego kawałka (zapobiega kliknięciom).
  - `Max audio cap` — maksymalny czas audio na jeden kawałek (sekundy). Ucina zbyt długie wyjście modelu.
- **Noise reduction** — checkbox włączający redukcję szumów po generacji.
- **DEVICE** — wybór urządzenia obliczeniowego: Auto (CUDA jeśli dostępne, inaczej CPU), CPU, CUDA.
- Przyciski: **Generate**, **Play**, **Save**.

#### `_build_waveform(parent)`

Tworzy wykres waveformu używając matplotlib osadzonego w tkinterze (`FigureCanvasTkAgg`). Rysunek ma stałą wysokość (1.6 cala), ciemne tło dopasowane do motywu. Na starcie rysowany jest pusty wykres z tekstem "no audio".

#### `_build_log(parent)`

Tworzy sekcję logu w opakowującym `wrapper` Frame, który zawiera dwie rzeczy spakowane pionowo (pack):
1. **hdr** — nagłówek z etykietą "LOG", poziomą linią dekoracyjną i przyciskami "clear log" / "save log" po prawej.
2. **log_frame** — panel z widżetem `Text` (tylko do odczytu) i scrollbarem. Wiadomości logowane są kolorami: szary = info, zielony = ok, pomarańczowy = warn, czerwony = error, fioletowy = accent.

---

### 2.5 Pomocniki UI

#### `_section(parent, title)`

Tworzy nagłówek sekcji: etykieta z tytułem po lewej + pozioma linia rozciągająca się do prawej krawędzi. Używany wewnątrz paneli z `pack`.

#### `_section_raw(parent, title)`

Identyczny wizualnie do `_section`, ale używa `grid` zamiast `pack` — konieczne dla sekcji WAVEFORM, która jest bezpośrednio w głównym gridzie `inner`.

#### `_path_entry(label, key, parent, is_dir, save)`

Tworzy wiersz z etykietą, polem tekstowym i przyciskiem `…` otwierającym dialog wyboru pliku/folderu. Zmienna tkinter (`StringVar`) jest przypisywana do atrybutu `self._<key>_var` dynamicznie przez `setattr`, co pozwala odczytywać wartości w innych metodach.

Trzy tryby dialogu:
- `save=True` → dialog zapisu pliku
- `is_dir=True` → dialog wyboru folderu
- domyślnie → dialog otwarcia pliku (filtrowany do `.wav`)

#### `_make_btn(parent, text, cmd, bg, fg)`

Fabryka przycisków — zwraca skonfigurowany `tk.Button` z jednolitym stylem (bez ramki, kursor "hand2", padding pionowy).

#### `_style_combo(cb)` i `_apply_combobox_listbox_style()`

Dwie metody stylizujące Combobox:
- `_style_combo` — stylizuje samo pole (wybrany element) przez `ttk.Style`.
- `_apply_combobox_listbox_style` — stylizuje listę rozwijaną przez `option_add`. Musi być wywołana przed stworzeniem widżetu, bo tkinter nie pozwala stylizować Listboxa przez ttk.Style.

#### `_on_model_select()` i `_toggle_custom_frame()`

Obsługa zmiany modelu w dropdownie. `_toggle_custom_frame` sprawdza czy wybrany model to `"__custom__"` — jeśli tak, pokazuje `_custom_model_frame` z polami ścieżek; jeśli nie, ukrywa je przez `pack_forget()`.

---

### 2.6 Log i status

#### `_log(msg, tag)`

Wstawia wiadomość do widżetu logu. Widżet jest na co dzień w trybie `state="disabled"` (tylko odczyt) — metoda tymczasowo go odblokuje, doda tekst z tagiem koloru, przewinie na dół i zablokuje z powrotem.

#### `_clear_log()`

Czyści zawartość logu (kasuje wszystko od `"1.0"` do `"end"`).

#### `_save_log()`

Otwiera dialog zapisu pliku `.txt` i zapisuje aktualną zawartość logu.

#### `_set_status(text, color)`

Aktualizuje etykietę i kolorową kropkę statusu w nagłówku okna.

---

### 2.7 Waveform

#### `_draw_empty_waveform()`

Rysuje pusty wykres z poziomą linią zerową i tekstem "no audio". Wywoływana przy starcie aplikacji.

#### `_draw_waveform(audio_np, sr)`

Rysuje waveform wygenerowanego audio:
1. Tworzy oś czasu `t` przez `np.linspace`.
2. Downsampluje dane (max 4000 próbek na wykres) — wyświetlanie milionów punktów byłoby zbyt wolne.
3. Rysuje wypełniony obszar (`fill_between`) i linię wierzchu (`plot`).
4. Odświeża canvas przez `canvas_widget.draw()`.

Wywoływana z wątku generacji przez `self.after(0, lambda: ...)` — tkinter nie jest thread-safe, więc wszystkie zmiany UI muszą być planowane w głównym wątku przez `after`.

---

### 2.8 Generowanie mowy

#### `_get_device()`

Zwraca nazwę urządzenia do obliczeń. Tryb "auto" sprawdza `torch.cuda.is_available()` i wybiera CUDA jeśli dostępne.

#### `_start_generation()`

Metoda wywoływana przez przycisk Generate:
1. Pobiera tekst z pola tekstowego.
2. Blokuje przycisk (żeby użytkownik nie klikał dwa razy).
3. Uruchamia `_generate_worker` w osobnym wątku (`daemon=True` — wątek umiera razem z aplikacją).

Osobny wątek jest konieczny — generacja trwa sekundy/minuty i blokowanie głównego wątku zamroziłoby GUI.

#### `_generate_worker(text)`

Główna logika generacji, wykonywana w tle:

**1. Lazy import**
Ładuje biblioteki ML jeśli jeszcze nie załadowane.

**2. Ładowanie modeli**
Sprawdza `_loaded_model_key != model_key or _spk_model is None`. Jeśli model nie jest załadowany lub speechbrain się nie załadował (np. po błędzie uprawnień), ładuje:
- `SpeechT5Processor` — tokenizuje tekst na token IDs
- `SpeechT5ForTextToSpeech` — model TTS generujący mel-spektrogram
- `SpeechT5HifiGan` — vocoder zamieniający spektrogram w audio
- `EncoderClassifier` (SpeechBrain) — extractor x-wektorów mówcy

Model ładowany jest tylko raz i trzymany w pamięci (`_loaded_model_key` zapamiętuje który). Przy zmianie modelu stary jest usuwany z pamięci (`del`, `torch.cuda.empty_cache()`).

**3. Speaker embedding**
Ładuje referencyjny plik `.wav`, resampleuje do 16kHz jeśli potrzeba, uśrednia kanały do mono. Przez `encode_batch` SpeechBrain wyciąga x-wektor (512 liczb opisujących charakterystykę głosu). Wektor jest normalizowany L2.

**4. Podział tekstu na chunki**
`textwrap.wrap(text, width=chunk_size)` dzieli tekst na kawałki nie dłuższe niż `chunk_size` znaków, łamiąc w miejscach spacji. Dzielenie jest konieczne bo SpeechT5 traci jakość i "halucynuje" (pomija słowa, powtarza sylaby) na długich wejściach.

**5. Pętla syntezy**
Dla każdego kawałka:
- Tokenizacja przez `processor`
- `model.generate_speech()` → mel-spektrogram
- `vocoder()` → surowe audio (tensor float32)
- Opcjonalna redukcja szumów przez `noisereduce` (używa pierwszych 0.2s jako próbki szumu)
- Ucięcie do `max_samples` (zabezpieczenie przed zbyt długim wyjściem)
- **Fade in/out** — mnożenie przez trapezoidalną kopertę (`np.linspace` od 0 do 1 na początku, od 1 do 0 na końcu). Zapobiega kliknięciom i trzaskom na łączeniach.
- Dodanie ciszy (`pause_t`) po kawałku

**6. Składanie i zapis**
Wszystkie segmenty sklejone przez `torch.cat`. Wynik zapisywany do pliku przez `torchaudio.save`. Waveform rysowany przez `self.after(0, ...)`.

Cała logika owinięta w `try/except` — błędy trafiają do logu zamiast crashować aplikację.

---

### 2.9 Odtwarzanie i zapis

#### `_play_audio()` / `_play_worker()`

Odtwarza `_audio_data` przez `sounddevice.play()`. Uruchamiane w osobnym wątku — `sd.wait()` blokuje do końca odtwarzania, co zamroziłoby GUI w głównym wątku.

#### `_save_audio()`

Otwiera dialog zapisu i zapisuje `_audio_data` przez `soundfile.write()`. Działa w głównym wątku (dialog i zapis są szybkie).

---

## 3. Pipeline treningu

### 3.1 `download.py`

Czyta listę URL z pliku `urls.txt` (jeden link na linię) i dla każdego:
- Pobiera najlepsze audio (lub wideo jeśli brak strumienia audio) przez `yt_dlp`
- Pobiera ręcznie tworzone napisy polskie w formacie SRT
- Konwertuje audio do WAV 192kbps przez FFmpeg
- Zapisuje w folderze `./dane/<tytuł_video>/`

Ręcznie tworzone napisy są kluczowe — auto-generowane mają błędy wyrównania i literówki, co psuje trening.

### 3.2 `cut_files.py`

Dla każdej pary plik audio + napisy SRT:
- Parsuje napisy przez `pysrt` — każdy wpis ma czas początku, końca i tekst
- Wycina odpowiadający fragment audio z paddingiem 0.2s po obu stronach (padding zapobiega urywaniu pierwszej/ostatniej sylaby)
- Zapisuje pary `<numer>.wav` + `<numer>.txt` w folderze wyjściowym

Wycinanie jest konieczne bo model trenuje na krótkich klipach (kilka sekund). Pełne nagrania trwają kilkadziesiąt minut.

### 3.3 `remove_noise.py`

Opcjonalny krok. Uruchamia **Demucs** — sieć neuronową do separacji źródeł dźwięku — w trybie `--two-stems vocals`. Demucs rozdziela audio na ścieżkę wokalną i tło muzyczne/szumy, zachowując tylko głos.

W praktyce dla tego datasetu okazało się, że redukcja szumów **pogarszała** jakość — materiał był już wystarczająco czysty, a Demucs wprowadzał artefakty. Dlatego krok jest oznaczony jako opcjonalny.

### 3.4 `generate_csv_file.py`

Przechodzi przez folder z parami `<id>.wav` + `<id>.txt` i generuje plik CSV:

```
id|audio_path|text
001|audio/clip_001.wav|Dzień dobry.
```

Po drodze:
- Paruje pliki na podstawie numerycznego ID wyciągniętego regexem z nazwy pliku
- Czyści tekst (usuwa niestandardowe cudzysłowy, trailing punctuation)
- Loguje pominięte pliki (bez pary)

CSV jest formatem wejściowym dla notebooka treningowego.

### 3.5 `SpeechT5_TTS_Fine_tuning.ipynb`

Notebook Jupyter realizujący właściwy trening:

**Ładowanie datasetu** — wczytuje CSV, ładuje pliki WAV, resampleuje do 16kHz.

**Normalizacja tekstu** — obsługa polskich znaków diakrytycznych, wielkich liter, cyfr.

**Speaker embeddings** — dla każdej próbki wyciągany jest x-wektor przez SpeechBrain (`spkrec-xvect-voxceleb`). W datasecie jest jeden mówca, więc wszystkie embeddingi są do siebie podobne — model uczy się kojarzyć ten konkretny głos z dźwiękiem.

**Fine-tuning** — parametry:
- Learning rate: `6e-5`
- Batch size: 16–32
- Maksymalna liczba kroków: 18 000
- Optymizer: AdamW z warmupem

**Checkpointy** — co N kroków trening zapisuje stan modelu do osobnego folderu (`checkpoint-<krok>`). Dzięki temu można wznowić trening po awarii lub wybrać najlepszy punkt zamiast ostatniego.

---

## 4. Kluczowe pojęcia

**Checkpoint** — migawka wag modelu zapisana w trakcie treningu. Każdy checkpoint to osobny folder z plikami wag (`.safetensors`), konfiguracji i stanu optymizera. Pozwala wrócić do wcześniejszego etapu treningu jeśli model "przeuczył się" (overfitting).

**Chunk (kawałek tekstu)** — fragment zdania, na który podzielony jest długi tekst przed syntezą. SpeechT5 jest podatny na halucynacje (opuszczanie słów, zapętlanie) przy długich wejściach — dlatego tekst jest dzielony na kawałki ≤60 znaków, syntetyzowany oddzielnie i sklejany z krótkimi pauzami.

**X-wektor / Speaker embedding** — 512-wymiarowy wektor liczb opisujący charakterystykę głosu mówcy (ton, barwa, tempo). Wyciągany przez SpeechBrain z referencyjnego nagrania. Podawany do modelu SpeechT5 podczas generacji — model "wie" jak ma brzmieć głos i stara się go naśladować.

**Mel-spektrogram** — wizualna reprezentacja dźwięku: oś X to czas, oś Y to częstotliwości w skali mel (logarytmiczna, zbliżona do ludzkiego słuchu), jasność to głośność danej częstotliwości w danym momencie. SpeechT5 generuje spektrogram jako pośredni krok — nie generuje audio bezpośrednio.

**HiFi-GAN (vocoder)** — sieć neuronowa zamieniająca mel-spektrogram w surowe audio. Trenowany osobno przez Microsoft na dużych datasetach angielskich, używany tu bez modyfikacji jako gotowy komponent.

**Fine-tuning** — trening wstępnie wytrenowanego modelu na nowych danych. Zamiast uczyć modelu od zera (co wymagałoby miesięcy i ogromnych datasetów), startuje się od istniejących wag i "dostraja" je na polskim głosie. Model zachowuje wiedzę o strukturze mowy, a uczy się nowych cech językowych i głosowych.

**Demucs** — sieć neuronowa do separacji źródeł audio (music source separation). W trybie `--two-stems vocals` rozdziela nagranie na: głos + reszta. Używana opcjonalnie do czyszczenia datasetu z szumów tła.

**16 kHz** — częstotliwość próbkowania audio używana przez SpeechT5. Oznacza 16 000 próbek na sekundę. Wystarczająca dla mowy (zakres 300–3400 Hz), ale niższa niż CD (44 100 Hz) — dlatego wygenerowane audio brzmi jak przez telefon w porównaniu do HD audio.

**`torch.no_grad()`** — kontekst wyłączający obliczanie gradientów podczas inferencji (generacji). Gradienty są potrzebne tylko podczas treningu. Wyłączenie ich przyspiesza generację i zmniejsza zużycie pamięci.

**Daemon thread** — wątek potomny który automatycznie kończy się gdy główny wątek (aplikacja) się zamknie. Używany do generacji i odtwarzania — gdyby były to zwykłe wątki, zamknięcie okna czekałoby na ich zakończenie.
