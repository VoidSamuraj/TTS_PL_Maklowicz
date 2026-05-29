# Omówienie projektu MaklowiczTTS

Projekt składa się z dwóch części: **pipeline przygotowania danych i treningu modelu** (skrypty + notebook) oraz **aplikacji desktopowej** (`app.py`) do generowania mowy.

---

## Część I – Trening modelu

### Ogólna idea

Celem było nauczenie modelu TTS (Text-to-Speech) głosu konkretnej osoby. Użyto podejścia **fine-tuningu** — zamiast trenować od zera, wzięto istniejący wytrenowany model języka polskiego i dostosowano go do nowego głosu na nowych danych.

Model bazowy: `Sagicc/speecht5_finetuned_multilingual_librispeech_pl`
Architektura: **SpeechT5** od Microsoft — model sekwencja-do-sekwencji (tekst → spektrogram mel → audio)

---

### Skrypty przygotowania danych

#### `download.py`
Pobiera filmy z YouTube wraz z napisami.
- Czyta listę linków z pliku `urls.txt`
- Używa biblioteki `yt_dlp` do pobierania audio w formacie WAV
- Pobiera ręcznie tworzone napisy w języku polskim (format SRT)
- Zapisuje pliki w folderze `./dane/`, posegregowane według tytułu

Dlaczego napisy ręczne, a nie auto-generowane? Bo automatyczne napisy YouTube mają błędy i złe synchronizacje — ręczne są znacznie dokładniejsze, co jest kluczowe dla wyrównania tekstu z dźwiękiem.

---

#### `trim_audio_and_update_sub.py`
Przycina całe pliki audio do fragmentów odpowiadających napisom.
- Parsuje pliki SRT w celu odczytania znaczników czasu
- Wyodrębnia fragmenty audio przez FFmpeg z marginesem 1 sekundy
- Przelicza znaczniki czasu dla ciągłego wyjścia
- Scala fragmenty w jeden plik
- Wyjście: WAV 44100 Hz, PCM 16-bit

Opcjonalny krok — jeśli i tak tniemy pliki w późniejszym kroku (`cut_files.py`), to jego rola może być zbędna.

---

#### `clean_weird_digits.py`
Naprawia dziwne znaki w nazwach plików i folderów (np. niestandardowe cyfry z Unicode), które mogą powodować błędy w późniejszych krokach przetwarzania.

---

#### `remove_noise.py`
Opcjonalne usuwanie szumu tła z nagrań. W praktyce okazało się, że dla czystych nagrań to pogarszało jakość — więc używać ostrożnie.

---

#### `copy_audio_and_subs_to_same_folder.py`
Porządkuje strukturę folderów — kopiuje pliki audio i napisy do wspólnych folderów. Używane po opcjonalnym kroku usuwania szumu.

---

#### `cut_files.py`
Kluczowy skrypt — tnie długie pliki audio na krótkie fragmenty zsynchronizowane z napisami.
- Czyta pliki WAV i SRT
- Dla każdego wpisu w napisach wycina odpowiadający fragment audio
- Dodaje padding 0.2 s przed i po każdym fragmencie
- Zapisuje krótkie klipy WAV (16-bit PCM) + pliki TXT z transkrypcją

**Dlaczego krótkie fragmenty?** Model SpeechT5 ma ograniczenie długości wejścia. Krótkie klipy (kilka sekund) są bardziej stabilne podczas treningu i łatwiej je wyrównać z tekstem.

---

#### `generate_csv_file.py`
Generuje plik CSV z metadanymi datasetu wymaganymi przez HuggingFace `datasets`.
- Przechodzi przez foldery z plikami WAV i TXT
- Normalizuje tekst (usuwa cudzysłowy, trailing punctuation, zapewnia kropkę na końcu)
- Pomija puste transkrypcje i niepasujące pary
- Wyjście: CSV z separatorem `|`, kolumny: `id | audio_path | text`

---

### `SpeechT5_TTS_Fine_tuning.ipynb` — notebook treningowy

Główny etap projektu. Notebook trenuje model SpeechT5 na przygotowanym datasecie.

**Kluczowe parametry treningu:**
| Parametr | Wartość |
|---|---|
| Batch size (trening) | 16 |
| Batch size (ewaluacja) | 32 |
| Learning rate | 6e-5 |
| Warmup steps | 2000 |
| Max steps | 18 000 |
| Gradient accumulation | 2 kroki |
| Optimizer | AdamW + cosine scheduler |
| Precyzja | FP16 (half precision) |
| Ewaluacja co | 300 kroków |
| Checkpoint co | 900 kroków |

**Przebieg treningu:**
1. Ładowanie modelu bazowego SpeechT5
2. Ładowanie datasetu z CSV (audio 16 kHz, tekst)
3. Generowanie **speaker embeddings** — 512-wymiarowych wektorów opisujących głos mówcy, wyodrębnionych przez `speechbrain/spkrec-xvect-voxceleb`
4. Dodanie polskich znaków (ąćęłńóśźż) do tokenizera
5. Filtrowanie: audio 1–12 s, tekst < 200 tokenów
6. Podział 90/10 (trening / test)
7. Pętla treningowa z walidacją

**Co to jest checkpoint?**
Checkpoint to zapis stanu modelu (wagi sieci neuronowej) w danym momencie treningu. Dzięki temu można wrócić do wcześniejszego stanu jeśli model zaczął się przeuczać, lub wybrać najlepszy checkpoint zamiast ostatniego.

**Co to są speaker embeddings (x-vector)?**
To 512-liczbowy wektor opisujący charakterystykę głosu konkretnej osoby — barwę, rytm, sposób mówienia. Model podczas generowania mowy dostaje ten wektor jako dodatkowe wejście i próbuje naśladować ten głos. Dzięki temu można "klonować" głos podając plik WAV referencyjny.

**Co to jest mel spectrogram?**
Pośrednia reprezentacja audio — zamiast surowych próbek dźwięku (fale), to obraz częstotliwości w czasie, w skali mel (zbliżonej do percepcji ludzkiego ucha). SpeechT5 generuje właśnie spektrogram, a dopiero vocoder (HifiGAN) zamienia go z powrotem na audio.

---

## Część II – Aplikacja `app.py`

Aplikacja desktopowa napisana w Tkinter umożliwiająca generowanie mowy przez dowolny model SpeechT5, w tym wytrenowany własny.

---

### Stałe globalne

**`PRESET_MODELS`** — słownik mapujący etykiety z dropdownu na identyfikatory modeli HuggingFace. Wartość `"__custom__"` jest specjalna — powoduje wyświetlenie pól do podania ścieżek lokalnych.

**Stałe kolorów** (`DARK_BG`, `PANEL_BG`, `ACCENT` itd.) — paleta ciemnego motywu używana w całym UI.

**Stałe fontów** (`FONT_TITLE`, `FONT_LABEL`, `FONT_MONO`, `FONT_BTN`) — definicje czcionek dla różnych elementów.

---

### `lazy_import()`

Importuje wszystkie ciężkie biblioteki ML (torch, transformers, speechbrain itd.) dopiero przy pierwszym kliknięciu Generate, a nie przy starcie aplikacji. Dzięki temu okno otwiera się natychmiastowo, zamiast czekać kilka sekund na załadowanie PyTorcha.

---

### Klasa `TTSApp`

Główna klasa aplikacji, dziedziczy po `tk.Tk` (jest samym oknem).

#### `__init__`
Inicjalizuje okno (rozmiar 1100×1000, ciemne tło, tytuł), ustawia zmienne stanu:
- `_audio_data` — wygenerowane audio jako numpy array
- `_model`, `_processor`, `_vocoder`, `_spk_model` — załadowane modele ML
- `_loaded_model_key` — który model jest aktualnie w pamięci (cache)

Przed budowaniem UI wywołuje `_apply_combobox_listbox_style()` — musi być przed stworzeniem widgetów.

---

### Budowanie interfejsu

#### `_apply_combobox_listbox_style()`
Stylizuje listę rozwijaną Combobox. `ttk.Style` nie kontroluje listy (jest zwykłym Listbox), więc jedynym sposobem jest `option_add()` wywołane **przed** stworzeniem widgetu.

#### `_build_ui()`
Tworzy scrollowalny canvas jako kontener dla całego UI. Wszystkie elementy są umieszczone w wewnętrznym frame (`inner`) wewnątrz canvasa.
- Wiąże `<Configure>` canvasa z aktualizacją szerokości `inner`
- Wiąże `<Configure>` `inner` z aktualizacją `scrollregion`
- Wiąże `<MouseWheel>` z przewijaniem kółkiem myszy
- Układ: row 0 = nagłówek, row 1 = dwa panele (lewy/prawy), row 2 = waveform, row 3 = log

#### `_build_left(parent)`
Buduje lewy panel z sekcjami:
- **MODEL** — dropdown z presetami + opcjonalne pola ścieżek dla własnego modelu
- **VOICE EMBEDDING** — pole do wskazania pliku WAV referencyjnego
- **TEXT** — pole tekstowe z domyślnym przykładem
- **OUTPUT** — pole do zapisu pliku wyjściowego

#### `_build_right(parent)`
Buduje prawy panel z sekcjami:
- **PARAMETERS** — 4 parametry generowania (chunk size, pause, fade, max cap)
- Checkbox **Noise reduction**
- **DEVICE** — radiobuttons: Auto / CPU / CUDA
- 3 przyciski: Generate, Play, Save

#### `_build_waveform(parent)`
Tworzy sekcję WAVEFORM z wykresem matplotlib osadzonym w Tkinter przez `FigureCanvasTkAgg`. Domyślnie rysuje pusty wykres przez `_draw_empty_waveform()`.

#### `_build_log(parent)`
Tworzy sekcję LOG w wrapperze — nagłówek z przyciskami "clear log" i "save log", pod nim pole tekstowe (disabled, z własnym scrollem) do wyświetlania komunikatów.

---

### Helpery UI

#### `_section(parent, title)`
Tworzy nagłówek sekcji: etykieta z tytułem + pozioma linia (`tk.Frame` o height=1) rozciągająca się do końca wiersza. Używa `pack`.

#### `_section_raw(parent, title)`
Jak `_section`, ale używa `grid` zamiast `pack` — dla sekcji WAVEFORM, która jest w kontekście gridowym.

#### `_path_entry(label, key, parent, is_dir, save)`
Tworzy wiersz z etykietą, polem tekstowym i przyciskiem `…` otwierającym dialog wyboru pliku/folderu. Zmienna `StringVar` jest zapisywana jako `self._<key>_var`.

#### `_make_btn(parent, text, cmd, bg, fg)`
Fabryka przycisków — tworzy przycisk z jednolitym stylem (flat, Courier New, cursor hand2).

#### `_style_combo(cb)`
Stylizuje widoczną część Combobox przez `ttk.Style` z motywem "clam".

#### `_on_model_select(_)`
Handler zdarzenia zmiany wyboru w dropdownie — wywołuje `_toggle_custom_frame()`.

#### `_toggle_custom_frame()`
Pokazuje lub ukrywa pola ścieżek do własnego modelu w zależności od tego, czy wybrano `"__custom__"`.

---

### Log i status

#### `_log(msg, tag)`
Dodaje wpis do pola logów. Tag determinuje kolor: `info`=szary, `ok`=zielony, `warn`=żółty, `error`=czerwony, `accent`=fioletowy. Widget jest na co dzień `disabled` (nieedytowalny) — metoda tymczasowo go odblokowuje, wpisuje tekst i blokuje z powrotem.

#### `_clear_log()`
Czyści całą zawartość pola logów.

#### `_save_log()`
Zapisuje zawartość logów do pliku TXT przez dialog zapisu.

#### `_set_status(text, color)`
Aktualizuje etykietę statusu i kropkę-wskaźnik w prawym górnym rogu nagłówka.

---

### Waveform

#### `_draw_empty_waveform()`
Rysuje pusty wykres z napisem "no audio" — wywoływany przy starcie i po wyczyszczeniu.

#### `_draw_waveform(audio_np, sr)`
Rysuje przebieg wygenerowanego audio.
- `audio_np` — numpy array z próbkami
- `sr` — sample rate (zawsze 16000 Hz)
- Downsampluje do max 4000 punktów dla wydajności (`step = len // 4000`)
- Rysuje wypełnienie (fill_between) + linię przebiegów

---

### Generowanie mowy

#### `_get_device()`
Na podstawie ustawienia radiobutton zwraca: `"cuda"` (jeśli dostępne i wybrano Auto/CUDA) lub `"cpu"`.

#### `_start_generation()`
Wywoływana przez przycisk Generate. Waliduje że tekst nie jest pusty, blokuje przycisk, ustawia status "generating…" i uruchamia `_generate_worker` w osobnym wątku (żeby nie blokować UI).

#### `_generate_worker(text)`
Główna logika generowania — działa w tle. Kroki:

1. **Lazy import** — ładuje biblioteki ML przy pierwszym uruchomieniu
2. **Ładowanie modeli** — jeśli model nie jest załadowany lub `_spk_model` jest None, ładuje:
   - `SpeechT5Processor` — tokenizer tekstu
   - `SpeechT5ForTextToSpeech` — główny model TTS
   - `SpeechT5HifiGan` — vocoder (spektrogram → audio)
   - `EncoderClassifier` (speechbrain) — ekstraktor x-vector
3. **Speaker embedding** — ładuje WAV referencyjny, resampuje do 16 kHz, uśrednia kanały, generuje 512-dim x-vector i normalizuje go
4. **Podział tekstu na chunki** — `textwrap.wrap` dzieli tekst na fragmenty o zadanej długości znaków (domyślnie 60). Krótkie chunki = mniej halucynacji
5. **Pętla syntezy** — dla każdego chunka:
   - Tokenizacja tekstu przez processor
   - `model.generate_speech()` — generuje mel spektrogram
   - `vocoder()` — zamienia spektrogram na audio PCM
   - Opcjonalna redukcja szumu (`noisereduce`)
   - Przycięcie do max długości
   - Fade in/out — envelope o kształcie trapezowym (linspace 0→1, jedynki, linspace 1→0)
   - Dodanie ciszy między chunkami
6. **Składanie** — `torch.cat` łączy wszystkie segmenty w jeden tensor
7. **Zapis** — przez `torchaudio.save` do pliku WAV
8. **Aktualizacja UI** — przez `self.after(0, ...)` (bo aktualizacje UI muszą być z głównego wątku)

---

### Odtwarzanie

#### `_play_audio()`
Sprawdza czy jest audio i uruchamia `_play_worker` w wątku.

#### `_play_worker()`
Odtwarza audio przez `sounddevice.play()` + `sd.wait()` (blokuje wątek do końca odtwarzania).

---

### Zapis

#### `_save_audio()`
Otwiera dialog zapisu i zapisuje `_audio_data` jako WAV przez `soundfile.write()`.

---

## Kluczowe koncepcje

| Termin | Co to jest |
|---|---|
| **Fine-tuning** | Dostosowanie wytrenowanego modelu do nowych danych, zamiast trenowania od zera |
| **Checkpoint** | Zapisany stan modelu (wagi) w danym momencie treningu |
| **Mel spectrogram** | Pośrednia reprezentacja dźwięku jako obraz częstotliwości w czasie |
| **Vocoder (HifiGAN)** | Sieć neuronowa zamieniająca spektrogram na surowe audio |
| **Speaker embedding / x-vector** | 512-liczbowy wektor opisujący charakterystykę głosu mówcy |
| **Chunk** | Fragment tekstu — podział długiego tekstu na krótkie kawałki przed syntezą |
| **Fade in/out** | Stopniowe wyciszanie początku i końca fragmentu audio, żeby nie było kliknięć |
| **Noise reduction** | Spektralne bramkowanie szumu — wyciszanie częstotliwości niższych niż próg szumu |
| **Lazy import** | Opóźnione ładowanie bibliotek do momentu gdy są faktycznie potrzebne |
| **Daemon thread** | Wątek pomocniczy który kończy się razem z programem głównym |
