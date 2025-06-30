import os
import pandas as pd
import re
from tqdm import tqdm

base_dir = r"./dane_trimmed_noise_reduced_cutted"
rows = []
id_counter = 1

def extract_id(name):
    match = re.search(r'_(\d+)\.', name)
    return match.group(1) if match else None

missing_transcripts = []

for root, dirs, files in tqdm(os.walk(base_dir)):
    wav_files = [f for f in files if f.lower().endswith(".wav")]
    txt_files = [f for f in files if f.lower().endswith(".txt")]

    transcript_map = {
        extract_id(f): os.path.join(root, f)
        for f in txt_files if extract_id(f) is not None
    }

    for wav_file in wav_files:
        audio_id = extract_id(wav_file)
        if audio_id is None:
            print(f"⚠️ Skipped file (missing ID): {wav_file}")
            continue

        wav_path = os.path.join(root, wav_file)
        txt_path = transcript_map.get(audio_id)

        if not txt_path or not os.path.exists(txt_path):
            print(f"❌ Missing transcript for: {wav_file}")
            missing_transcripts.append(wav_file)
            continue

        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if not text:
            print(f"⚠️ Skipped file (empty transcript): {wav_file}")
            continue

        text = text.replace('\n', ' ').replace('\r', ' ')
        text = re.sub(r'[„”"“]', '', text)
        text = re.sub(r'[,\.]+$', '', text).strip()

        if text and not re.search(r'[.?!…]$', text):
            text += '.'

        if not text:
            print(f"⚠️ Skipped file (empty transcript): {wav_file}")
            continue    

        relative_path = os.path.relpath(wav_path, base_dir).replace("\\", "/")

        rows.append({
            "id": id_counter,
            "audio_path": (base_dir + "/" + relative_path),
            "text": text
        })
        id_counter += 1

# Save to CSV
df = pd.DataFrame(rows)
output_csv = os.path.join(base_dir, "output.csv")
df.to_csv(output_csv, index=False, encoding="utf-8", sep='|')

print(f"\n✅ Saved {len(df)} records to: {output_csv}")
print(f"🧾 Skipped {len(missing_transcripts)} WAV files without matching TXT.")

if missing_transcripts:
    print("\n📛 List of missing transcripts:")
