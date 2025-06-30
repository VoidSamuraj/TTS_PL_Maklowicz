import os
import subprocess
import shutil
from tqdm import tqdm

# Paths
input_folder_path = "./dane_trimmed"
output_folder_path = "./dane_trimmed_noise_reduced"
temp_output_root = "./demucs_temp_output"
os.makedirs(output_folder_path, exist_ok=True)

# Collect WAV files (recursively)
all_wav_files = []
for root, _, files in os.walk(input_folder_path):
    for file in files:
        if file.lower().endswith(".wav"):
            full_path = os.path.join(root, file)
            all_wav_files.append(full_path)

print(f"Found {len(all_wav_files)} WAV files to process.")

for wav_path in tqdm(all_wav_files, desc="Processing WAV files with Demucs"):
    try:
        base_name = os.path.splitext(os.path.basename(wav_path))[0]

        # Relative path (e.g., folder1/audio.wav)
        relative_path = os.path.relpath(wav_path, input_folder_path)
        relative_dir = os.path.dirname(relative_path)

        # Run Demucs
        subprocess.run([
            "python", "-m", "demucs",
            "--two-stems", "vocals",
            "-o", temp_output_root,
            wav_path
        ], check=True)

        # Path to extracted vocals
        temp_vocals_path = os.path.join(
            temp_output_root, "separated", "htdemucs", base_name, "vocals.wav"
        )

        # Target directory structure
        final_output_dir = os.path.join(output_folder_path, relative_dir)
        os.makedirs(final_output_dir, exist_ok=True)
        final_output_path = os.path.join(final_output_dir, base_name + ".wav")

        # Move the audio file
        shutil.move(temp_vocals_path, final_output_path)

        # Copy corresponding .srt file if it exists
        srt_path = os.path.join(os.path.dirname(wav_path), base_name + ".srt")
        if os.path.exists(srt_path):
            shutil.copy2(srt_path, os.path.join(final_output_dir, base_name + ".srt"))

    except Exception as e:
        print(f"❌ Error while processing {wav_path}: {e}")
