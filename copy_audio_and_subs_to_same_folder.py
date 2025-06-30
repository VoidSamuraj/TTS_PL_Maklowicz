import os
import shutil
from tqdm import tqdm

audio_root = "./demucs_temp_output/htdemucs"
subs_root = "./dane_trimmed"
output_root = "./dane_trimmed_noise_reduced"

# It removes "#" from directory names
dir_to_clean_hash = "./dane_trimmed"

# Start from the deepest folders to avoid renaming conflicts
for dirpath, dirnames, filenames in os.walk(dir_to_clean_hash, topdown=False):  
    for dirname in dirnames:
        if "#" in dirname:
            old_path = os.path.join(dirpath, dirname)
            new_name = dirname.replace("#", "")
            new_path = os.path.join(dirpath, new_name)

            # Make sure we don't overwrite an existing folder
            if not os.path.exists(new_path):
                os.rename(old_path, new_path)
                print(f"✅ Renamed: {old_path} → {new_path}")
            else:
                print(f"⚠️ Skipped (already exists): {new_path}")

# Copy section

os.makedirs(output_root, exist_ok=True)

for root, _, files in os.walk(audio_root):
    for file in files:
        if file.lower() == "vocals.wav":
            # Get the folder path relative to audio_root
            relative_dir = os.path.relpath(root, audio_root)

            # Remove "_trimmed" from every part of the path
            corrected_relative_dir = os.path.join(*[
                part.replace("_trimmed", "") for part in relative_dir.split(os.sep)
            ])
            
            # Create the output directory while preserving structure
            output_dir = os.path.join(output_root, relative_dir)
            os.makedirs(output_dir, exist_ok=True)

            # Copy vocals.wav
            src_vocals = os.path.join(root, file)
            dst_vocals = os.path.join(output_dir, file)
            shutil.copy2(src_vocals, dst_vocals)

            # Look for the corresponding .srt file in subs_root (without _trimmed)
            subs_dir = os.path.join(subs_root, corrected_relative_dir)
            srt_files = [f for f in os.listdir(subs_dir) if f.lower().endswith(".srt")] if os.path.exists(subs_dir) else []

            if srt_files:
                src_srt = os.path.join(subs_dir, srt_files[0])
                dst_srt = os.path.join(output_dir, srt_files[0])
                shutil.copy2(src_srt, dst_srt)
                print(f"✅ Copied vocals.wav and {srt_files[0]} to {output_dir}")
            else:
                print(f"⚠️ Subtitles not found in {subs_dir} for {corrected_relative_dir}")
