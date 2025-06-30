import os

# It also removes "_trimmed" from folder names
root_dir = "./dane_trimmed_noise_reduced"  # Your root directory

for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
    # Check if the folder is empty (no subfolders or files)
    if not dirnames and not filenames:
        try:
            os.rmdir(dirpath)
            print(f"🗑️ Removed empty folder: {dirpath}")
            # No point in renaming after deletion
            continue
        except OSError as e:
            print(f"❌ Error removing {dirpath}: {e}")

    # If the folder exists and has "_trimmed" in its name, rename it
    current_folder = os.path.basename(dirpath)
    if "_trimmed" in current_folder:
        parent_dir = os.path.dirname(dirpath)
        new_name = current_folder.replace("_trimmed", "")
        new_path = os.path.join(parent_dir, new_name)

        # Make sure the new name doesn't conflict with an existing folder
        if not os.path.exists(new_path):
            os.rename(dirpath, new_path)
            print(f"✅ Renamed: {dirpath} → {new_path}")
        else:
            print(f"⚠️ Skipped rename (already exists): {new_path}")
