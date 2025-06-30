import os
import unicodedata
import re

def clean_filename(text):
    # Unicode normalization (e.g., ą → a +  ̨)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")

    # Remove quotation marks and other weird characters
    to_remove = [
        '“', '”', '„', '‟', '«', '»', '＂', '"', '’', '‘', '‚', "'", '‹', '›'
    ]
    for char in to_remove:
        text = text.replace(char, "")

    # Replace other problematic characters
    replacements = {
        '—': '-', '–': '-', '…': '...', '•': '-', 
        ' ': '_',  # ← important: replace spaces with _
    }
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)

    # Remove anything that's not a safe character
    text = re.sub(r'[^A-Za-z0-9_\-\.]', '', text)

    # Reduce multiple underscores
    text = re.sub(r'_+', '_', text)
    
    return text.strip("_")  # removes leading/trailing underscores


def clean_directories_and_files(root_path):
    for dirpath, dirnames, filenames in os.walk(root_path, topdown=False):
        # Files
        for filename in filenames:
            clean_name = clean_filename(filename)
            src = os.path.join(dirpath, filename)
            dst = os.path.join(dirpath, clean_name)
            if src != dst:
                os.rename(src, dst)

        # Directories
        for dirname in dirnames:
            clean_name = clean_filename(dirname)
            src = os.path.join(dirpath, dirname)
            dst = os.path.join(dirpath, clean_name)
            if src != dst:
                os.rename(src, dst)

# 🔧 Usage:
clean_directories_and_files("./dane_trimmed_noise_reduced_cutted")
