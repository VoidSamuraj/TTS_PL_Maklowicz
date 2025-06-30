import os
import pysrt
import soundfile as sf
from tqdm import tqdm

PADDING = 0.2  # seconds

input_folder_path = "./dane_trimmed"
output_folder_path = "./dane_trimmed_noise_reduced_cutted"

def cut_audio(input_audio, start_time, end_time, output_path):
    # Load the entire audio file
    data, samplerate = sf.read(input_audio)
    
    start_sample = int(start_time * samplerate)
    end_sample = int(end_time * samplerate)
    
    # Ensure we don't go beyond the length of the audio
    end_sample = min(end_sample, len(data))
    
    fragment = data[start_sample:end_sample]
    
    # Save the fragment as WAV (16-bit PCM)
    sf.write(output_path, fragment, samplerate, subtype='PCM_16')

def process_audio_and_subtitles(audio_file, srt_file, output_dir):
    # Load the .srt subtitle file
    subs = pysrt.open(srt_file)

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over subtitles and cut audio fragments
    for idx, sub in enumerate(tqdm(subs, desc=audio_file)):
        # Convert subtitle times to seconds
        start_time = max(0, convert_time_to_seconds(sub.start) - PADDING)
        end_time = convert_time_to_seconds(sub.end) + PADDING
        text = sub.text

        # Path for the saved audio fragment
        output_audio_path = os.path.join(output_dir, f"audio_{idx+1}.wav")
        
        # Cut the audio fragment
        cut_audio(audio_file, start_time, end_time, output_audio_path)

        # Save the transcript to a text file
        transcript_file = os.path.join(output_dir, f"transcript_{idx+1}.txt")
        with open(transcript_file, 'w', encoding='utf-8') as f:
            f.write(text)

def convert_time_to_seconds(time_obj):
    hours = time_obj.hours
    minutes = time_obj.minutes
    seconds = time_obj.seconds
    milliseconds = time_obj.milliseconds
    total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0
    return total_seconds


for foldername in tqdm(os.listdir(input_folder_path), desc="Processing folders"):
    folder_path = os.path.join(input_folder_path, foldername)
    if os.path.isdir(folder_path):
        # Look for audio files (e.g., with .wav extension)
        audio_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.wav')]
        if not audio_files:
            print(f"No audio file in folder {foldername}, skipping...")
            continue
        audio_file = os.path.join(folder_path, audio_files[0])  # take the first audio file

        # Look for subtitle files (e.g., with .srt extension)
        srt_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.srt')]
        if not srt_files:
            print(f"No .srt file in folder {foldername}, skipping...")
            continue

        # take the first .srt file
        srt_file = os.path.join(folder_path, srt_files[0])  
        output_dir = os.path.join(output_folder_path, foldername)
        os.makedirs(output_dir, exist_ok=True)
        process_audio_and_subtitles(audio_file, srt_file, output_dir)
