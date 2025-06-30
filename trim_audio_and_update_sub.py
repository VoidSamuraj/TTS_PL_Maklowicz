# STEP 2
import os
import srt
import subprocess
from datetime import timedelta

MARGIN_SECONDS = 1
main_folder_path = "./dane"
output_folder_path = "./dane_trimmed"
temp_segments_dir = "./temp_segments"

def format_time(seconds: float):
    """Format float seconds to ffmpeg-friendly HH:MM:SS.MS format"""
    td = timedelta(seconds=seconds)
    return str(td)

def cut_audio_using_srt_ffmpeg(audio_file, srt_file, output_audio_file, output_srt_file, margin=MARGIN_SECONDS):
    os.makedirs(temp_segments_dir, exist_ok=True)

    with open(srt_file, "r", encoding="utf-8") as f:
        subtitles = list(srt.parse(f.read()))

    new_subtitles = []
    current_time = 0.0
    segment_paths = []

    for i, subtitle in enumerate(subtitles):
        start_time = subtitle.start.total_seconds()
        end_time = subtitle.end.total_seconds()

        # Add margins between subtitles
        if i > 0:
            prev_end = subtitles[i - 1].end.total_seconds()
            if start_time - prev_end > margin:
                start_time -= margin
        if i < len(subtitles) - 1:
            next_start = subtitles[i + 1].start.total_seconds()
            if next_start - end_time > margin:
                end_time += margin

        duration = end_time - start_time
        start_ffmpeg = format_time(start_time)
        duration_ffmpeg = format_time(duration)

        segment_path = os.path.join(temp_segments_dir, f"segment_{i:04d}.wav")
        segment_paths.append(segment_path)

        # Cut segment using ffmpeg
        cmd = [
            "ffmpeg", "-y", "-i", audio_file,
            "-ss", start_ffmpeg,
            "-t", duration_ffmpeg,
            "-acodec", "pcm_s16le",
            "-ar", "44100",
            "-ac", "2",
            segment_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Update subtitle timestamps
        new_start = current_time
        new_end = current_time + duration
        new_subtitles.append(
            srt.Subtitle(
                index=subtitle.index,
                start=timedelta(seconds=new_start),
                end=timedelta(seconds=new_end),
                content=subtitle.content
            )
        )
        current_time = new_end

    # Create a concat list for ffmpeg
    concat_list_path = os.path.join(temp_segments_dir, "concat_list.txt")
    with open(concat_list_path, "w", encoding="utf-8") as f:
        for path in segment_paths:
            f.write(f"file '{os.path.abspath(path)}'\n")

    # Concatenate all segments into one file
    cmd_concat = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", concat_list_path, "-acodec", "pcm_s16le", output_audio_file
    ]
    subprocess.run(cmd_concat, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Write updated SRT file
    with open(output_srt_file, "w", encoding="utf-8") as f:
        f.write(srt.compose(new_subtitles))

    # Clean up temporary files
    try:
        for path in segment_paths + [concat_list_path]:
            os.remove(path)
        os.rmdir(temp_segments_dir)
    except Exception as e:
        print(f"Cleanup failed: {e}")

    print(f"Saved new audio file: {output_audio_file}")
    print(f"Updated subtitle file: {output_srt_file}")

if __name__ == "__main__":
    os.makedirs(output_folder_path, exist_ok=True)

    for subdir, dirs, files in os.walk(main_folder_path):
        for file in files:
            if file.endswith(".wav"):
                base_name = os.path.splitext(file)[0]
                audio_path = os.path.join(subdir, file)

                # Assuming subtitle file ends with .pl.srt, change if needed
                srt_path = os.path.join(subdir, base_name + ".pl.srt")
                if not os.path.exists(srt_path):
                    # If .pl.srt is missing, try standard .srt
                    srt_path = os.path.join(subdir, base_name + ".srt")

                if os.path.exists(srt_path):
                    relative_subfolder = os.path.relpath(subdir, main_folder_path)
                    output_subfolder = os.path.join(output_folder_path, relative_subfolder)
                    os.makedirs(output_subfolder, exist_ok=True)

                    output_audio_path = os.path.join(output_subfolder, f"{base_name}_trimmed.wav")
                    output_srt_path = os.path.join(output_subfolder, f"{base_name}_trimmed.srt")

                    print(f"▶️ Processing: {audio_path}")
                    cut_audio_using_srt_ffmpeg(audio_path, srt_path, output_audio_path, output_srt_path)
                else:
                    print(f"⚠️ No SRT file for: {audio_path}")
