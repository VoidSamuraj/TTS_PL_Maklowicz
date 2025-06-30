# STEP 1
import os
import yt_dlp

url_file = "urls.txt"

if not os.path.exists(url_file):
    print(f"File {url_file} does not exist!")
    exit(1)

with open(url_file, 'r') as file:
    urls = [url.strip() for url in file if url.strip()]

for url in urls:
    print(f"Downloading from: {url}")

    ydl_opts = {
        'format': 'bestaudio/best',  # download best audio (or video if no audio available)
        'outtmpl': "./dane/%(title)s/%(title)s.%(ext)s",
        'writesubtitles': True,
        'subtitleslangs': ['pl'],
        'subtitlesformat': 'srt',
        'postprocessors': [{             # convert audio to wav after download
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': False,
        'no_warnings': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        print(f"Error at {url}: {e}")

print("Downloading completed.")