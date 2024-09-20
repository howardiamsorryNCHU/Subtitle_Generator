from faster_whisper import WhisperModel
import srt
import os
import glob
from datetime import timedelta
os.environ['KMP_DUPLICATE_LIB_OK']='True'

model = WhisperModel("large-v3", device="cuda", compute_type="int8_float16")

def transcribe_to_srt(path):
    segments, info = model.transcribe(path, task="translate")
    # Create subtitles
    subtitles = []
    for segment in segments:
        start = timedelta(seconds=segment.start)
        end = timedelta(seconds=segment.end)
        content = segment.text
        subtitle = srt.Subtitle(index=len(subtitles)+1, start=start, end=end, content=content)
        subtitles.append(subtitle)
        print("\n" + content)

    # Write to .srt file
    with open(os.path.splitext(path)[0] + ".srt", 'w', encoding='utf-8') as srt_file:
        srt_file.write(srt.compose(subtitles))

def list_audio_files(folder_path):
    audio_extensions = ['*.mp3', '*.wav', '*.flac', '*.aac', '*.ogg', '*.m4a', "*.mp4"]

    audio_files = []
    for ext in audio_extensions:
        # Using glob to find files with the specified extensions
        files = glob.glob(os.path.join(folder_path, ext))
        audio_files.extend(files)

    return audio_files

while True:
    path = os.path.join(input("Input Folder: "))
    audio_files = list_audio_files(path)
    for i in audio_files:    
        transcribe_to_srt(i)
    print("\n ---------------Finished!---------------")
