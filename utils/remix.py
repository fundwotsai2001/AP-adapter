from pydub import AudioSegment
import random
import os

def load_files(num_files):
    files = []
    for i in range(0, num_files):
        filename = f"output_slice_{i}.wav"
        file_path = os.path.join("/data/home/fundwotsai/DreamSound/2s_training_audio", filename)
        audio = AudioSegment.from_wav(file_path)
        files.append(audio)
    return files

# Load 30 WAV files
wav_files = load_files(15)

# Function to create a remixed file with smoother transitions
def create_smooth_remix(files, duration, output_name):
    remix = AudioSegment.empty()
    crossfade_duration = 500  # Duration of crossfade in milliseconds

    while len(remix) < duration:
        segment = random.choice(files)
        segment = segment.fade_in(100).fade_out(100)  # Apply fade in and fade out
        if len(remix) == 0:
            remix = segment
        else:
            remix = remix.append(segment, crossfade=crossfade_duration)

    remix.export(output_name, format="wav")

# Create 3 remixed files, each 10 seconds long
for i in range(12):
    create_smooth_remix(wav_files, 10000, f"smooth_remixed_file{i}.wav")

print("Smooth remixing completed. Created 3 remixed files.")