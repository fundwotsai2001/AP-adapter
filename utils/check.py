import os
from pydub import AudioSegment
from audioldm.audio.tools import get_mel_from_wav, _pad_spec, normalize_wav, pad_wav
import numpy as np
import torchaudio
# Replace this with your directory
directory = "/home/fundwotsai/DreamSound/audioset/seg_audio"
file = "/home/fundwotsai/DreamSound/audioset/seg_audio/cut_ZEbjwwVKzaQ_Speech, Clapping.wav"
# Function to check if WAV file contains audio
i = 0
def check_wav_file(file_path):
    global i 
    try:
        waveform, sr = torchaudio.load(file_path)  # Faster!!!
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
        # if augment_data:
        #     waveform = augment_audio(
        #         waveform,
        #         sr,
        #         p=0.8,
        #         noise=True,
        #         reverb=True,
        #         low_pass=True,
        #         pitch_shift=True,
        #         delay=True)
        # print(waveform)
        waveform = waveform.numpy()[0, ...]
        waveform = normalize_wav(waveform)
        waveform = waveform[None, ...]
        waveform = pad_wav(waveform, 1024*160)
        # print(waveform)
        max_val = np.max(np.abs(waveform))
        # if max_val > 0:
        waveform = waveform / max_val
        # print(waveform)
        # print(max_val = np.max(np.abs(waveform)))
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        i += 1
# List and check all WAV files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".wav"):
        file_path = os.path.join(directory, filename)
        check_wav_file(file_path)
# check_wav_file(file)
print(i)