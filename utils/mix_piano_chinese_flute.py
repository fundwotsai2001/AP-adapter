from pydub import AudioSegment
import os
# Load the first audio file
piano_audio_list = os.listdir("class-piano-audio")
chinese_flute_audio_list = os.listdir("/data/home/fundwotsai/DreamSound/training_audio_lute")

dir = "mix_chinese_lute_piano"
os.makedirs(dir, exist_ok=True)

for i in range(50):
    piano_audio = os.path.join("class-piano-audio", piano_audio_list[i])
    chinese_flute_audio = os.path.join("/data/home/fundwotsai/DreamSound/training_audio_lute", chinese_flute_audio_list[i%3])
    audio1 = AudioSegment.from_file(piano_audio)
    audio2 = AudioSegment.from_file(chinese_flute_audio)
    # Make sure the audio files are the same frame rate
    audio1 = audio1.set_frame_rate(audio2.frame_rate)
    # Mixing the audio files
    mixed = audio1.overlay(audio2)
    # Exporting the mixed audio file
    file = os.path.join(dir, f"mixed_{i}.wav")
    mixed.export(file, format='wav')
