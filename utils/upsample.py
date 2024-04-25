import librosa
import soundfile as sf

# Load the audio file at its original sample rate
file_path = '/home/fundwotsai/DreamSound/training_audio/output_slice_0.wav' # Replace with your audio file path
audio, sr = librosa.load(file_path, sr=16000)

# Resample the audio to 44,100 Hz
audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=44100)

# Save the resampled audio
output_path = '44100_flute_test_0.wav' # Replace with desired output file path
sf.write(output_path, audio_resampled, 44100)
