import librosa
import soundfile as sf
import os
def slice_wav(input_file, output_prefix, slice_length):
    """
    Slices a wav file into smaller chunks using librosa.
    
    Parameters:
    - input_file: Path to the input wav file
    - output_prefix: Prefix for the output sliced wav files
    - slice_length: Duration of each slice in seconds
    """
    
    # Load the audio file
    y, sr = librosa.load(input_file, sr=None)
    
    total_length = len(y) / sr
    n_slices = int(total_length // slice_length)
    
    for i in range(n_slices):
        start_sample = i * slice_length * sr
        end_sample = start_sample + slice_length * sr
        
        slice = y[int(start_sample):int(end_sample)]
        
        sf.write(f"{output_prefix}_slice_{i}.wav", slice, sr)

    print(f"Sliced into {n_slices} files with a duration of {slice_length} seconds each.")
os.makedirs("flute_d",exist_ok=True)
# Example usage
slice_wav("piano.wav","piano",10)