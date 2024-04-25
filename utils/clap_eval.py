import librosa
from glob import glob
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import os
from CLAP.src import laion_clap


folder_name = "step19_sec10_emo_gen"


input_dir = f"../samples/{folder_name}"
input_files = glob(os.path.join(input_dir, '*.wav'))
input_files.sort()

df = pd.DataFrame(columns = ["path", "sim_score"])

# load model 
model = laion_clap.CLAP_Module(enable_fusion=False, amodel= 'HTSAT-base', device='cpu')
model.load_ckpt('CLAP/ckpt/music_audioset_epoch_15_esc_90.14.pt')
model.eval()

for input_file in input_files:
    fname = os.path.basename(input_file).split('.')[0]
    emo = os.path.basename(input_file).split("_")[0]
    emo_dict = {"Q1" : "happy", "Q2" : "angry",  "Q3" : "sad", "Q4" : "tender"}
    # emo_dict = {'Q1': 'happy', 'Q2' : 'mad', 'Q3': 'sorrow', 'Q4': 'exciting'}
    prompt = f"a recording of a {emo_dict[emo]} piano solo, high quality"

    audio, sr = librosa.load(input_file)
    # audio = librosa.resample(y= audio, orig_sr = sr, target_sr = 48000)
    
    # audios = torch.tensor(audio)[None] # [B, T]
    audios = audio[None]

    audio_embed = model.get_audio_embedding_from_data(x = audios, use_tensor=False)
    text_embed = model.get_text_embedding(prompt)
    audio_norm = audio_embed / np.linalg.norm(audio_embed)
    text_norm = text_embed / np.linalg.norm(text_embed)
    similarity = np.dot(audio_norm, text_norm.T)

    df.loc[len(df)] = [fname, similarity[0][0]]

df.to_csv(f"samples/{folder_name}/clap_benchmark.csv", index=False)
print(df["sim_score"].mean())