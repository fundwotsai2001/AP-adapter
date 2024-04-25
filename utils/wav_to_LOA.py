import torch
import torchaudio
import os
from torch.utils.data import Dataset, DataLoader
from audio_encoder.AudioMAE import AudioMAEConditionCTPoolRand, extract_kaldi_fbank_feature

class AudioDataset(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.audios = os.listdir(directory)

    def __len__(self):
        return len(self.audios)

    def __getitem__(self, idx):
        audio = self.audios[idx]
        filename = os.path.join(self.directory, audio)
        waveform, sr = torchaudio.load(filename)
        fbank = torch.zeros((1024, 128))
        ta_kaldi_fbank = extract_kaldi_fbank_feature(waveform, 16000, fbank)
        mel_spect_tensor = torch.tensor(ta_kaldi_fbank).unsqueeze(0)  # [Batch, Channel, Time, Frequency]
        # print("mel_spect_tensor.shape",mel_spect_tensor.shape)
        return mel_spect_tensor.squeeze(0)

def process_and_save(dataset, model, save_dir, batch_size=32):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for i, batch in enumerate(dataloader):
        batch = batch.to("cuda")
        print("batch",batch.shape)
        embed = model(batch, time_pool=8, freq_pool=8)
        print(embed[0].shape)
        for j, tensor in enumerate(embed[0]):
            tensor = tensor.unsqueeze(0)
            torch.save(tensor, os.path.join(save_dir, f'embed_{i * batch_size + j}.pt'))
            # print(tensor.shape)

# Initialize model
model = AudioMAEConditionCTPoolRand().cuda()
model.eval()
# Process training audios
# train_dataset = AudioDataset('/data/home/fundwotsai/DreamSound/training_audio_lute')
# process_and_save(train_dataset, model, 'training_audio_lute_LOA')

# Process class audios
class_dataset = AudioDataset('/mnt/gestalt/home/fundwotsai/DreamSound/mix_chinese_flute_piano')
process_and_save(class_dataset, model, 'class_piano_mixed_flute_audio_LOA')
