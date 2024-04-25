"""
Reference Repo: https://github.com/facebookresearch/AudioMAE
"""

import torch
import torch.nn as nn
from timm.models.layers import to_2tuple
from . import models_vit
from . import models_mae
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torchaudio

# model = mae_vit_base_patch16(in_chans=1, audio_exp=True, img_size=(1024, 128))
class Vanilla_AudioMAE(nn.Module):
    """Audio Masked Autoencoder (MAE) pre-trained on AudioSet (for AudioLDM2)"""

    def __init__(
        self,
    ):
        super().__init__()
        model = models_mae.__dict__["mae_vit_base_patch16"](
            in_chans=1, audio_exp=True, img_size=(1024, 128)
        )

        checkpoint_path = '/home/fundwotsai/DreamSound/pretrained.pth'
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        msg = model.load_state_dict(checkpoint['model'], strict=False)

        # Skip the missing keys of decoder modules (not required)
        # print(f'Load AudioMAE from {checkpoint_path} / message: {msg}')

        self.model = model.eval()

    def forward(self, x, mask_ratio=0.0, no_mask=False, no_average=False):
        """
        x: mel fbank [Batch, 1, 1024 (T), 128 (F)]
        mask_ratio: 'masking ratio (percentage of removed patches).'
        """
        with torch.no_grad():
            # embed: [B, 513, 768] for mask_ratio=0.0
            if no_mask:
                if no_average:
                    # raise RuntimeError("This function is deprecated")
                    embed = self.model.forward_encoder_no_random_mask_no_average(
                        x
                    )  # mask_ratio
                else:
                    embed = self.model.forward_encoder_no_mask(x)  # mask_ratio
            else:
                raise RuntimeError("This function is deprecated")
                embed, _, _, _ = self.model.forward_encoder(x, mask_ratio=mask_ratio)
        return embed
import torchaudio
import numpy as np
import torch

# def roll_mag_aug(waveform):
#     idx = np.random.randint(len(waveform))
#     rolled_waveform = np.roll(waveform, idx)
#     mag = np.random.beta(10, 10) + 0.5
#     return torch.Tensor(rolled_waveform * mag)

def wav_to_fbank(filename, melbins, target_length, roll_mag_aug_flag=False):
    waveform, sr = torchaudio.load(filename)
    waveform = waveform - waveform.mean()
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, 
        htk_compat=True, 
        sample_frequency=sr, 
        use_energy=False,
        window_type='hanning', 
        num_mel_bins=melbins, 
        dither=0.0, 
        frame_shift=10
    )

    n_frames = fbank.shape[0]
    p = target_length - n_frames

    # Cut and pad
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]

    return fbank

# Example usage
import torch.nn.functional as F
class AudioMAEConditionCTPoolRand(nn.Module):
    """
    audiomae = AudioMAEConditionCTPool2x2()
    data = torch.randn((4, 1024, 128))
    output = audiomae(data)
    import ipdb;ipdb.set_trace()
    exit(0)
    """

    def __init__(
        self,
        time_pooling_factors=[1, 2, 4, 8],
        freq_pooling_factors=[1, 2, 4, 8],
        eval_time_pooling=8,
        eval_freq_pooling=8,
        mask_ratio=0.0,
        regularization=False,
        no_audiomae_mask=True,
        no_audiomae_average=True,
    ):
        super().__init__()
        self.device = None
        self.time_pooling_factors = time_pooling_factors
        self.freq_pooling_factors = freq_pooling_factors
        self.no_audiomae_mask = no_audiomae_mask
        self.no_audiomae_average = no_audiomae_average

        self.eval_freq_pooling = eval_freq_pooling
        self.eval_time_pooling = eval_time_pooling
        self.mask_ratio = mask_ratio
        self.use_reg = regularization

        self.audiomae = Vanilla_AudioMAE()
        self.audiomae.eval()
        for p in self.audiomae.parameters():
            p.requires_grad = False

    # Required
    def get_unconditional_condition(self, batchsize):
        param = next(self.audiomae.parameters())
        assert param.requires_grad == False
        device = param.device
        # time_pool, freq_pool = max(self.time_pooling_factors), max(self.freq_pooling_factors)
        time_pool, freq_pool = min(self.eval_time_pooling, 64), min(
            self.eval_freq_pooling, 8
        )
        # time_pool = self.time_pooling_factors[np.random.choice(list(range(len(self.time_pooling_factors))))]
        # freq_pool = self.freq_pooling_factors[np.random.choice(list(range(len(self.freq_pooling_factors))))]
        token_num = int(512 / (time_pool * freq_pool))
        return [
            torch.zeros((batchsize, token_num, 768)).to(device).float(),
            torch.ones((batchsize, token_num)).to(device).float(),
        ]

    def pool(self, representation, time_pool=None, freq_pool=None):
        assert representation.size(-1) == 768
        representation = representation[:, 1:, :].transpose(1, 2)
        # print("representation.shape",representation.shape)
        bs, embedding_dim, token_num = representation.size()
        representation = representation.reshape(bs, embedding_dim, 64, 8)

        # if self.training:
        #     if time_pool is None and freq_pool is None:
        #         time_pool = min(
        #             64,
        #             self.time_pooling_factors[
        #                 np.random.choice(list(range(len(self.time_pooling_factors))))
        #             ],
        #         )
        #         # freq_pool = self.freq_pooling_factors[np.random.choice(list(range(len(self.freq_pooling_factors))))]
        #         freq_pool = min(8, time_pool)  # TODO here I make some modification.
        # else:
        #     time_pool, freq_pool = min(self.eval_time_pooling, 64), min(
        #         self.eval_freq_pooling, 8
        #     )

        self.avgpooling = nn.AvgPool2d(
            kernel_size=(time_pool, freq_pool), stride=(time_pool, freq_pool)
        )
        self.maxpooling = nn.MaxPool2d(
            kernel_size=(time_pool, freq_pool), stride=(time_pool, freq_pool)
        )

        pooled = (
            self.avgpooling(representation) + self.maxpooling(representation)
        ) / 2  # [bs, embedding_dim, time_token_num, freq_token_num]
        # print("pooled.shape",pooled.shape)
        pooled = pooled.flatten(2).transpose(1, 2)
        return pooled  # [bs, token_num, embedding_dim]

    def regularization(self, x):
        assert x.size(-1) == 768
        x = F.normalize(x, p=2, dim=-1)
        return x

    # Required
    def forward(self, batch, time_pool=None, freq_pool=None):
        assert batch.size(-2) == 1024 and batch.size(-1) == 128
        
        if self.device is None:
            self.device = next(self.audiomae.parameters()).device

        batch = batch.unsqueeze(1).to(self.device)
        with torch.no_grad():
            representation = self.audiomae(
                batch,
                mask_ratio=self.mask_ratio,
                no_mask=self.no_audiomae_mask,
                no_average=self.no_audiomae_average,
            )
            representation = self.pool(representation, time_pool, freq_pool)
            if self.use_reg:
                representation = self.regularization(representation)
            return [
                representation,
                torch.ones((representation.size(0), representation.size(1)))
                .to(representation.device)
                .float(),
            ]


class AudioMAEConditionCTPoolRandTFSeparated(nn.Module):
    """
    audiomae = AudioMAEConditionCTPool2x2()
    data = torch.randn((4, 1024, 128))
    output = audiomae(data)
    import ipdb;ipdb.set_trace()
    exit(0)
    """

    def __init__(
        self,
        time_pooling_factors=[8],
        freq_pooling_factors=[8],
        eval_time_pooling=8,
        eval_freq_pooling=8,
        mask_ratio=0.0,
        regularization=False,
        no_audiomae_mask=True,
        no_audiomae_average=False,
    ):
        super().__init__()
        self.device = None
        self.time_pooling_factors = time_pooling_factors
        self.freq_pooling_factors = freq_pooling_factors
        self.no_audiomae_mask = no_audiomae_mask
        self.no_audiomae_average = no_audiomae_average

        self.eval_freq_pooling = eval_freq_pooling
        self.eval_time_pooling = eval_time_pooling
        self.mask_ratio = mask_ratio
        self.use_reg = regularization

        self.audiomae = Vanilla_AudioMAE()
        self.audiomae.eval()
        for p in self.audiomae.parameters():
            p.requires_grad = False

    # Required
    def get_unconditional_condition(self, batchsize):
        param = next(self.audiomae.parameters())
        assert param.requires_grad == False
        device = param.device
        # time_pool, freq_pool = max(self.time_pooling_factors), max(self.freq_pooling_factors)
        time_pool, freq_pool = min(self.eval_time_pooling, 64), min(
            self.eval_freq_pooling, 8
        )
        # time_pool = self.time_pooling_factors[np.random.choice(list(range(len(self.time_pooling_factors))))]
        # freq_pool = self.freq_pooling_factors[np.random.choice(list(range(len(self.freq_pooling_factors))))]
        token_num = int(512 / (time_pool * freq_pool))
        return [
            torch.zeros((batchsize, token_num, 768)).to(device).float(),
            torch.ones((batchsize, token_num)).to(device).float(),
        ]

    def pool(self, representation, time_pool=None, freq_pool=None):
        assert representation.size(-1) == 768
        representation = representation[:, 1:, :].transpose(1, 2)
        bs, embedding_dim, token_num = representation.size()
        representation = representation.reshape(bs, embedding_dim, 64, 8)

        # if self.training:
        #     if time_pool is None and freq_pool is None:
        #         time_pool = min(
        #             64,
        #             self.time_pooling_factors[
        #                 np.random.choice(list(range(len(self.time_pooling_factors))))
        #             ],
        #         )
        #         freq_pool = min(
        #             8,
        #             self.freq_pooling_factors[
        #                 np.random.choice(list(range(len(self.freq_pooling_factors))))
        #             ],
        #         )
        #         # freq_pool = min(8, time_pool) # TODO here I make some modification.
        # else:
        #     time_pool, freq_pool = min(self.eval_time_pooling, 64), min(
        #         self.eval_freq_pooling, 8
        #     )

        self.avgpooling = nn.AvgPool2d(
            kernel_size=(time_pool, freq_pool), stride=(time_pool, freq_pool)
        )
        self.maxpooling = nn.MaxPool2d(
            kernel_size=(time_pool, freq_pool), stride=(time_pool, freq_pool)
        )

        pooled = (
            self.avgpooling(representation) + self.maxpooling(representation)
        ) / 2  # [bs, embedding_dim, time_token_num, freq_token_num]
        pooled = pooled.flatten(2).transpose(1, 2)
        return pooled  # [bs, token_num, embedding_dim]

    def regularization(self, x):
        assert x.size(-1) == 768
        x = F.normalize(x, p=2, dim=-1)
        return x

    # Required
    def forward(self, batch, time_pool=None, freq_pool=None):
        assert batch.size(-2) == 1024 and batch.size(-1) == 128

        if self.device is None:
            self.device = batch.device

        batch = batch.unsqueeze(1)
        with torch.no_grad():
            representation = self.audiomae(
                batch,
                mask_ratio=self.mask_ratio,
                no_mask=self.no_audiomae_mask,
                no_average=self.no_audiomae_average,
            )
            representation = self.pool(representation, time_pool, freq_pool)
            if self.use_reg:
                representation = self.regularization(representation)
            return [
                representation,
                torch.ones((representation.size(0), representation.size(1)))
                .to(representation.device)
                .float(),
            ]
def apply_time_mask(spectrogram, mask_width_range=(1000, 1001), max_masks=2):
    """
    Apply time masking to a spectrogram (PyTorch tensor).

    :param spectrogram: A PyTorch tensor of shape (time_steps, frequency_bands)
    :param mask_width_range: A tuple indicating the min and max width of the mask
    :param max_masks: Maximum number of masks to apply
    :return: Masked spectrogram
    """
    time_steps, frequency_bands = spectrogram.shape
    masked_spectrogram = spectrogram.clone()

    for _ in range(max_masks):
        mask_width = torch.randint(mask_width_range[0], mask_width_range[1], (1,)).item()
        start_step = torch.randint(0, time_steps - mask_width, (1,)).item()
        masked_spectrogram[100:1024, :] = 0  # or another constant value

    return masked_spectrogram

def extract_kaldi_fbank_feature(waveform, sampling_rate,log_mel_spec):
    norm_mean = -4.2677393
    norm_std = 4.5689974

    if sampling_rate != 16000:
        waveform_16k = torchaudio.functional.resample(
            waveform, orig_freq=sampling_rate, new_freq=16000
        )
    else:
        waveform_16k = waveform

    waveform_16k = waveform_16k - waveform_16k.mean()
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform_16k,
        htk_compat=True,
        sample_frequency=16000,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=128,
        dither=0.0,
        frame_shift=10,
    )

    TARGET_LEN = log_mel_spec.size(0)

    # cut and pad
    n_frames = fbank.shape[0]
    p = TARGET_LEN - n_frames
    # print(TARGET_LEN)
    # print(n_frames)
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[:TARGET_LEN, :]

    fbank = (fbank - norm_mean) / (norm_std * 2)
    # fbank = apply_time_mask(fbank)
    return fbank # [1024, 128]
if __name__ == "__main__":

    filename = '/home/fundwotsai/DreamSound/training_audio_v2/output_slice_18.wav'
    waveform, sr = torchaudio.load(filename)
    fbank = torch.zeros(
            (1024, 128)
        )
    ta_kaldi_fbank = extract_kaldi_fbank_feature(waveform, 16000,fbank)
    print(ta_kaldi_fbank.shape)
    # melbins = 128  # Number of Mel bins
    # target_length = 1024  # Number of frames
    # fbank = wav_to_fbank(file_path, melbins, target_length, roll_mag_aug_flag=False)
    # print(fbank.shape)
    # # Convert to PyTorch tensor and reshape
    mel_spect_tensor = torch.tensor(ta_kaldi_fbank).unsqueeze(0)  # [Batch, Channel, Time, Frequency]
    
    mel_spect_tensor = mel_spect_tensor.to("cuda")
    # Save the figure
    print("mel_spect_tensor111.shape",mel_spect_tensor.shape)
    model = AudioMAEConditionCTPoolRand().cuda()
    print("The first run")
    embed = model(mel_spect_tensor, time_pool=1, freq_pool=1)
    print(embed[0].shape)

    # Reshape tensor for 2D pooling: treat each 768 as a channel
    # Example usage
    # Assuming the pooling operation reduces the second dimension from 513 to 8
    
    
    torch.save(embed[0], "MAE_feature1_stride-no-pool.pt")
    with open('output_tensor.txt', 'w') as f:
        print(embed[0], file=f)
    