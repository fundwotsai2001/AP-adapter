import torchaudio
import torch
from torchaudio_augmentations import Noise, RandomApply, Delay, HighLowPass, PitchShift, Compose
from torchaudio.transforms import  TimeStretch, FrequencyMasking, TimeMasking
from torchlibrosa.augmentation import SpecAugmentation
import random
def augment_audio(
        waveform,
        sr,
        p=0.8,
        noise=False,
        reverb=False,
        low_pass=False,
        pitch_shift=False,
        delay=False
):
    transforms = []
    if noise:
        transforms.append(RandomApply([Noise(min_snr=0.003, max_snr=0.005)], p=p))       
    if reverb:
        transforms.append(RandomApply([Delay(sample_rate=sr)], p=p))
    if low_pass:
        transforms.append(RandomApply([HighLowPass(sample_rate=sr)], p=p)) 
    if pitch_shift:
        transforms.append(RandomApply([PitchShift(n_samples=waveform.shape[1],
        sample_rate=sr
    )], p=p))
    if delay:
        transforms.append(RandomApply([Delay(sample_rate=sr)], p=p))

    # Dont perform any augmentation
    if transforms == []:
        return waveform

    transforms = random.sample(transforms, 1)

    transform = Compose(transforms=transforms)
    augmented_waveform =  transform(waveform)
    return augmented_waveform


def augment_spectrogram(
    spec,
):
    spec_augmenter = SpecAugmentation(
        time_drop_width=64,
        time_stripes_num=2,
        freq_drop_width=8,
        freq_stripes_num=2)


    return spec_augmenter(spec)