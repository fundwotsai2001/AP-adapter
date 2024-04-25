import argparse
import gc
import hashlib
import itertools
import logging
import math
import os
import random
import shutil
import warnings
import configparser
import ast
import contextlib
import glob
import csv
from pathlib import Path
import pandas as pd
import soundfile as sf
import os
import json
import numpy as np
import PIL
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from IPAdapter.ip_adapter.attention_processor import AttnProcessor2_0, CNAttnProcessor2_0,IPAttnProcessor2_0
# TODO: remove and import from diffusers.utils when the new version of diffusers is released
from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchaudio
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import ClapTextModelWithProjection, RobertaTokenizer, RobertaTokenizerFast, SpeechT5HifiGan
from pipeline.modeling_audioldm2 import AudioLDM2UNet2DConditionModel
from diffusers.models.attention_processor import LoRAAttnProcessor2_0

import diffusers

from pipeline.pipeline_audioldm2 import AudioLDM2Pipeline

# from transformers import SpeechT5HifiGan

from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from audioldm.audio import TacotronSTFT, read_wav_file
from audioldm.utils import default_audioldm_config
from scipy.io.wavfile import write
from utils.templates import imagenet_templates_small, imagenet_style_templates_small, text_editability_templates, minimal_templates, imagenet_templates_small_class
# from evaluate import LAIONCLAPEvaluator
from diffusers.loaders import AttnProcsLayers
import matplotlib
matplotlib.use('Agg') # No pictures displayed 
import matplotlib.pyplot as plt
import pylab
import librosa
import librosa.display
import torch.multiprocessing as mp

# Set the start method to 'spawn'
mp.set_start_method(method='forkserver', force=True)

if is_wandb_available():
    import wandb
logger = get_logger(__name__)


def save_model_card(repo_id: str, audios=None, base_model=str, repo_folder=None):
    audio_str = ""
    for i, audio in enumerate(audios):
        write(os.path.join(repo_folder, f"audio_{i}.wav"),16000, audio)
        audio_str += f"![aud_{i}](./audio_{i}.wav)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- textual_inversion
inference: true
---
    """
    model_card = f"""
# Textual inversion text2image fine-tuning - {repo_id}
These are textual inversion adaption weights for {base_model}. You can find some example images in the following. \n
{audio_str}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)

def log_csv(csv_file,row):
    #row is a list of strings, eg
    # row = ['Jane Smith', '28', 'Designer']

    # Check if the CSV file exists
    if os.path.exists(csv_file):
        # File exists, open in append mode
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)
    else:
        # File does not exist, create and add line
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            # header = ['Name', 'Age', 'Occupation']
            # writer.writerow(header)
            writer.writerow(row)

def create_mixture(waveform1, waveform2, snr):

    min_length = min(waveform1.shape[1], waveform2.shape[1])
    waveform1 = waveform1[:, :min_length]
    waveform2 = waveform2[:, :min_length]

    # Calculate the power of each waveform
    power1 = torch.mean(waveform1 ** 2)
    power2 = torch.mean(waveform2 ** 2)

    # Calculate the desired power ratio based on SNR (Signal-to-Noise Ratio)
    desired_snr = 10 ** (-snr / 10)
    scale_factor = torch.sqrt(desired_snr * power1 / power2)

    # Scale the second waveform to achieve the desired SNR
    scaled_waveform2 = waveform2 * scale_factor

    mixture = waveform1 + scaled_waveform2
    return mixture.numpy()

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--config", type=str, default=None, help="Path to .ini file.")
    parser.add_argument("--train_text_encoder", action="store_true", help="Whether to train the text encoder.")
    parser.add_argument("--train_gpt2", action="store_true", help="Whether to train the text encoder.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save learned_embeds.bin every X updates steps.",
    )
    parser.add_argument(
        "--save_as_full_pipeline",
        action="store_true",
        help="Save the complete stable diffusion pipeline.",
    )
    parser.add_argument(
        "--num_vectors",
        type=int,
        default=1,
        help="How many textual inversion vectors shall be used to learn the concept.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--ipadapter",
        default=False,
        required=False,
        help="use ipadapter or not",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--train_data_dir", type=str, default=None, required=False, help="A folder containing the training data."
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_audio_files",
        type=int,
        default=50,
        help=(
            "Minimal class audio files for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional audio files will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--file_list", type=str, default=None, help="Path to a csv file containing which files to train on from the training data directory."
    )
    
    parser.add_argument("--initializer", type=str, default="random_token",choices=["random_token","random_tokens","multitoken_word","saved_embedding","mean"], help="How to initialize the placeholder.")
    parser.add_argument(
        "--initializer_token", type=str, default=None, required=False, help="A token to use as initializer word."
    )
    parser.add_argument("--learnable_property", type=str, default="object", help="Choose between 'object' and 'style'")
    parser.add_argument("--object_class", type=str, default=None, help="Choose a class to learn, works with learnable property 'object_class'")
    parser.add_argument("--instance_word", type=str, default=None, help="Choose a specific word to describe your personal sound")

    parser.add_argument("--repeats", type=int, default=100, help="How many times to repeat the training data.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="audio-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    #todo:change resolution
    # parser.add_argument(
    #     "--resolution",
    #     type=int,
    #     default=512,
    #     help=(
    #         "The resolution for input images, all the images in the train/validation dataset will be resized to this"
    #         " resolution"
    #     ),
    # )
    parser.add_argument("--sample_rate", type=int, default=16000, help="Sample rate for audio.")
    parser.add_argument("--duration", type=float, default=10.0, help="Duration of audio.")
    # parser.add_argument(
    #     "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution."
    # )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=300,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    # parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_audio_files",
        type=int,
        default=0,
        help="Number of audio files that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=None,
        help=(
            "Deprecated in favor of validation_steps. Run validation every X epochs. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument("--validate_experiments", action="store_true", help="Whether to validate experiments.")
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=300,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--offset_noise",
        action="store_true",
        default=False,
        help=(
            "Fine-tuning against a modified noise"
            " See: https://www.crosslabs.org//blog/diffusion-with-offset-noise for more information."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--save_concept_audio", action="store_true",default=False, help="Whether or not to save concept audio.")
    parser.add_argument("--augment_data", action="store_true",default=False, help="Whether or not to augment the training data")
    parser.add_argument("--mix_data", type=str,default=None, help="If a path to an dir containing background audios is specified performs mixture training")
    parser.add_argument(
        "--snr",
        type=int,
        default=20,
        help="In mixture training specify SNR of",
    )
    parser.add_argument(
    "--num_audio_files_to_train",
    type=int,
    default=None,
    help="Number of files to use for training if None will use all files in training dir",
    )




    def read_args_from_config(filename):
        config = configparser.ConfigParser()
        config.read(filename)
        args = dict(config["Arguments"])

        # Convert the values to the appropriate data types
        for key, value in args.items():
            try:
                args[key] = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                pass  # If the value cannot be evaluated, keep it as a string

        return args

    cli_args, _ = parser.parse_known_args()

    if cli_args.config:
        print("Reading arguments from config file")
        config_args = read_args_from_config(cli_args.config)

        # Update the argparse namespace with config_args
        for key, value in config_args.items():
            setattr(cli_args, key, value)

    args = cli_args

    # args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.train_data_dir is None:
        raise ValueError("You must specify a train data directory.")
    
    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    else:
        # logger is not available yet
        if args.class_data_dir is not None:
            warnings.warn("You need not use --class_data_dir without --with_prior_preservation.")
        if args.class_prompt is not None:
            warnings.warn("You need not use --class_prompt without --with_prior_preservation.")

    # if args.train_text_encoder and args.pre_compute_text_embeddings:
    #     raise ValueError("`--train_text_encoder` cannot be used with `--pre_compute_text_embeddings`")
    if args.instance_word and args.object_class:
        args.validation_prompt = f"a recording of a {args.instance_word} {args.object_class} solo"
        args.class_prompt = f"a recording of a piano with {args.instance_word} {args.object_class} "
        print("Overriding validation and class prompts!!!")
    return args
from audio_encoder.AudioMAE import AudioMAEConditionCTPoolRand, extract_kaldi_fbank_feature
from audioldm.audio.tools import get_mel_from_wav, _pad_spec, normalize_wav, pad_wav
from utils.augment_data import augment_audio, augment_spectrogram
def read_wav_file(filename, segment_length, augment_data=False):
    # waveform, sr = librosa.load(filename, sr=None, mono=True) # 4 times slower
    waveform, sr = torchaudio.load(filename)  # Faster!!!
    waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
    if augment_data:
        waveform = augment_audio(
            waveform,
            sr,
            p=0.8,
            noise=True,
            reverb=True,
            low_pass=True,
            pitch_shift=True,
            delay=True)
        
    waveform = waveform.numpy()[0, ...]
    waveform = normalize_wav(waveform)
    waveform = waveform[None, ...]
    waveform = np.copy(waveform)

    waveform = pad_wav(waveform, segment_length)
    max_val = np.max(np.abs(waveform))
    if max_val > 0:
        waveform = waveform / max_val
    else:
        waveform = waveform / 0.000001
    waveform = 0.5 * waveform
    # print(waveform)
    # if not waveform.flags['OWNDATA']:
    #     print("waveform is a view of another array.")
    # else:
    #     print("waveform is not a view; it owns its data.")
    # if waveform.flags['C_CONTIGUOUS']:
    #     print("waveform is stored in a contiguous block of memory (C-style).")
    # else:
    #     print("waveform is not stored in a contiguous block of memory (C-style).")
    return waveform
def investigate_tensor(tensor):
    if not tensor.is_leaf:
        print("The tensor is a view of another tensor.")
    else:
        print("The tensor is not a view; it is a standalone tensor.")

    if tensor.is_contiguous():
        print("The tensor is stored in a contiguous block of memory.")
    else:
        print("The tensor is not stored in a contiguous block of memory.")
def wav_to_fbank(
        filename,
        target_length=1024,
        fn_STFT=None,
        augment_data=False,
        mix_data=False,
        snr=None
    ):
    assert fn_STFT is not None

    # mixup

    if mix_data:
        assert snr is not None, "You specified mixed training but didn't provide SNR!"
        background_file_paths = [os.path.join(mix_data, p) for p in os.listdir(mix_data)]
        background_file_path = random.sample(background_file_paths,1)[0]
        waveform = read_wav_file(filename, target_length * 160, augment_data=augment_data)
        background = read_wav_file(background_file_path, target_length * 160)
        waveform = create_mixture(torch.tensor(waveform), torch.tensor(background), snr)
    else:
        waveform = read_wav_file(filename, target_length * 160, augment_data=augment_data)  # hop size is 160
    waveform = waveform[0, ...]
    waveform = torch.FloatTensor(waveform)

    fbank, log_magnitudes_stft, energy = get_mel_from_wav(waveform, fn_STFT)

    fbank = torch.FloatTensor(fbank.T)
    log_magnitudes_stft = torch.FloatTensor(log_magnitudes_stft.T)

    fbank, log_magnitudes_stft = _pad_spec(fbank, target_length), _pad_spec(
        log_magnitudes_stft, target_length
    )
    fbank = fbank.contiguous()
    log_magnitudes_stft = log_magnitudes_stft.contiguous()
    waveform = waveform.contiguous()

    # investigate_tensor(fbank)
    # investigate_tensor(log_magnitudes_stft)
    # investigate_tensor(waveform)
    

    return fbank, log_magnitudes_stft, waveform



def wav_to_mel(
        original_audio_file_path,
        duration,
        augment_data=False,
        mix_data=False,
        snr=None
):
    config=default_audioldm_config()
    
    fn_STFT = TacotronSTFT(
        config["preprocessing"]["stft"]["filter_length"],
        config["preprocessing"]["stft"]["hop_length"],
        config["preprocessing"]["stft"]["win_length"],
        config["preprocessing"]["mel"]["n_mel_channels"],
        config["preprocessing"]["audio"]["sampling_rate"],
        config["preprocessing"]["mel"]["mel_fmin"],
        config["preprocessing"]["mel"]["mel_fmax"],
    )

    mel, _, _ = wav_to_fbank(
        original_audio_file_path,
        target_length=int(duration * 102.4),
        fn_STFT=fn_STFT,
        augment_data=augment_data,
        mix_data=mix_data,
        snr=snr
    )
    mel = mel.unsqueeze(0)
    # mel = repeat(mel, "1 ... -> b ...", b=batchsize)
    if augment_data:
        mel = mel.unsqueeze(0)
        mel = augment_spectrogram(mel)
        mel = mel.squeeze(0)
    # investigate_tensor(mel)
    return mel
def get_audio_embeds(wav_file=None):
    # print(wav_file)
    waveform, sr = torchaudio.load(wav_file)
    fbank = torch.zeros((1024, 128))
    ta_kaldi_fbank = extract_kaldi_fbank_feature(waveform, sr, fbank)
    # print("ta_kaldi_fbank.shape",ta_kaldi_fbank.shape)
    # mel_spect_tensor = ta_kaldi_fbank.unsqueeze(0)
    # print("mel_spect_tensor.shape",mel_spect_tensor.shape)
    # model = AudioMAEConditionCTPoolRand().cuda()
    # model.eval()
    # LOA_embed = model(mel_spect_tensor, time_pool=8, freq_pool=8)
    # uncond_LOA_embed = model(torch.zeros_like(mel_spect_tensor), time_pool=8, freq_pool=8)
    # print(LOA_embed[0].size(),uncond_LOA_embed[0].size())
    # investigate_tensor(ta_kaldi_fbank)
    return ta_kaldi_fbank
def list_elements_as_string(my_list):
    # Convert all elements to string and join them with a separator
    return ', '.join(map(str, my_list))
# from utils.check import check_wav_file
class AudioInversionDataset(Dataset):
    def __init__(
        self,
        data_root,
        instance_prompt,
        tokenizer,
        device,
        audioldmpipeline,
        class_data_root=None,
        class_prompt=None,
        class_num=None,
        learnable_property="object",  # [object, style, minimal]
        sample_rate=16000,
        duration=2.0,
        repeats=100,
        set="train",
        instance_word=None,
        class_name=None,
        object_class=None,
        augment_data=False,
        mix_data=False,
        snr=None,
        file_list=None,
        num_files_to_train=None
    ):  
        self.data_root = data_root
        self.instance_prompt = instance_prompt
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.sample_rate = sample_rate
        self.duration = duration
        self.instance_word = instance_word
        self.class_name = class_name
        self.audioldmpipeline = audioldmpipeline
        self.augment_data = augment_data
        self.mix_data = mix_data
        self.snr = snr
        self.device = device
        self.data_pairs = []
        self._prepare_dataset()
        
    def get_data_pairs(self):
        return self.data_pairs
    def _prepare_dataset(self):
        for root, dirs, files in os.walk(self.data_root):
            for file in files:
                if file.endswith('.json'):
                    json_path = os.path.join(root, file)
                    with open(json_path, 'r') as f:
                        metadata = json.load(f)
                        audio_path = os.path.join('/home/fundwotsai/DreamSound/Fast-Audioset-Download', metadata['path'])
                        # check_wav_file(audio_path)
                        labels = metadata['labels']
                        # print("audio_path", audio_path)
                        if os.path.exists(audio_path):
                            self.data_pairs.append((labels, audio_path))
    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, i):
        example = {}
        labels, audio_path = self.data_pairs[i]
        example["mel"]=wav_to_mel(
            audio_path,
            self.duration,
            augment_data=self.augment_data,
            mix_data=self.mix_data,
            snr=self.snr
        )
        # print("mel shape",example["mel"].shape)
        # waveform, _ = torchaudio.load(audio_file, normalize=True, num_frames=int(self.duration * self.sample_rate))
        # example["waveform"] = waveform
        rand_num = random.random()
        audioset_templates_small = [
            "a recording of a {}",
            "a {} recording",
            "a synthesized {} audio",
            "a cropped recording of the {}",
            "the recording of a {}",
            "my {} recording",
            "the {} recording",
            "a rendition of the {}",
            "a synthesized {} rendition",
            "the sound of a {}",
            "the sound of {}",
            "the voice of {}",
            "the voice of a {}",
            "a voice of the {}",
            "a synthesized {} voice",

        ]
        labels = list_elements_as_string(labels)
        text = random.choice(audioset_templates_small).format(labels)
        # print("text",text)
        example["ta_kaldi_fbank"] = get_audio_embeds(wav_file=audio_path)
        rand_num = random.random()
        if rand_num < 0.05:
            text = ""
        prompt_embeds, attention_mask, generated_prompt_embeds = self.audioldmpipeline.encode_prompt(
            prompt=text,
            device=self.device,
            negative_prompt = "worst quality, low quality",
            num_waveforms_per_prompt=1,
            do_classifier_free_guidance=False
        )
        example["prompt_embeds"], example["attention_mask"], example["generated_prompt_embeds"] = prompt_embeds, attention_mask, generated_prompt_embeds
        return example
def collate_fn(examples, with_prior_preservation=False):
    mels=[example["mel"] for example in examples]
    ta_kaldi_fbank=[example["ta_kaldi_fbank"] for example in examples]
    prompt_embeds=[example["prompt_embeds"] for example in examples]
    attention_mask = [example["attention_mask"] for example in examples]
    generated_prompt_embeds = [example["generated_prompt_embeds"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        mels += [example["class_mel"] for example in examples]
        prompt_embeds += [example["class_prompt_embeds"] for example in examples]
        attention_mask += [example["class_attention_mask"] for example in examples]
        generated_prompt_embeds += [example["class_generated_prompt_embeds"] for example in examples]
        max_length = max(embed.size(1) for embed in prompt_embeds)
        mask_max_length = max(mask.size(1) for mask in attention_mask)
        num_features = prompt_embeds[0].size(2)
        # Pad each tensor to the max size and collect them in a list
        padded_embeds = []
        padded_masks = []
        for embed in prompt_embeds:
            # Calculate padding size
            pad_size = max_length - embed.size(1)
            # Apply padding
            pad = torch.full((1, pad_size, num_features), 0, dtype=embed.dtype, device=embed.device)
            # print("embed.size()",embed.size())
            # print("pad.size()",pad.size())
            padded_embed = torch.cat([embed, pad], dim=1)
            # print("padded_embed.size()",padded_embed.size())
            padded_embeds.append(padded_embed)
        for mask in attention_mask:
            # Calculate padding size
            pad_size = max_length - mask.size(1)
            # Apply padding
            pad = torch.full((1, pad_size), 0, dtype=mask.dtype, device=mask.device)
            padded_mask = torch.cat([mask, pad], dim=1)
            # print("padded_mask.size()",padded_mask.size())
            padded_masks.append(padded_mask)

        mels = torch.stack(mels)
        mels = mels.to(memory_format=torch.contiguous_format).float()

        prompt_embeds = torch.stack(padded_embeds)
        attention_mask = torch.stack(padded_masks)
        generated_prompt_embeds = torch.stack(generated_prompt_embeds)
    else:
        max_length = max(embed.size(1) for embed in prompt_embeds)
        mask_max_length = max(mask.size(1) for mask in attention_mask)
        num_features = prompt_embeds[0].size(2)
        # Pad each tensor to the max size and collect them in a list
        padded_embeds = []
        padded_masks = []
        for embed in prompt_embeds:
            # Calculate padding size
            pad_size = max_length - embed.size(1)
            # Apply padding
            pad = torch.full((1, pad_size, num_features), 0, dtype=embed.dtype, device=embed.device)
            # print("embed.size()",embed.size())
            # print("pad.size()",pad.size())
            padded_embed = torch.cat([embed, pad], dim=1)
            # print("padded_embed.size()",padded_embed.size())
            padded_embeds.append(padded_embed)
        for mask in attention_mask:
            # Calculate padding size
            pad_size = mask_max_length - mask.size(1)
            # Apply padding
            pad = torch.full((1, pad_size), 0, dtype=mask.dtype, device=mask.device)
            padded_mask = torch.cat([mask, pad], dim=1)
            # print("padded_mask.size()",padded_mask.size())
            padded_masks.append(padded_mask)

        mels = torch.stack(mels)
        mels = mels.to(memory_format=torch.contiguous_format).float()
        ta_kaldi_fbank = torch.stack(ta_kaldi_fbank)
        prompt_embeds = torch.stack(padded_embeds)
        attention_mask = torch.stack(padded_masks)
        generated_prompt_embeds = torch.stack(generated_prompt_embeds)
    # investigate_tensor(prompt_embeds)
    # investigate_tensor(attention_mask)
    # investigate_tensor(generated_prompt_embeds)
    # investigate_tensor(mels)
    # investigate_tensor(ta_kaldi_fbank)
    batch = {
        "mel": mels,
        "ta_kaldi_fbank" : ta_kaldi_fbank,
        "prompt_embeds": prompt_embeds,
        "attention_mask": attention_mask,
        "generated_prompt_embeds": generated_prompt_embeds,
    }
    return batch

def log_validation(outside_data_pairs, audioldmpipeline, text_encoder, tokenizer, unet, vae, args, accelerator, weight_dtype, global_step,vocoder,concept_audio_dir, validate_experiments=False):
    # Select a random file
    labels, random_file= random.choice(outside_data_pairs)
    labels = list_elements_as_string(labels)
    # print("labels",labels)
    audioset_templates_small = [
            "a recording of a {}",
            "a {} recording",
            "a synthesized {} audio",
            "a cropped recording of the {}",
            "the recording of a {}",
            "my {} recording",
            "the {} recording",
            "a rendition of the {}",
            "a synthesized {} rendition",
            "the sound of a {}",
            "the sound of {}",
            "the voice of {}",
            "the voice of a {}",
            "a voice of the {}",
            "a synthesized {} voice",

        ]
    args.validation_prompt = random.choice(audioset_templates_small).format(labels)
    logger.info(
        f"Running validation... \n Generating {args.num_validation_audio_files} audio files with prompt:"
        f" {args.validation_prompt}."
    )
    pipeline=audioldmpipeline
    pipeline.set_progress_bar_config(disable=True)
    
    # import scipy
    # run inference
    generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
    audios = []
    for _ in range(args.num_validation_audio_files):
        print("validation_prompt: {}".format(args.validation_prompt))
        audio_gen = pipeline(audio_file = random_file, prompt = args.validation_prompt,negative_prompt = "worst quality, low quality",num_inference_steps=50,audio_length_in_s=10.0).audios[0]
        audios.append(audio_gen)
        val_audio_dir = os.path.join(args.output_dir, "val_audio_{}".format(global_step))
        os.makedirs(val_audio_dir, exist_ok=True)
        for i, audio in enumerate(audios):
            # scipy.io.wavfile.write(os.path.join(val_audio_dir, f"{'_'.join(args.validation_prompt.split(' '))}_{i}.wav"), rate=16000, data=audio)
            write(os.path.join(val_audio_dir, f"{'_'.join(args.validation_prompt.split(' '))}_{i}.wav"),16000, audio)
            shutil.copy(random_file, os.path.join(val_audio_dir, "original.wav"))
       

class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example
class CustomCollateFn:
        def __init__(self, with_prior_preservation):
            self.with_prior_preservation = with_prior_preservation

        def __call__(self, examples):
            return collate_fn(examples, self.with_prior_preservation)
def main():
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # def list_open_fds():
    #     fds = []
    #     for fd in os.listdir("/proc/self/fd"):
    #         try:
    #             fds.append(os.readlink(f"/proc/self/fd/{fd}"))
    #         except OSError:
    #             # Ignore invalid FDs
    #             continue
    #     return fds

    # # List open file descriptors before initializing DataLoader
    # print("Open file descriptors:", list_open_fds())

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    audioldmpipeline= AudioLDM2Pipeline.from_pretrained(
        args.pretrained_model_name_or_path,
    ).to(accelerator.device)
    tokenizer=audioldmpipeline.tokenizer
    tokenizer_2=audioldmpipeline.tokenizer_2
    text_encoder=audioldmpipeline.text_encoder
    text_encoder_2=audioldmpipeline.text_encoder_2
    GPT2 = audioldmpipeline.language_model
    projection_model=audioldmpipeline.projection_model
    vae=audioldmpipeline.vae
    vocoder=audioldmpipeline.vocoder
    noise_scheduler=audioldmpipeline.scheduler
    unet = audioldmpipeline.unet
    vae.requires_grad_(False)
    if not args.train_gpt2:
        GPT2.requires_grad_(False)
    # Freeze text encoder unless we are training it
    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)
        text_encoder_2.requires_grad_(False)
    vocoder.requires_grad_(False)
    projection_model.requires_grad_(False)
    unet.requires_grad_(False)
    # with open('grad_param_unet.txt', 'w') as file:
    #     for name, param in unet.named_parameters():
    #         if param.requires_grad:
    #             file.writelines(name)
    # class IPAdapter(torch.nn.Module):
    #     def __init__(self, unet, adapter_modules, ckpt_path=None):
    #         super().__init__()
    #         self.unet = unet
    #         self.adapter_modules = adapter_modules

    #         if ckpt_path is not None:
    #             self.load_from_checkpoint(ckpt_path)

    #     def forward(self, noisy_latents, timesteps, generated_prompt_embeds, prompt_embeds, attention_mask):
    #         # Predict the noise residual
    #         # print("generated_prompt_embeds",generated_prompt_embeds.shape)
    #         noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states=generated_prompt_embeds, encoder_hidden_states_1=prompt_embeds, encoder_attention_mask_1=attention_mask, return_dict=False)[0]
    #         return noise_pred

    #     def load_from_checkpoint(self, ckpt_path: str):
    #         # Calculate original checksums
    #         orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

    #         state_dict = torch.load(ckpt_path, map_location="cpu")

    #         # Load state dict for image_proj_model and adapter_modules
    #         self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

    #         # Calculate new checksums
    #         new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

    #         # Verify if the weights have changed
    #         assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

    #         print(f"Successfully loaded weights from checkpoint {ckpt_path}")
    if args.ipadapter:
        print("use ipadapter")
        # Set correct lora layers
        attn_procs = {}
        i = 0
        cross = [None,None,768,768,1024,1024,None,None]
        unet_sd = unet.state_dict()
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor2_0()
                # print(f"{attn_procs[name]}, Type: {type(attn_procs[name])}")
            else:
                cross_attention_dim = cross[i%8]
                i = i + 1
                if cross_attention_dim == 768:
                    layer_name = name.split(".processor")[0]
                    attn_procs[name] = IPAttnProcessor2_0(
                        hidden_size=hidden_size,
                        name = name,
                        cross_attention_dim=cross_attention_dim,
                        scale=1.0,
                        num_tokens=8,
                        do_copy = True
                    ).to(accelerator.device, dtype=torch.float)
                    # attn_procs[name].load_state_dict(weights)
                else:
                    attn_procs[name] = AttnProcessor2_0()
        # # state_dict = torch.load("/home/fundwotsai/DreamSound/audioldm2-large-ipadapter-audioset-unet-random-pooling_v2/checkpoint-113000/pytorch_model.bin", map_location="cuda")
        # # Iterate through each attention processor
        # for name, processor in attn_procs.items():
        #     # Assuming the state_dict's keys match the names of the processors
        # #     if name in state_dict:
        #         # Load the weights
        #         if hasattr(processor, 'to_v_ip') or hasattr(processor, 'to_k_ip'):
        #                 weight_name_v = name + ".to_v_ip.weight"
        #                 weight_name_k = name + ".to_k_ip.weight"
        #                 processor.to_v_ip.weight = torch.nn.Parameter(state_dict[weight_name_v].float())
        #                 processor.to_k_ip.weight = torch.nn.Parameter(state_dict[weight_name_k].float())
        #                 processor.to_k_ip.weight.requires_grad = True
        #                 processor.to_v_ip.weight.requires_grad = True
        unet.set_attn_processor(attn_procs)
        # for idx, module in enumerate(unet.attn_processors.values()):
        #     print(f"Processor {idx}: {module}, Type: {type(module)}")
        # adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())

        # ip_adapter = IPAdapter(unet, adapter_modules)
        # this is a hack to synchronize gradients properly. the module that registers the parameters we care about (in
        # this case, AttnProcsLayers) needs to also be used for the forward pass. AttnProcsLayers doesn't have a
        # `forward` method, so we wrap it to add one and capture the rest of the unet parameters using a closure.
        class _Wrapper(AttnProcsLayers):
            def forward(self, *args, **kwargs):
                return audioldmpipeline.unet(*args, **kwargs)

        unet = _Wrapper(audioldmpipeline.unet.attn_processors)
    else:
        unet = audioldmpipeline.unet
    
    # for name, param in unet.named_parameters():
    #     param.requires_grad = True
    #     print(name,param.requires_grad)
    # ip_ckpt = "/home/fundwotsai/DreamSound/audioldm2-large-ipadapter-audioset-unet-with-valid-resample-v2/checkpoint-9000/pytorch_model.bin"
    from safetensors import safe_open

    # tensors = {}
    # with safe_open("/home/fundwotsai/DreamSound/audioldm2-large-ipadapter-audioset-unet/pipeline_step_81000/unet/diffusion_pytorch_model.safetensors", framework="pt", device="cpu") as f:
    #     for k in f.keys():
    #         tensors[k] = f.get_tensor(k).half()
    #         # print(tensors[k].dtype)


    # state_dict_bin = torch.load(ip_ckpt, map_location="cpu")
    # # pipeline_trained.unet.load_state_dict(checkpoint, strict=True)
    # # This will update the state_dict_bin with tensors from the Safetensor file
    # state_dict_bin.update(tensors)
    # # print(state_dict_bin)
    # audioldmpipeline.unet.load_state_dict(state_dict_bin)
    
    for name, param in audioldmpipeline.text_encoder_2.named_parameters():
        if param.requires_grad:
            print(name)
    for name, param in audioldmpipeline.language_model.named_parameters():
        if param.requires_grad:
            print(name)
    for name, param in vae.named_parameters():
        if param.requires_grad:
            print(name)
    for name, param in vocoder.named_parameters():
        if param.requires_grad:
            print(name)

    # Generate class images if prior preservation is enabled.
    if args.with_prior_preservation:
        class_audio_files_dir = Path(args.class_data_dir)
        if not class_audio_files_dir.exists():
            class_audio_files_dir.mkdir(parents=True)
        cur_class_audio_files = len(list(class_audio_files_dir.iterdir()))

        if cur_class_audio_files < args.num_class_audio_files:
            torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
            if args.prior_generation_precision == "fp32":
                torch_dtype = torch.float32
            elif args.prior_generation_precision == "fp16":
                torch_dtype = torch.float16
            elif args.prior_generation_precision == "bf16":
                torch_dtype = torch.bfloat16
            pipeline = AudioLDM2Pipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                safety_checker=None,
                revision=args.revision,
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_audio_files = args.num_class_audio_files - cur_class_audio_files
            logger.info(f"Number of class images to sample: {num_new_audio_files}.")

            sample_dataset = PromptDataset(args.class_prompt, num_new_audio_files)
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size, num_workers=4)

            sample_dataloader = accelerator.prepare(sample_dataloader)
            pipeline.to(accelerator.device)

            for example in tqdm(
                sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
            ):
                audios = pipeline(example["prompt"],negative_prompt=["Low quality","Low quality","Low quality","Low quality"]).audios
                for i, audio in enumerate(audios):
                    hash_audio = hashlib.sha1(audio.tobytes()).hexdigest()
                    audio_filename=class_audio_files_dir / f"{example['index'][i] + cur_class_audio_files}-{hash_audio}.wav"
                    write(audio_filename, 16000, audio)

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()                                                     
    
    # Save object_class and instance word to output dir.
    import json
    with open(os.path.join(args.output_dir, "class_name.json"), "w") as fd:
        if args.instance_word and args.object_class:
            data = {
                "object_class": args.object_class,
                "instance_word": args.instance_word
            }
        else:
            data = {
                "validation_prompt": args.validation_prompt,
                "class_prompt": args.class_prompt
            }
        json.dump(data, fd)


    print("validation_prompt: ", args.validation_prompt)
          
    # Freeze vae
    
   
    
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()
        if args.train_gpt2:
            GPT2.gradient_checkpointing_enable()
    # Check that all trainable models are in full precision
    low_precision_error_string = (
        "Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training. copy of the weights should still be float32."
    )

    # if accelerator.unwrap_model(unet).dtype != torch.float32:
    #     raise ValueError(
    #         f"Unet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}"
    #     )

    if args.train_text_encoder and accelerator.unwrap_model(text_encoder).dtype != torch.float32:
        raise ValueError(
            f"Text encoder loaded as datatype {accelerator.unwrap_model(text_encoder).dtype}."
            f" {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    if args.train_gpt2 and args.train_text_encoder:
        params_to_optimize = (
            itertools.chain(unet.parameters(), text_encoder.parameters(), GPT2.parameters())
        )
    elif args.train_gpt2:
        params_to_optimize = (
            itertools.chain(unet.parameters(), GPT2.parameters())
        )
    elif args.train_text_encoder:
        params_to_optimize = (
            itertools.chain(unet.parameters(), text_encoder.parameters())
        )
    else:
        params_to_optimize = (
            itertools.chain(unet.parameters())
        )
    
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    # with open('grad_param_ipadapter3.txt', 'w') as file:
    #     for name, param in unet.named_parameters():
    #         if param.requires_grad:
    #             file.writelines(name)
    # save concept audio files to directory
    if args.save_concept_audio:
        
        concept_audio_dir = os.path.join(args.output_dir, "training_audio")
        os.makedirs(concept_audio_dir, exist_ok=True)
        if args.file_list:
            file_list=list(pd.read_csv(args.file_list, header=None)[0])
            if args.num_audio_files_to_train:
                audio_files = sorted([os.path.join(args.train_data_dir, file_path) for file_path in file_list])[:args.num_audio_files_to_train]
            else:
                audio_files = [os.path.join(args.train_data_dir, file_path) for file_path in file_list]
        else:
            if args.num_audio_files_to_train:
                audio_files = sorted([
                    os.path.join(args.train_data_dir, file_path) for file_path in os.listdir(args.train_data_dir) if file_path.endswith(".wav")
                ])[:args.num_audio_files_to_train]
            else:
                audio_files = [
                    os.path.join(args.train_data_dir, file_path) for file_path in os.listdir(args.train_data_dir) if file_path.endswith(".wav")
                ]
        for audio_file in audio_files:
            wave,sr=librosa.load(audio_file, sr=args.sample_rate)
            # wave=wave[:args.duration*sr]
            save_path=os.path.join(concept_audio_dir, os.path.basename(audio_file))
            sf.write(save_path, wave, sr)

            # shutil.copy(audio_file, concept_audio_dir)
        
    else:
        if args.file_list:
            # if file list is provided, we assume that the concept audio files are in the same directory as the file list
            concept_audio_dir = None
        else:
            concept_audio_dir = args.train_data_dir



    # Dataset and DataLoaders creation:
    train_dataset = AudioInversionDataset(
        data_root=args.train_data_dir,
        instance_prompt=args.validation_prompt,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_prompt=args.class_prompt,
        class_num=args.num_class_audio_files,
        tokenizer=tokenizer,
        sample_rate=args.sample_rate,
        duration=args.duration,
        instance_word=args.instance_word,
        class_name=args.object_class,
        repeats=args.repeats,
        learnable_property=args.learnable_property,
        set="train",
        device=accelerator.device,
        audioldmpipeline=audioldmpipeline,
        file_list=args.file_list,
        object_class=args.object_class,
        augment_data=True if args.augment_data else False,
        mix_data=args.mix_data if args.mix_data else False,
        snr=args.snr if args.mix_data else None,
        num_files_to_train=args.num_audio_files_to_train
    )
    outside_data_pairs = train_dataset.get_data_pairs()
    # print("length",len(train_dataset))
    # DataLoader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=CustomCollateFn(args.with_prior_preservation),
        num_workers=4,
    )
    # Assuming your dataset is instantiated as 'dataset'
    # for i in range(len(train_dataset)):
    #     item = train_dataset[i]
    # This should trigger your logging in __getitem__


    if args.validation_epochs is not None:
        warnings.warn(
            f"FutureWarning: You are doing logging with validation_epochs={args.validation_epochs}."
            " Deprecated validation_epochs in favor of `validation_steps`"
            f"Setting `args.validation_steps` to {args.validation_epochs * len(train_dataset)}",
            FutureWarning,
            stacklevel=2,
        )
        args.validation_steps = args.validation_epochs * len(train_dataset)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    if args.train_text_encoder and args.train_gpt2:
        unet, text_encoder, GPT2, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, GPT2, optimizer, train_dataloader, lr_scheduler
        )
    elif args.train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    elif args.train_gpt2:
        unet, GPT2, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, GPT2, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and unet to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder and text_encoder is not None:
        text_encoder.to(accelerator.device, dtype=weight_dtype)
    if not args.train_gpt2 and GPT2 is not None:
        GPT2.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth_audio", config=vars(args))
    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    print("Current process is using device:", torch.cuda.current_device())

    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            # if we're not going to resume from checkpoint, we need to save the initial embeddings
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # keep original embeddings as reference

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        if args.train_text_encoder:
            text_encoder.train()
        if args.train_gpt2:
            GPT2.train()
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            with accelerator.accumulate(unet):
                # print("batch[mel]",batch["mel"].shape)
                # Convert audios to latent space
                latents = vae.encode(batch["mel"].to(dtype=weight_dtype)).latent_dist.sample()
                
                latents = latents * vae.config.scaling_factor
                
                # Sample noise that we'll add to the latents
                if args.offset_noise:
                    noise = torch.randn_like(latents) + 0.1 * torch.randn(
                        latents.shape[0], latents.shape[1], 1, 1, device=latents.device
                    )
                else:
                    noise = torch.randn_like(latents)
              
                bsz, channels, height, width = latents.shape
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                # latent_model_input = torch.cat([latents] * 2)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                mel_spect_tensor = batch["ta_kaldi_fbank"]
                # mel_spect_tensor = ta_kaldi_fbank.unsqueeze(0)
                # print("mel_spect_tensor.shape",mel_spect_tensor.shape)
                model = AudioMAEConditionCTPoolRand().cuda()
                model.eval()
                # print(LOA_embed[0].size(),uncond_LOA_embed[0].size())
                # Get the text embedding for conditioning
                prompt_embeds = batch["prompt_embeds"]
                prompt_embeds = prompt_embeds.squeeze(-3)
                generated_prompt_embeds=batch["generated_prompt_embeds"]
                generated_prompt_embeds=generated_prompt_embeds.squeeze(0)
                # print('generated_prompt_embeds.shape',generated_prompt_embeds.shape)
                attention_mask=batch["attention_mask"]
                attention_mask=attention_mask.squeeze(-2)
                rand_num = random.random()
                pool_list = [1,2,4,8]
                pooling_rate = random.choice(pool_list)
                if rand_num < 0.05:
                    uncond_LOA_embed = model(torch.zeros_like(mel_spect_tensor), time_pool=pooling_rate, freq_pool=pooling_rate)
                    uncond_LOA_embed = uncond_LOA_embed[0].unsqueeze(1)
                    generated_prompt_embeds = torch.cat((generated_prompt_embeds, uncond_LOA_embed), dim=2)
                else:
                    LOA_embed = model(mel_spect_tensor, time_pool=pooling_rate, freq_pool=pooling_rate)
                    LOA_embed = LOA_embed[0].unsqueeze(1)
                    generated_prompt_embeds = torch.cat((generated_prompt_embeds, LOA_embed), dim=2)
                # with open('grad_param_ipadapter1.txt', 'w') as file:
                #     for name, param in unet.named_parameters():
                #         if param.requires_grad:
                #             file.writelines(name)
                # print("noisy_latents.shape",noisy_latents.shape)
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=generated_prompt_embeds,
                    encoder_hidden_states_1=prompt_embeds,
                    encoder_attention_mask_1=attention_mask,
                    return_dict=False,
                )[0]
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if args.with_prior_preservation:
                    # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                  
                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)

                    # Compute instance loss
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                    # Compute prior loss
                    prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")
                   
                    # Add the prior loss to the instance loss.
                    loss = loss + args.prior_loss_weight * prior_loss
                else:
                    # print("model_pred.requires_grad",model_pred.requires_grad)
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    # with open('grad_param_ipadapter0.txt', 'w') as file:
                    #     for name, param in unet.named_parameters():
                    #         if param.requires_grad:
                    #             file.writelines(name)
                    # print("loss.requires_grad",loss.requires_grad)
                
                # import pdb; pdb.set_trace()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if args.train_gpt2 and args.train_text_encoder:
                        params_to_clip = (
                            itertools.chain(unet.parameters(), text_encoder.parameters(), GPT2.parameters())
                        )
                    elif args.train_gpt2:
                        params_to_clip = (
                            itertools.chain(unet.parameters(), GPT2.parameters())
                        )
                    elif args.train_text_encoder:
                        params_to_clip = (
                            itertools.chain(unet.parameters(), text_encoder.parameters())
                        )
                    else:
                        params_to_clip = (
                            itertools.chain(unet.parameters())
                        )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                audios = []
                progress_bar.update(1)
                global_step += 1
               
                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if global_step % args.validation_steps == 0:
                        audios = log_validation(
                            outside_data_pairs,
                            audioldmpipeline,
                            text_encoder, 
                            tokenizer, 
                            unet, vae, args, accelerator, weight_dtype, global_step, vocoder,
                            concept_audio_dir, 
                            validate_experiments=args.validate_experiments,
                        )
                    

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], }
            
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.push_to_hub and not args.save_as_full_pipeline:
            logger.warn("Enabling full model saving because --push_to_hub=True was specified.")
            save_full_model = True
        else:
            save_full_model = args.save_as_full_pipeline
            
        if save_full_model:
            # pipeline.save_pretrained(os.path.join(args.output_dir, "trained_pipeline"))
            audioldmpipeline.save_pretrained(os.path.join(args.output_dir, "trained_pipeline"))
    
        if args.push_to_hub:
            save_model_card(
                repo_id,
                audios=audios,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    main()
