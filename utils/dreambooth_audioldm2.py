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

# TODO: remove and import from diffusers.utils when the new version of diffusers is released
from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchaudio
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import ClapTextModelWithProjection, RobertaTokenizer, RobertaTokenizerFast, SpeechT5HifiGan

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DDIMScheduler
)
# from pipeline.pipeline_audioldm import AudioLDMPipeline
from pipeline.pipeline_audioldm2 import AudioLDM2Pipeline
from pipeline.modeling_audioldm2 import AudioLDM2UNet2DConditionModel
# from transformers import SpeechT5HifiGan
from transformers import (
    ClapFeatureExtractor,
    ClapModel,
    GPT2Model,
    RobertaTokenizer,
    RobertaTokenizerFast,
    SpeechT5HifiGan,
    T5EncoderModel,
    T5Tokenizer,
    T5TokenizerFast,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from audioldm.audio import TacotronSTFT, read_wav_file
from audioldm.utils import default_audioldm_config
from scipy.io.wavfile import write
from utils.templates import imagenet_templates_small, imagenet_style_templates_small, text_editability_templates, minimal_templates, imagenet_templates_small_class
# from evaluate import LAIONCLAPEvaluator
from pipeline.modeling_audioldm2 import AudioLDM2ProjectionModel, AudioLDM2UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor2_0
import matplotlib
matplotlib.use('Agg') # No pictures displayed 
import matplotlib.pyplot as plt
import pylab
import librosa
import librosa.display

if is_wandb_available():
    import wandb

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }
# ------------------------------------------------------------------------------


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")

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

def log_validation(audioldmpipeline, text_encoder, tokenizer, unet, vae, args, accelerator, weight_dtype, global_step,vocoder,concept_audio_dir, validate_experiments=False):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_audio_files} audio files with prompt:"
        f" {args.validation_prompt}."
    )
    # create pipeline (note: unet and vae are loaded again in float32)
    # pipeline = AudioLDM2Pipeline.from_pretrained(
    #     args.pretrained_model_name_or_path,
    #     text_encoder=accelerator.unwrap_model(text_encoder),
    #     tokenizer=tokenizer,
    #     unet=accelerator.unwrap_model(unet),
    #     vae=vae,
    #     safety_checker=None,
    #     revision=args.revision,
    #     torch_dtype=weight_dtype,
    # ).to(accelerator.device)
    pipeline=audioldmpipeline
    pipeline.save_pretrained(args.output_dir+"/pipeline_step_{}".format(global_step))

    # pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    # pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    
    # import scipy
    # run inference
    generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
    audios = []
    class_audios = []
    for _ in range(args.num_validation_audio_files):
        # with torch.autocast("cuda"):
            
            # pipe(prompt, num_inference_steps=50, audio_length_in_s=10.0).audios[0]
        print("validation_prompt: {}".format(args.validation_prompt))
        audio_gen = pipeline(args.validation_prompt, num_inference_steps=50,audio_length_in_s=10.0).audios[0]
        class_gen = pipeline(args.class_prompt, num_inference_steps=50,audio_length_in_s=10.0).audios[0]
        audios.append(audio_gen)
        class_audios.append(class_gen)
        val_audio_dir = os.path.join(args.output_dir, "val_audio_{}".format(global_step))
        os.makedirs(val_audio_dir, exist_ok=True)
        for i, audio in enumerate(audios):
            # scipy.io.wavfile.write(os.path.join(val_audio_dir, f"{'_'.join(args.validation_prompt.split(' '))}_{i}.wav"), rate=16000, data=audio)
            write(os.path.join(val_audio_dir, f"{'_'.join(args.validation_prompt.split(' '))}_{i}.wav"),16000, audio)
        for i, audio in enumerate(class_audios):
            # scipy.io.wavfile.write(os.path.join(val_audio_dir, f"{'_'.join(args.validation_prompt.split(' '))}_{i}.wav"), rate=16000, data=audio)
            write(os.path.join(val_audio_dir, f"{'_'.join(args.class_prompt.split(' '))}_{i}.wav"),16000, audio)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            # np_images = np.stack([np.asarray(img) for img in images])
            for i, audio in enumerate(audios):

                tracker.writer.add_audio(f"validation_{i}", audio, global_step, sample_rate=16000)
        if tracker.name == "wandb":
            # tracker.log(
            #     {
            #         "validation": [
            #             wandb.Image(image, caption=f"{i}: {args.validation_prompt}") for i, image in enumerate(images)
            #         ]
            #     }
            # )
            raise NotImplementedError("Wandb not implemented yet for audio")
    if validate_experiments:
        # TODO check if audio length should be set, for now setting to training length
        audios_rec = pipeline(args.validation_prompt, num_inference_steps=50,audio_length_in_s=10.24, generator=generator,num_waveforms_per_prompt=10).audios
        
        # print(audios_rec.shape)
        # audios_rec = np.concatenate((audios_rec, pipeline(args.validation_prompt, num_inference_steps=50, generator=generator,num_waveforms_per_prompt=20).audios))
        # print("audios_rec shape: {}".format(audios_rec.shape))
        val_audio_dir = os.path.join(args.output_dir, "reconstruction_audio_{}".format(global_step))
        os.makedirs(val_audio_dir, exist_ok=True)
        for i, audio in enumerate(audios_rec):
            write(os.path.join(val_audio_dir, f"{'_'.join(args.validation_prompt.split(' '))}_{i}.wav"),16000, audio)
        
        print("loading clap evaluator")

        with contextlib.redirect_stdout(None):
            
            evaluator = LAIONCLAPEvaluator(device=accelerator.device)
        reconstruction_score=evaluator.audio_to_audio_similarity(concept_audio_dir, val_audio_dir)
        print("Reconstruction score: {}".format(reconstruction_score))
        print(type(reconstruction_score))
        accelerator.log({"reconstruction_score": reconstruction_score.item()},step=global_step)
        reconstruction_csv_path=os.path.join(args.output_dir, "reconstruction_score.csv")
        log_csv(reconstruction_csv_path,[global_step,reconstruction_score.item()])
        # accelerator.log({"reconstruction_scored": np.float16(reconstruction_score)},step=global_step)
        del audios_rec
        del evaluator
        
    # del pipeline
    torch.cuda.empty_cache()
    return audios


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
        "--lora",
        default=False,
        required=False,
        help="use lora or not",
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
        default=0,
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
        default="tensorboard",
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
    parser.add_argument("--save_concept_audio", action="store_true",default=True, help="Whether or not to save concept audio.")
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
    waveform = pad_wav(waveform, segment_length)
    
    waveform = waveform / np.max(np.abs(waveform))
    waveform = 0.5 * waveform
    
    return waveform

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
    return mel

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
        self.num_files_to_train = num_files_to_train
        if file_list is not None:
            file_list=list(pd.read_csv(file_list, header=None)[0])
            self.audio_files = [os.path.join(self.data_root, file_path) for file_path in file_list]
        else:
            self.audio_files = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root) if file_path.endswith(".wav")]

        if self.num_files_to_train:
            self.audio_files = sorted(self.audio_files)[:self.num_files_to_train]
        # if self.mix_data or self.augment_data:
        #     self.audio_files = sorted(self.audio_files)[:1]
        self.num_files = len(self.audio_files)
        self._length = self.num_files
        self.device = device

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_audio_files_paths = [os.path.join(self.class_data_root, file_path) for file_path in os.listdir(self.class_data_root) if file_path.endswith(".wav")]
            if class_num is not None:
                self.num_class_audio_files = min(len(self.class_audio_files_paths), class_num)
            else:
                self.num_class_audio_files = len(self.class_audio_files_paths)
            self._length = max(self.num_class_audio_files, self.num_files)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        if self.learnable_property == "object":
            self.templates = imagenet_templates_small
        elif self.learnable_property == "style":
            self.templates = imagenet_style_templates_small
        elif self.learnable_property == "minimal":
            self.templates = minimal_templates
        elif self.learnable_property == "object_class":
            self.templates = imagenet_templates_small_class
            self.object_class = object_class
        if set == "train":
            self._length = self.num_files * repeats

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        # print("i",i)
        audio_file = self.audio_files[i % self.num_files]
        example["mel"]=wav_to_mel(
            audio_file,
            self.duration,
            augment_data=self.augment_data,
            mix_data=self.mix_data,
            snr=self.snr
        )
        # waveform, _ = torchaudio.load(audio_file, normalize=True, num_frames=int(self.duration * self.sample_rate))
        # example["waveform"] = waveform
        if self.instance_word and self.class_name:
        
            text = "a recording of a {}".format(self.instance_word+" "+self.class_name )
            
        else:
            text= self.instance_prompt
        
        # if self.learnable_property == "object_class":
        #     text = random.choice(self.templates).format(placeholder_string, self.object_class)
        # else:
        #     text = random.choice(self.templates).format(placeholder_string)
        # example["input_ids"] = self.tokenizer(
        #     text,
        #     padding="max_length",
        #     truncation=True,
        #     max_length=self.tokenizer.model_max_length,
        #     return_tensors="pt",
        # ).input_ids[0]
        
        example["prompt_embeds"], example["attention_mask"], example["generated_prompt_embeds"] = self.audioldmpipeline.encode_prompt(

            prompt=text,
            device=self.device,
            num_waveforms_per_prompt=1,
            do_classifier_free_guidance=False
        )
        if self.class_data_root:
            class_audio=self.class_audio_files_paths[i % self.num_class_audio_files]
            # print(class_audio)
            example["class_mel"]=wav_to_mel(
                class_audio,
                self.duration)
            example["class_prompt_embeds"], example["class_attention_mask"], example["class_generated_prompt_embeds"] = self.audioldmpipeline.encode_prompt(

                prompt=self.class_prompt,
                device=self.device,
                num_waveforms_per_prompt=1,
                do_classifier_free_guidance=False
            )
            # class_text_inputs = tokenize_prompt(
            #         self.tokenizer, self.class_prompt, tokenizer_max_length=self.tokenizer_max_length
            #     )
            #     example["class_prompt_ids"] = class_text_inputs.input_ids
            #     example["class_attention_mask"] = class_text_inputs.attention_mask
            
            
        return example
def collate_fn(examples, with_prior_preservation=False):
    mels=[example["mel"] for example in examples]
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
        mels = torch.stack(mels)
        mels = mels.to(memory_format=torch.contiguous_format).float()

        prompt_embeds = torch.stack(prompt_embeds)
        attention_mask = torch.stack(attention_mask)
        generated_prompt_embeds = torch.stack(generated_prompt_embeds)
    batch = {
        "mel": mels,
        "prompt_embeds": prompt_embeds,
        "attention_mask": attention_mask,
        "generated_prompt_embeds": generated_prompt_embeds,
    }

    # if has_attention_mask:
    #     attention_mask = torch.cat(attention_mask, dim=0)
    #     batch["attention_mask"] = attention_mask

    return batch


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

    # Load the tokenizer
    # if args.tokenizer_name:
    #     tokenizer = RobertaTokenizerFast.from_pretrained(args.tokenizer_name)
    # elif args.pretrained_model_name_or_path:
    #     tokenizer = RobertaTokenizerFast.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    
    # Load scheduler and models
    # noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    # text_encoder = ClapModel.from_pretrained(
    #     args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    # )
    #  vae: AutoencoderKL,
    #     text_encoder: ClapModel,
    #     text_encoder_2: T5EncoderModel,
    #     projection_model: AudioLDM2ProjectionModel,
    #     language_model: GPT2Model,
    #     tokenizer: Union[RobertaTokenizer, RobertaTokenizerFast],
    #     tokenizer_2: Union[T5Tokenizer, T5TokenizerFast],
    #     feature_extractor: ClapFeatureExtractor,
    #     unet: AudioLDM2UNet2DConditionModel,
    #     scheduler: KarrasDiffusionSchedulers,
    #     vocoder: SpeechT5HifiGan,
    # projection_model = AudioLDM2ProjectionModel.from_pretrained(
    #     args.pretrained_model_name_or_path, subfolder="projection_model", revision=args.revision
    # )
    # vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    # unet = AudioLDM2UNet2DConditionModel.from_pretrained(
    #     args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    # )
    # vocoder = SpeechT5HifiGan.from_pretrained(args.pretrained_model_name_or_path, subfolder="vocoder", revision=args.revision
    # )

    # audioldmpipeline=AudioLDM2Pipeline(
    #     text_encoder=text_encoder,
    #     vae=vae,
    #     unet=unet,
    #     vocoder=vocoder,
    #     scheduler=noise_scheduler,
    #     tokenizer=tokenizer
    # ).to(accelerator.device)
    audioldmpipeline= AudioLDM2Pipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        # text_encoder=accelerator.unwrap_model(text_encoder),
        # tokenizer=tokenizer,
        # unet=accelerator.unwrap_model(unet),
        # vae=vae,
        safety_checker=None,
        revision=args.revision,
        torch_dtype=torch.float32, #TODO: change it to weight_dtype
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
    unet = AudioLDM2UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )
    # Freeze vae
    vae.requires_grad_(False)
    if not args.train_gpt2:
        GPT2.requires_grad_(False)
    # Freeze text encoder unless we are training it
    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)
        text_encoder_2.requires_grad_(False)
    vocoder.requires_grad_(False)
    projection_model.requires_grad_(False)
    unet.requires_grad_(not args.lora)
    if args.lora:
        print("use lora")
        # Set correct lora layers
        lora_attn_procs = {}
        i = 0
        cross = [None,None,768,768,1024,1024,None,None]
        for name in audioldmpipeline.unet.attn_processors.keys():
            # print(name)
            if name.startswith("mid_block"):
                hidden_size = audioldmpipeline.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(audioldmpipeline.unet.config.block_out_channels))[
                    block_id
                ]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = audioldmpipeline.unet.config.block_out_channels[block_id]
            # print("hidden_size",hidden_size)
            if name.endswith("attn1.processor"):
                cross_attention_dim = None
            else:
                # if "down_blocks" in name or "up_blocks" in name:
                cross_attention_dim = cross[i%8]
                i = i+1
            
            # print(hidden_size)
            # print("cross_attention_dim",cross_attention_dim)
            lora_attn_procs[name] = LoRAAttnProcessor2_0(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim
            )
        audioldmpipeline.unet.set_attn_processor(lora_attn_procs)

        # this is a hack to synchronize gradients properly. the module that registers the parameters we care about (in
        # this case, AttnProcsLayers) needs to also be used for the forward pass. AttnProcsLayers doesn't have a
        # `forward` method, so we wrap it to add one and capture the rest of the unet parameters using a closure.
        class _Wrapper(AttnProcsLayers):
            def forward(self, *args, **kwargs):
                return audioldmpipeline.unet(*args, **kwargs)

        unet = _Wrapper(audioldmpipeline.unet.attn_processors)
    else:
        unet = audioldmpipeline.unet


    


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
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)

            sample_dataloader = accelerator.prepare(sample_dataloader)
            pipeline.to(accelerator.device)

            for example in tqdm(
                sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
            ):
                # images = pipeline(example["prompt"]).images

                # for i, image in enumerate(images):
                #     hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                #     image_filename = class_audio_files_dir / f"{example['index'][i] + cur_class_audio_files}-{hash_image}.jpg"
                #     image.save(image_filename)
                audios = pipeline(example["prompt"],negative_prompt=["Low quality","Low quality","Low quality","Low quality"]).audios
                for i, audio in enumerate(audios):
                    hash_audio = hashlib.sha1(audio.tobytes()).hexdigest()
                    audio_filename=class_audio_files_dir / f"{example['index'][i] + cur_class_audio_files}-{hash_audio}.wav"
                    write(audio_filename, 16000, audio)

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()                                                     

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        # print("Number of models:", len(models))
        # for model in models:
        #     print(type(model))
        # assert len(models) == 1
        if args.lora and isinstance(models[0], AttnProcsLayers):
            audioldmpipeline.unet.save_attn_procs(output_dir)
        # elif not args.lora and isinstance(models[0], UNet2DConditionModel):
        #     models[0].save_pretrained(os.path.join(output_dir, "unet"))
        # else:
        #     raise ValueError(f"Unknown model type {type(models[0])}")
        weights.pop()  # ensures that accelerate doesn't try to handle saving of the model

    def load_model_hook(models, input_dir):
        # assert len(models) == 1
        if args.lora and isinstance(models[0], AttnProcsLayers):
            # pipeline.unet.load_attn_procs(input_dir)
            tmp_unet = UNet2DConditionModel.from_pretrained(
                config.pretrained.model,
                revision=config.pretrained.revision,
                subfolder="unet",
            )
            tmp_unet.load_attn_procs(input_dir)
            models[0].load_state_dict(
                AttnProcsLayers(tmp_unet.attn_processors).state_dict()
            )
            del tmp_unet
        elif not args.lora and isinstance(models[0], UNet2DConditionModel):
            load_model = UNet2DConditionModel.from_pretrained(
                input_dir, subfolder="unet"
            )
            models[0].register_to_config(**load_model.config)
            models[0].load_state_dict(load_model.state_dict())
            del load_model
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        models.pop()  # ensures that accelerate doesn't try to handle loading of the model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)
    
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
          
    
    # print("Trainable parameters: ")
     #print trainable parameters
    for name, param in audioldmpipeline.text_encoder_2.named_parameters():
        if param.requires_grad:
            print(name)
    for name, param in audioldmpipeline.text_encoder_2.named_parameters():
        if param.requires_grad:
            print(name)
    for name, param in unet.named_parameters():
        if param.requires_grad:
            print(name)
    for name, param in vae.named_parameters():
        if param.requires_grad:
            print(name)
    for name, param in vocoder.named_parameters():
        if param.requires_grad:
            print(name)
   
    
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

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
        num_workers=args.dataloader_num_workers,
    )
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
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                prompt_embeds = batch["prompt_embeds"]
                prompt_embeds = prompt_embeds.squeeze(-3)
                generated_prompt_embeds=batch["generated_prompt_embeds"]
                generated_prompt_embeds=generated_prompt_embeds.squeeze(-3)
                attention_mask=batch["attention_mask"]
                attention_mask=attention_mask.squeeze(-2)
                # print("prompt_embeds: ", prompt_embeds.shape)
                # print("generated_prompt_embeds: ", generated_prompt_embeds.shape)
                # print("attention_mask: ", attention_mask.shape)

                # Predict the noise residual
                # model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=None,class_labels=prompt_embeds).sample
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=generated_prompt_embeds,
                    encoder_hidden_states_1=prompt_embeds,
                    encoder_attention_mask_1=attention_mask,
                    return_dict=False,
                )[0]
                
                # Get the target for loss depending on the prediction type
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
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

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

                    if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                        audios = log_validation(
                            audioldmpipeline,
                            text_encoder, 
                            tokenizer, 
                            unet, vae, args, accelerator, weight_dtype, global_step, vocoder,
                            concept_audio_dir, 
                            validate_experiments=args.validate_experiments
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
