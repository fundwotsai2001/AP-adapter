from pipeline.pipeline_audioldm2 import AudioLDM2Pipeline
import torch
import torch.nn.functional as F
import argparse
import itertools
from audio_encoder.AudioMAE import AudioMAEConditionCTPoolRand, extract_kaldi_fbank_feature
import os
import torchaudio
from torch.utils.data import Dataset, DataLoader
import random
# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
parser.add_argument("--train_gpt2", action="store_true")
parser.add_argument("--train_text_encoder", action="store_true")
parser.add_argument("--num_train_epochs", type=int, default=3)
parser.add_argument("--revision", type=str, default=None)
parser.add_argument("--device", type=int, default=0)

args = parser.parse_args()

# Initialize Accelerator
from accelerate import Accelerator
accelerator = Accelerator()
# Load the pipeline
audioldmpipeline = AudioLDM2Pipeline.from_pretrained(
    args.pretrained_model_name_or_path,
    safety_checker=None,
    revision=args.revision,
    torch_dtype=torch.float32,  # TODO: change it to weight_dtype
).to(f"cuda:{args.device}")

# Extract components from the pipeline
tokenizer=audioldmpipeline.tokenizer
tokenizer_2=audioldmpipeline.tokenizer_2
text_encoder=audioldmpipeline.text_encoder
text_encoder_2=audioldmpipeline.text_encoder_2
GPT2 = audioldmpipeline.language_model
projection_model=audioldmpipeline.projection_model
vae=audioldmpipeline.vae
unet=audioldmpipeline.unet
vocoder=audioldmpipeline.vocoder
noise_scheduler=audioldmpipeline.scheduler
# Freeze components if not training
if not args.train_gpt2:
    GPT2.requires_grad_(False)
if not args.train_text_encoder:
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
projection_model.requires_grad_(False)
vae.requires_grad_(False)
unet.requires_grad_(False)
optimizer_class = torch.optim.AdamW
params_to_optimize = (itertools.chain(GPT2.parameters()))
optimizer = optimizer_class(
        params_to_optimize,
        lr=1.0e-05,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )
train_dir_name = '/mnt/gestalt/home/fundwotsai/DreamSound/training_audio_LOA_v2'
class_dir_name = '/mnt/gestalt/home/fundwotsai/DreamSound/class_piano_mixed_flute_audio_LOA'


class AudioMAEDataset(Dataset):
    def __init__(self, Directory):
        """
        Args:
            directory (string): Directory with all the audio files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.Directory = Directory
        self.audio_files = os.listdir(Directory)

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_file = os.path.join(self.Directory, self.audio_files[idx])
        LOA = torch.load(audio_file)
        return LOA

train_dataset = AudioMAEDataset(train_dir_name)
class_dataset = AudioMAEDataset(class_dir_name)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
# class_loader = DataLoader(class_dataset, batch_size=32, shuffle=True)


# Training loop with teacher forcing
for epoch in range(args.num_train_epochs):
    for train_features in train_loader:
        GPT2.train()
        # Assuming 'text' is defined and holds the prompts
        prompt_embeds, attention_mask, generated_prompt_embeds = audioldmpipeline.encode_prompt(
            prompt="a recording of a hjrdgsjgsgt chinese flute solo.",
            device=accelerator.device,
            num_waveforms_per_prompt=1,
            do_classifier_free_guidance=False
        )
        # class_prompt_embeds, class_attention_mask, class_generated_prompt_embeds = audioldmpipeline.encode_prompt(
        #     prompt="a recording of a flute",
        #     device=accelerator.device,
        #     num_waveforms_per_prompt=1,
        #     do_classifier_free_guidance=False
        # )
        class_prompt_embeds, class_attention_mask, class_generated_prompt_embeds = audioldmpipeline.encode_prompt(
            prompt="a recording of a piano with hjrdgsjgsgt chinese flute",
            device=accelerator.device,
            num_waveforms_per_prompt=1,
            do_classifier_free_guidance=False
        )
        num = random.randint(0, 49)
        prior_loss = F.mse_loss(class_generated_prompt_embeds.float(), class_dataset[num].float(), reduction="mean")
        # Calculate loss using true sequences (teacher forcing)
        loss = F.mse_loss(generated_prompt_embeds.float(), train_features.float(), reduction="mean")
        # print("train_features:",train_features.shape)
        # print("generated_prompt_embeds:",generated_prompt_embeds.shape)
        total_loss = loss + prior_loss
        print(f"epoch:{epoch}, loss:{loss}, prior_loss:{prior_loss}")
        # Backward pass and optimization steps
        accelerator.backward(total_loss)
        optimizer.step()
        optimizer.zero_grad()

# Save the model after training
audioldmpipeline.save_pretrained("teacher_forcing_prior_mixed_flute")
print("saved pipeline")