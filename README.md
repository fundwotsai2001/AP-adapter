# AP-adapter

## Installation

Provide a step by step series of examples that tell you how to get a development environment running.

```bash
git clone https://github.com/fundwotsai2001/AP-adapter-full.git
cd AP-adapter-full
git lfs install
git clone https://huggingface.co/cvssp/audioldm2-large
pip install -r requirements.txt
```
## Downloading checkpoint
for AudioMAE checkpoint you can download it from 
[pretrain](https://drive.google.com/file/d/1ni_DV4dRf7GxM8k-Eirx71WP9Gg89wwu/view?usp=share_link)

You will need to change the path in audio_encoder/AudioMAE.py.
For IP-adapter checkpoint you can download it from
[IP-adpater](https://drive.google.com/drive/u/0/folders/1TPbiVx4ijjd2tdbLNmwPgpR8UUoRizmj)
```
gdown https://drive.google.com/uc?id=1ni_DV4dRf7GxM8k-Eirx71WP9Gg89wwu
gdown https://drive.google.com/uc?id=1rA1zgCdioOpUpds-CdxL8uOKTx1-cAcH
```
You will need to change the paths in inference.py


## Parameter in inference.py

AudioMAE will do pooling to both time and frequncy tokens in order to avoid just reconstruct the original audio. Largeer pooling rate will abandon information, thus will enhance the editability. We have standard parameter sets for the three tasks, you can go to https://young-almond-689.notion.site/Zero-shot-music-text-fusionfbbfeb0608664f61a6bf894d56e85820.

```
python inference.py --dir timbre trans --num_files 5 --audio_prompt_file path_to_audio.wav --audio_prompt_file2 path_to_secondary_audio.wav --ip_ckpt path_to_checkpoint --ip_scale 0.5 --time_pooling 2 --freq_pooling 2
```
## Train from scratch
It's also recommend to train from scratch if you have powerful computation resource, the checkpoint we provide was only trained for 35000 steps, with effective batchsize 32.
For the original training code, we use the Audioset download from https://github.com/dlrudco/Fast-Audioset-Download. We only use 200k audio-text pairs due to memory capacity. 
```
##change the DATA_DIR and OUTPUT_DIR
./train.sh
```
