# AP-adapter
This is the official implementation of AP-adapter.
## Installation
We provide a step by step series of examples that tell you how to get a development environment running.

```bash
git clone https://github.com/fundwotsai2001/AP-adapter-full.git
cd AP-adapter-full
conda create -n APadapter python=3.11
conda activate APadapter
pip install -r requirements.txt

```
## Downloading checkpoint
for AudioMAE checkpoint you can download it from 
[pretrain](https://drive.google.com/file/d/1ni_DV4dRf7GxM8k-Eirx71WP9Gg89wwu/view?usp=share_link)

You will need to change the path in audio_encoder/AudioMAE.py.
For AP-adapter checkpoint you can download it from
[AP-adpater](https://drive.google.com/drive/u/0/folders/1TPbiVx4ijjd2tdbLNmwPgpR8UUoRizmj)
```
gdown https://drive.google.com/uc?id=1ni_DV4dRf7GxM8k-Eirx71WP9Gg89wwu
gdown https://drive.google.com/uc?id=1jbovWcsiVY4gzWfcqGq2O6ftOlIblAcw
```


## Parameters in inference.py

We have standard parameter sets for the three tasks, you can go to [demo](https://young-almond-689.notion.site/Zero-shot-music-text-fusionfbbfeb0608664f61a6bf894d56e85820) to see the detail settings, or directly use the template in config.py, you can also change the prompt settings there. Note that the effect of hyper-parameters are mentioned in the paper, but generally "ap_scale" is proportional to the audio strength, "time_pooling" and "frequency_pooling" are inversely proportional to the audio control strength. You can adjust these parameters to fit your requirement, or just use the default settings.
```
python inference.py --task timbre_transfer
python inference.py --task style_transfer
python inference.py --task accompaniment_generation
## if you want to try something cool, use test and change the template in config.py
python inference.py --task test
```
Unfortunately, the accompaniment generation does not perform well enough with the previous training, we are still working on it.
## Train from scratch
It's also recommend to train from scratch if you have powerful computation resource, the checkpoint we provide was only trained for 35000 steps, with effective batchsize 32.
For the original training code, we use the Audioset download from https://github.com/dlrudco/Fast-Audioset-Download. We only use 200k audio-text pairs due to memory capacity. 
```
##change the DATA_DIR and OUTPUT_DIR in train.sh, and run
./train.sh
```
