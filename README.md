# AP-adapter
This is the official implementation of AP-adapter. Please give me a star if you found this project useful~
## Installation
We provide a step by step series of examples that tell you how to get a development environment running.

```bash
git clone https://github.com/fundwotsai2001/AP-adapter-full.git
cd AP-adapter-full
```
## Downloading checkpoint
For AudioMAE checkpoint you can download it from 
[pretrain](https://drive.google.com/file/d/1ni_DV4dRf7GxM8k-Eirx71WP9Gg89wwu/view?usp=share_link)\
For AP-adapter checkpoint you can download it from
[AP-adpater](https://drive.google.com/drive/u/0/folders/1TPbiVx4ijjd2tdbLNmwPgpR8UUoRizmj)
```
gdown https://drive.google.com/uc?id=1ni_DV4dRf7GxM8k-Eirx71WP9Gg89wwu
gdown https://drive.google.com/uc?id=1rA1zgCdioOpUpds-CdxL8uOKTx1-cAcH
```


## Parameters in inference.py

We have standard parameter sets for the three tasks, you can go to [demo](https://young-almond-689.notion.site/Zero-shot-music-text-fusionfbbfeb0608664f61a6bf894d56e85820) to see the detail settings, or directly use the template in config.py, you can also change the prompt settings there. Note that the effect of hyper-parameters are mentioned in the paper, but generally "ap_scale" is proportional to the audio strength, "time_pooling" and "frequency_pooling" are inversely proportional to the audio control strength. You can adjust these parameters to fit your requirement, or just use the default settings.
```
python inferece --task timbre_transfer
python inferece --task style_transfer
python inferece --task accompaniment_generation
## if you want to try something cool, use test and change the template in config.py
python inferece --task test
```
Unfortunately, the accompaniment generation does not perform well enough with the previous training, we are still working on it.
## Train from scratch
It's also recommend to train from scratch if you have powerful computation resource, the checkpoint we provide was only trained for 35000 steps, with effective batchsize 32.
We only use 200k audio-text pairs from Audioset due to memory capacity. 
To use our training code, you can use https://github.com/dlrudco/Fast-Audioset-Download to download the dataset, and put "Fast-Audioset-Download" in the way below.
```
AP-ADAPTER-FULL/
├── pycache/
├── pipeline/
├── APadapter/
├── audio_encoder/
├── copied_cross_attention/
└── Fast-Audioset-Download/
    ├── csvs/
    ├── temps/
    └── wavs/
        ├── balanced_train/
        ├── eval/
        └── unbalanced_train/
            └── 000000
                ├── id__-5esUcUAk.json
                └── id__-5esUcUAk.m4a
```
After handling the dataset structure, you can run the command to train the adapter from scratch:
```
##change the DATA_DIR and OUTPUT_DIR in train.sh, and run
./train.sh
```
Or you can start from the previously downloaded checkpoint [AP-adpater](https://drive.google.com/drive/u/0/folders/1TPbiVx4ijjd2tdbLNmwPgpR8UUoRizmj).
```
##change the DATA_DIR and OUTPUT_DIR in finetune.sh, and run
./finetune.sh
```
