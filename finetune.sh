#!/bin/bash
export DATA_DIR="/data/home/fundwotsai/Fast-Audioset-Download/audioset_unbalanced_train_metadata_clean.json" ## Change to your path
export OUTPUT_DIR="/data/home/fundwotsai/test_AP/AP-adapter/audioldm2-large-ipadapter-audioset-unet-random-pooling_v4"  ## Change to your path
export CUDA_VISIBLE_DEVICES=0
export MODEL_NAME="cvssp/audioldm2-large"

accelerate launch train_apadapter_v2.py \
--pretrained_model_name_or_path=$MODEL_NAME \
--train_data_dir=$DATA_DIR \
--train_batch_size=7 \
--gradient_accumulation_steps=4 \
--max_train_steps=1000000 \
--learning_rate=1e-4 \
--output_dir=$OUTPUT_DIR \
--validation_steps=3000 \
--num_validation_audio_files=3 \
--checkpointing_steps=3000 \
--resume_from_checkpoint "./pytorch_model.bin" \
--apadapter=True \
--dataloader_num_workers 8 \
# --mixed_precision "fp16" \
# --use_8bit_adam \


