#!/bin/bash
export DATA_DIR="/home/fundwotsai/DreamSound/Fast-Audioset-Download/wavs/unbalanced_train" # path to be edit 
export OUTPUT_DIR="/home/fundwotsai/DreamSound/audioldm2-large-ipadapter-audioset-unet-random-pooling_v3" # path to be edit 
export CUDA_VISIBLE_DEVICES=0
export MODEL_NAME="cvssp/audioldm2-large"

accelerate launch train_ipadapter_v2.py \
--pretrained_model_name_or_path=$MODEL_NAME \
--train_data_dir=$DATA_DIR \
--train_batch_size=4 \
--gradient_accumulation_steps=4 \
--max_train_steps=1000000 \
--learning_rate=1e-4 \
--output_dir=$OUTPUT_DIR \
--validation_steps=1000 \
--num_validation_audio_files=1 \
--num_vectors=1 \
--checkpointing_steps=1000 \
--apadapter=True \
# --train_gpt2 \
# --train_text_encoder \
# --with_prior_preservation \
# --class_prompt="chinese flute" \
# --class_data_dir="/home/fundwotsai/DreamSound/mix_chinese_flute_piano" \
# --lora=True \
