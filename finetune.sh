#!/bin/bash
export DATA_DIR="/home/fundwotsai/DreamSound/Fast-Audioset-Download/wavs/unbalanced_train"
export OUTPUT_DIR="/home/fundwotsai/DreamSound/audioldm2-large-ipadapter-audioset-unet-random-pooling_v4"
export CUDA_VISIBLE_DEVICES=0
export MODEL_NAME="cvssp/audioldm2-large"

accelerate launch train_ipadapter_v2.py \
--pretrained_model_name_or_path=$MODEL_NAME \
--train_data_dir=$DATA_DIR \
--train_batch_size=8 \
--gradient_accumulation_steps=4 \
--max_train_steps=1000000 \
--learning_rate=1e-4 \
--output_dir=$OUTPUT_DIR \
--validation_steps=2000 \
--num_validation_audio_files=1 \
--checkpointing_steps=2000 \
--apadapter=True \
--resume_from_checkpoint "./pytorch_model.bin" \
--dataloader_num_workers 4 \
# --use_8bit_adam \
# --mixed_precision "bf16" \
# --enable_xformers_memory_efficient_attention \
# --train_gpt2 \
# --train_text_encoder \
# --with_prior_preservation \
# --class_prompt="chinese flute" \
# --class_data_dir="/home/fundwotsai/DreamSound/mix_chinese_flute_piano" \
# --lora=True \
