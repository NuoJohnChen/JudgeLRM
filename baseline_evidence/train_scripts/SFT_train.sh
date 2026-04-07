#!/bin/bash

torchrun --nproc_per_node=2 --master_port=20010 judgelm/train/train_mem.py \
    --model_name_or_path="/path/to/models/Qwen3-8B" \
    --data_path /path/to/datasets/JudgeLM-100K/judgelm_train_100k.jsonl \
    --bf16 True \
    --output_dir="/path/to/JudgeLM/output/Qwen3-8B-Instruct-judgelmsft-evaluator" \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --eval_strategy no \
    --save_safetensors True \
    --save_strategy steps \
    --save_steps 400 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --run_name Qwen3-8B-evaluator \
    --swap_aug_ratio 0.5 \
    --ref_drop_ratio 0.5 \
    --fsdp "full_shard auto_wrap offload" \
    --fsdp_config fsdp_config.json