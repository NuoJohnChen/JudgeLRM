#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda init
conda activate judgelm

torchrun --nproc_per_node=2 --master_port=20004 scripts/train_sft_think.py \
  --model_name_or_path "/shared/ssd/models/Qwen2.5-3B-Instruct" \
  --data_path /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_train_100k_think.jsonl \
  --bf16 True \
  --output_dir "/disk2/nuochen/JudgeLM/output/Qwen2.5-3B-Instruct-sft-think" \
  --num_train_epochs 2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --save_strategy steps \
  --save_steps 150 \
  --save_total_limit 5 \
  --learning_rate 2e-5 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --tf32 True \
  --model_max_length 2048 \
  --run_name qwen2.5-3b-instruct-sft-think \
  --report_to none


