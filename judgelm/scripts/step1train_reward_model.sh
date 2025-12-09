#!/bin/bash

# Reward Model training script
# Base model: Qwen2.5-3B (non-Instruct version)

source ~/miniconda3/etc/profile.d/conda.sh
conda init
conda activate judgelm

#pip install wandb==0.17
echo "\n\n\n\n"
python -c 'import wandb; print(wandb.__version__)'

bash --norc --noprofile -c "
python judgelm/train/get_dpo_data.py &&
# First convert DPO pairwise data to single-sample reward data
python scripts/convert_dpo_to_reward.py --input /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_train_100k_dpo.jsonl --output /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_train_100k_reward.jsonl &&
# Train the Reward Model using the converted reward data
torchrun --nproc_per_node=2 --master_port=20002 scripts/train_reward_model.py \
    --model_name_or_path=\"/shared/ssd/models/Qwen2.5-3B\" \
    --data_path /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_train_100k_reward.jsonl \
    --bf16 True \
    --output_dir=\"/shared/hdd/nuochen/JudgeLM/output/Qwen2.5-3B-reward-model\" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --save_strategy steps \
    --save_steps 200 \
    --save_total_limit 3 \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 2048 \
    --run_name qwen2.5-3b-reward-model \
    --report_to none
"
