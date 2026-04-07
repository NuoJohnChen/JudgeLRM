#!/bin/bash

# Reward Model training script
# Using Qwen2.5-3B as base model (non-Instruct)

source $CONDA_PREFIX/etc/profile.d/conda.sh
conda init
conda activate judgelm

#pip install wandb==0.17
export WANDB_API_KEY='<REDACTED>'
echo "\n\n\n\n"
python -c 'import wandb; print(wandb.__version__)'

bash --norc --noprofile -c "
python judgelm/train/get_dpo_data.py &&
# Convert DPO pairwise data to reward single-sample data
python scripts/convert_dpo_to_reward.py --input /path/to/datasets/JudgeLM-100K/judgelm_train_100k_dpo.jsonl --output /path/to/datasets/JudgeLM-100K/judgelm_train_100k_reward.jsonl &&
# Run Reward Model training (consuming reward data directly)
torchrun --nproc_per_node=2 --master_port=20002 scripts/train_reward_model.py \
    --model_name_or_path=\"/path/to/models/Qwen2.5-3B\" \
    --data_path /path/to/datasets/JudgeLM-100K/judgelm_train_100k_reward.jsonl \
    --bf16 True \
    --output_dir=\"/path/to/JudgeLM/output/Qwen2.5-3B-reward-model\" \
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
