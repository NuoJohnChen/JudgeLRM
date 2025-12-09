#!/bin/bash




source ~/miniconda3/etc/profile.d/conda.sh
conda init
conda activate judgelm

pip install wandb==0.17
echo "\n\n\n\n"
python -c 'import wandb; print(wandb.__version__)'

bash --norc --noprofile -c "
python judgelm/train/get_dpo_data.py &&

torchrun --nproc_per_node=2 --master_port=20001 scripts/train_dpo_fixed.py \
    --model_name_or_path=\"/shared/ssd/models/Qwen2.5-3B-Instruct\" \
    --data_path /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_train_100k_dpo.jsonl \
    --bf16 True \
    --output_dir=\"/shared/hdd/nuochen/JudgeLM/output/Qwen2.5-3B-Instruct-evaluator-dpo-fixed\" \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --save_strategy steps \
    --save_steps 120 \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --beta 0.1 \
    --run_name qwen2.5-3b-evaluator-dpo-fixed \
    --report_to none
"
