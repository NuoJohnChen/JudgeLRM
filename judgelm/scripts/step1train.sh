#!/bin/bash

torchrun --nproc_per_node=2 --master_port=20010 judgelm/train/train_mem.py \
    --model_name_or_path="/shared/hdd/nuochen/models/Qwen3-8B" \
    --data_path /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_train_100k.jsonl \
    --bf16 True \
    --output_dir="/disk2/nuochen/nuochen/JudgeLM/output/Qwen3-8B-Instruct-judgelmsft-evaluator" \
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
    
# torchrun --nproc_per_node=4 --master_port=20001 judgelm/train/train_mem.py \
#     --model_name_or_path="/shared/ssd/models/Qwen2.5-7B-Instruct" \
#     --data_path /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_train_100k_think.jsonl \
#     --bf16 True \
#     --output_dir="/shared/hdd/nuochen/JudgeLM/output/Qwen2.5-7B-Instruct-think-evaluator" \
#     --num_train_epochs 2 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 4 \
#     --eval_strategy no \
#     --save_strategy steps \
#     --save_steps 120 \
#     --save_total_limit 1 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type cosine \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 4096 \
#     --gradient_checkpointing True \
#     --run_name gemma-7b-evaluator \
#     --swap_aug_ratio 0.5 \
#     --ref_drop_ratio 0.5 \
#     --fsdp "full_shard auto_wrap offload" \


# torchrun --nproc_per_node=2 --master_port=20001 judgelm/train/train_mem.py \
#     --model_name_or_path="/shared/hdd/nuochen/models/gemma-7b" \
#     --data_path /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_train_100k.jsonl \
#     --bf16 True \
#     --output_dir="/shared/hdd/nuochen/JudgeLM/output/gemma-7b-evaluator" \
#     --num_train_epochs 2 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 32 \
#     --eval_strategy no \
#     --save_strategy steps \
#     --save_steps 120 \
#     --save_total_limit 1 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type cosine \
#     --logging_steps 1 \
#     --fsdp "full_shard auto_wrap offload" \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --run_name gemma-7b-evaluator \
#     --swap_aug_ratio 0.5 \
#     --ref_drop_ratio 0.5