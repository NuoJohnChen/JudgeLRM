#!/bin/bash

# judge
# python ./judgelm/llm_judge/gen_model_judgement_multi.py \
# --model-path "/shared/hdd/nuochen/JudgeLM/output/DeepSeek-R1-Distill-Qwen-7B-evaluator/checkpoint/" \
# --model-id 7b-full-model \
# --question-file /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k.jsonl \
# --answer-file /shared/hdd/nuochen/JudgeLM/output/DeepSeek-R1-Distill-Qwen-7B-full-model \
# --num-gpus-per-model 1 \
# --num-gpus-total 8 \
# --temperature 0.2 \
# --reference-file /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k_references.jsonl \
# --if-fast-eval 1 \
# --answer-num 1

# # judge
# python ./judgelm/llm_judge/gen_model_judgement_multi.py \
# --model-path "/shared/hdd/nuochen/JudgeLM/output/vicuna-7b-v1.3-evaluator/checkpoint/" \
# --model-id 7b-full-model \
# --question-file /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k.jsonl \
# --answer-file /shared/hdd/nuochen/JudgeLM/output/vicuna-7b-v1.3-full-model \
# --num-gpus-per-model 1 \
# --num-gpus-total 8 \
# --temperature 0.2 \
# --reference-file /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k_references.jsonl \
# --if-fast-eval 1 \
# --answer-num 1


# python ./judgelm/llm_judge/gen_model_judgement_multi.py \
# --model-path "/shared/hdd/nuochen/JudgeLM/output/Qwen2.5-3B-Instruct-evaluator/checkpoint" \
# --model-id 3b-full-model \
# --question-file /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k.jsonl \
# --answer-file /shared/hdd/nuochen/JudgeLM/output/Qwen2.5-3B-Instruct-evaluator/result \
# --num-gpus-per-model 1 \
# --num-gpus-total 8 \
# --temperature 0.2 \
# --reference-file /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k_references.jsonl \
# --if-fast-eval 1 \
# --answer-num 1

# python ./judgelm/llm_judge/gen_model_judgement_multi.py \
# --model-path "/shared/hdd/nuochen/JudgeLM/output/Qwen2.5-3B-Instruct-evaluator-dpo" \
# --model-id 3b-full-model \
# --question-file /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k.jsonl \
# --answer-file /shared/hdd/nuochen/JudgeLM/output/Qwen2.5-3B-Instruct-evaluator-dpo/result \
# --num-gpus-per-model 1 \
# --num-gpus-total 8 \
# --temperature 0.2 \
# --reference-file /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k_references.jsonl \
# --if-fast-eval 1 \
# --answer-num 1

python ./judgelm/llm_judge/gen_model_judgement_multi.py \
--model-path "/shared/ssd/models/Qwen2.5-7B-Instruct" \
--model-id 7b-full-model \
--question-file /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k.jsonl \
--answer-file /shared/hdd/nuochen/JudgeLM/output/Qwen2.5-7B-Instruct-full-model \
--num-gpus-per-model 1 \
--num-gpus-total 8 \
--temperature 0.2 \
--reference-file /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k_references.jsonl \
--if-fast-eval 1 \
--answer-num 1

