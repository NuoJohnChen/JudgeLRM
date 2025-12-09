#!/bin/bash

# # w/ reference & w/o reverse
python ./judgelm/llm_judge/gen_model_judgement.py \
--model-path "/shared/ssd/models/Qwen2.5-7B-Instruct" \
--model-id 7b-full-model \
--question-file /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k.jsonl \
--answer-file /shared/hdd/nuochen/JudgeLM/output/Qwen2.5-7B-Instruct-full-model-pycharm-debug \
--reference-file /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k_references.jsonl \
--num-gpus-per-model 1 \
--num-gpus-total 8 \
--temperature 0.2 \
--if-reverse 0 \
--if-fast-eval 1

# # w/ reference & w/ reverse
python ./judgelm/llm_judge/gen_model_judgement.py \
--model-path "/shared/ssd/models/Qwen2.5-7B-Instruct" \
--model-id 7b-full-model \
--question-file /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k.jsonl \
--answer-file /shared/hdd/nuochen/JudgeLM/output/Qwen2.5-7B-Instruct-full-model-pycharm-debug-reverse \
--reference-file /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k_references.jsonl \
--num-gpus-per-model 1 \
--num-gpus-total 8 \
--temperature 0.2 \
--if-reverse 1 \
--if-fast-eval 1

python ./judgelm/llm_judge/eval_model_judgement.py \
--gt-answer-file-path /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k_gpt4.jsonl \
--sequential-pred-answer-file-path /shared/hdd/nuochen/JudgeLM/output/Qwen2.5-7B-Instruct-full-model-pycharm-debug \
--reversed-pred-answer-file-path /shared/hdd/nuochen/JudgeLM/output/Qwen2.5-7B-Instruct-full-model-pycharm-debug-reverse \
--save-path /shared/hdd/nuochen/JudgeLM/output/Qwen2.5-7B-full-model-pycharm-debug-reverse-metrics.json
