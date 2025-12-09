#!/bin/bash

python ./judgelm/llm_judge/gen_model_judgement.py \
--model-path "/shared/ssd/models/Qwen2.5-7B-Instruct" \
--model-id 7b-full-model \
--question-file /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k.jsonl \
--answer-file /shared/hdd/nuochen/JudgeLM/output/Qwen2.5-7B-Instruct-full-model \
--num-gpus-per-model 1 \
--num-gpus-total 8 \
--temperature 0.2 \
--reference-file /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k_references.jsonl \
--if-fast-eval 1
