#!/bin/bash

python ./judgelm/llm_judge/gen_model_judgement.py \
--model-path "/shared/hdd/nuochen/JudgeLM/output/Qwen2.5-3B-Instruct-evaluator-dpo" \
--model-id 3b-full-model \
--question-file /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k.jsonl \
--answer-file /shared/hdd/nuochen/JudgeLM/output/Qwen2.5-3B-Instruct-evaluator-dpo/result \
--num-gpus-per-model 1 \
--num-gpus-total 8 \
--temperature 0.2 \
--if-fast-eval 1