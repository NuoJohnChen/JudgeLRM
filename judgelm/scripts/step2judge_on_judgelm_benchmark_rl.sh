#!/bin/bash

python ./judgelm/llm_judge/gen_model_judgement.py \
--model-path "/shared/hdd/nuochen/models/GRPO_logic_KK/actor/global_step_720/" \
--model-id id \
--question-file /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k.jsonl \
--answer-file /shared/hdd/nuochen/models/GRPO_logic_KK/actor \
--num-gpus-per-model 1 \
--num-gpus-total 8 \
--temperature 0.2 \
--reference-file /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k_references.jsonl \
--if-fast-eval 1

