#!/bin/bash

# # w/o reference & w/o reverse
python ./judgelm/llm_judge/gen_model_judgement.py \
--model-path "/shared/hdd/nuochen/models/global_step_250" \
--model-id rl \
--question-file /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k.jsonl \
--answer-file /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result \
--num-gpus-per-model 1 \
--num-gpus-total 8 \
--temperature 0.2 \
--if-reverse 0 \
--if-fast-eval 1

# # w/o reference & w/ reverse
python ./judgelm/llm_judge/gen_model_judgement.py \
--model-path "/shared/hdd/nuochen/models/global_step_250" \
--model-id rl \
--question-file /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k.jsonl \
--answer-file /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result_new \
--reference-file /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k_references.jsonl \
--num-gpus-per-model 1 \
--num-gpus-total 8 \
--temperature 0.2 \
--if-reverse 1 \
--if-fast-eval 1

python ./judgelm/llm_judge/eval_model_judgement.py \
--gt-answer-file-path /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k_gpt4.jsonl \
--sequential-pred-answer-file-path /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result \
--reversed-pred-answer-file-path /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result_new \
--save-path /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result-metrics_111.json
