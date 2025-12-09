#!/bin/bash

# #Qwen3-8B 6k02 (not the intended evaluation setup)
# # w/o reference & w/o reverse
# python ./judgelm/llm_judge/gen_model_judgement.py \
# --model-path "/shared/hdd/nuochen/models/Qwen3-8B" \
# --model-id 8b \
# --question-file /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k.jsonl \
# --answer-file /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result_8b \
# --num-gpus-per-model 1 \
# --num-gpus-total 1 \
# --temperature 0.2 \
# --if-reverse 0 \
# --if-fast-eval 0 \
# --question-begin 1 \
# --question-end 10

# # # w/o reference & w/ reverse
# python ./judgelm/llm_judge/gen_model_judgement.py \
# --model-path "/shared/hdd/nuochen/models/Qwen3-8B" \
# --model-id 8b \
# --question-file /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k.jsonl \
# --answer-file /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result_8b_new \
# --reference-file /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k_references.jsonl \
# --num-gpus-per-model 1 \
# --num-gpus-total 8 \
# --temperature 0.2 \
# --if-reverse 1 \
# --if-fast-eval 1

# python ./judgelm/llm_judge/eval_model_judgement.py \
# --gt-answer-file-path /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k_gpt4.jsonl \
# --sequential-pred-answer-file-path /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result_8b \
# --reversed-pred-answer-file-path /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result_8b_new \
# --save-path /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result-metrics_8b.json


#Qwen3-4B a100 (not the intended evaluation; many ties)
# # # w/o reference & w/o reverse
# python ./judgelm/llm_judge/gen_model_judgement.py \
# --model-path "/shared/hdd/nuochen/models/Qwen3-4B" \
# --model-id 4b \
# --question-file /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k.jsonl \
# --answer-file /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result_4b \
# --num-gpus-per-model 1 \
# --num-gpus-total 8 \
# --temperature 0.2 \
# --if-reverse 0 \
# --if-fast-eval 1

# # # w/o reference & w/ reverse
# python ./judgelm/llm_judge/gen_model_judgement.py \
# --model-path "/shared/hdd/nuochen/models/Qwen3-4B" \
# --model-id 4b \
# --question-file /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k.jsonl \
# --answer-file /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result_4b_new \
# --reference-file /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k_references.jsonl \
# --num-gpus-per-model 1 \
# --num-gpus-total 8 \
# --temperature 0.2 \
# --if-reverse 1 \
# --if-fast-eval 1

# python ./judgelm/llm_judge/eval_model_judgement.py \
# --gt-answer-file-path /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k_gpt4.jsonl \
# --sequential-pred-answer-file-path /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result_4b \
# --reversed-pred-answer-file-path /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result_4b_new \
# --save-path /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result-metrics_4b.json


# # Qwen3-8B-SFT-think 6k04
# # # w/o reference & w/o reverse
# python ./judgelm/llm_judge/gen_model_judgement.py \
# --model-path "/disk2/nuochen/nuochen/JudgeLM/output/Qwen3-8B-Instruct-think-evaluator/checkpoint" \
# --model-id 8bsft \
# --question-file /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k.jsonl \
# --answer-file /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result_8bsft3k_newprompt \
# --num-gpus-per-model 1 \
# --num-gpus-total 8 \
# --temperature 0.2 \
# --if-reverse 0 \
# --if-fast-eval 1

# # w/o reference & w/ reverse
# python ./judgelm/llm_judge/gen_model_judgement.py \
# --model-path "/disk2/nuochen/nuochen/JudgeLM/output/Qwen3-8B-Instruct-think-evaluator/checkpoint" \
# --model-id 8bsft \
# --question-file /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k.jsonl \
# --answer-file /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result_8bsft3k_new \
# --reference-file /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k_references.jsonl \
# --num-gpus-per-model 1 \
# --num-gpus-total 8 \
# --temperature 0.2 \
# --if-reverse 1 \
# --if-fast-eval 1

# python ./judgelm/llm_judge/eval_model_judgement.py \
# --gt-answer-file-path /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k_gpt4.jsonl \
# --sequential-pred-answer-file-path /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result_8bsft3k \
# --reversed-pred-answer-file-path /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result_8bsft3k_new \
# --save-path /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result-metrics_8bsft3k.json

# # Qwen3-8B-SFT 6k02
# # # w/o reference & w/o reverse
# python ./judgelm/llm_judge/gen_model_judgement.py \
# --model-path "/disk2/nuochen/nuochen/JudgeLM/output/Qwen3-8B-Instruct-judgelmsft-evaluator/checkpoint" \
# --model-id 8bsft \
# --question-file /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k.jsonl \
# --answer-file /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result_8bsft2k7_test \
# --num-gpus-per-model 1 \
# --num-gpus-total 8 \
# --temperature 0.2 \
# --if-reverse 0 \
# --if-fast-eval 1

# # # w/o reference & w/ reverse
# python ./judgelm/llm_judge/gen_model_judgement.py \
# --model-path "/disk2/nuochen/nuochen/JudgeLM/output/Qwen3-8B-Instruct-judgelmsft-evaluator/checkpoint" \
# --model-id 8bsft \
# --question-file /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k.jsonl \
# --answer-file /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result_8bsft2k7_new \
# --reference-file /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k_references.jsonl \
# --num-gpus-per-model 1 \
# --num-gpus-total 8 \
# --temperature 0.2 \
# --if-reverse 1 \
# --if-fast-eval 1

# python ./judgelm/llm_judge/eval_model_judgement.py \
# --gt-answer-file-path /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k_gpt4.jsonl \
# --sequential-pred-answer-file-path /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result_8bsft2k7 \
# --reversed-pred-answer-file-path /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result_8bsft2k7_new \
# --save-path /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result-metrics_8bsft2k7.json

# # Qwen3-4B-SFT 6k03
# # # w/o reference & w/o reverse
# python ./judgelm/llm_judge/gen_model_judgement.py \
# --model-path "/disk2/nuochen/nuochen/JudgeLM/output/Qwen3-4B-Instruct-judgelmsft-evaluator/checkpoint/" \
# --model-id 4bsft \
# --question-file /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k.jsonl \
# --answer-file /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result_4bsft9k5 \
# --num-gpus-per-model 1 \
# --num-gpus-total 8 \
# --temperature 0.2 \
# --if-reverse 0 \
# --if-fast-eval 1

# # # w/o reference & w/ reverse
# python ./judgelm/llm_judge/gen_model_judgement.py \
# --model-path "/disk2/nuochen/nuochen/JudgeLM/output/Qwen3-4B-Instruct-judgelmsft-evaluator/checkpoint/" \
# --model-id 4bsft \
# --question-file /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k.jsonl \
# --answer-file /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result_4bsft9k5_new \
# --reference-file /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k_references.jsonl \
# --num-gpus-per-model 1 \
# --num-gpus-total 8 \
# --temperature 0.2 \
# --if-reverse 1 \
# --if-fast-eval 1

# python ./judgelm/llm_judge/eval_model_judgement.py \
# --gt-answer-file-path /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k_gpt4.jsonl \
# --sequential-pred-answer-file-path /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result_4bsft9k5 \
# --reversed-pred-answer-file-path /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result_4bsft9k5_new \
# --save-path /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result-metrics_4bsft9k5.json



# # SFT-Think(Qwen2.5-3B-Ins) 6k04
# # # w/o reference & w/o reverse
# python ./judgelm/llm_judge/gen_model_judgement.py \
# --model-path "/disk2/nuochen/JudgeLM/output/Qwen2.5-3B-Instruct-sft-think/checkpoint/" \
# --model-id 3bsftthink \
# --question-file /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k.jsonl \
# --answer-file /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result_3bsftthink \
# --num-gpus-per-model 1 \
# --num-gpus-total 8 \
# --temperature 0.2 \
# --if-reverse 0 \
# --if-fast-eval 1

# # # w/o reference & w/ reverse
# python ./judgelm/llm_judge/gen_model_judgement.py \
# --model-path "/disk2/nuochen/JudgeLM/output/Qwen2.5-3B-Instruct-sft-think/checkpoint/" \
# --model-id 3bsftthink \
# --question-file /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k.jsonl \
# --answer-file /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result_3bsftthink_new \
# --reference-file /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k_references.jsonl \
# --num-gpus-per-model 1 \
# --num-gpus-total 8 \
# --temperature 0.2 \
# --if-reverse 1 \
# --if-fast-eval 1

# python ./judgelm/llm_judge/eval_model_judgement.py \
# --gt-answer-file-path /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k_gpt4.jsonl \
# --sequential-pred-answer-file-path /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result_3bsftthink \
# --reversed-pred-answer-file-path /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result_3bsftthink_new \
# --save-path /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result-metrics_3bsftthink.json

## JudgeLRM-14B h100 0,1 (不是这么测的)
# # w/o reference & w/o reverse
# python ./judgelm/llm_judge/gen_model_judgement.py \
# --model-path "/shared/hdd/nuochen/models/JudgeLRM-14B-reward-wo-score-step400" \
# --model-id lrm14b \
# --question-file /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k.jsonl \
# --answer-file /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result_lrm14b \
# --num-gpus-per-model 1 \
# --num-gpus-total 2 \
# --temperature 0.2 \
# --if-reverse 0 \
# --if-fast-eval 0 \
# --judge-format judgelrm

# # # w/o reference & w/ reverse
# python ./judgelm/llm_judge/gen_model_judgement.py \
# --model-path "/shared/hdd/nuochen/models/JudgeLRM-14B-reward-wo-score-step400" \
# --model-id lrm14b \
# --question-file /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k.jsonl \
# --answer-file /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result_lrm14b_new \
# --num-gpus-per-model 1 \
# --num-gpus-total 2 \
# --temperature 0.2 \
# --if-reverse 1 \
# --if-fast-eval 0 \
# --judge-format judgelrm

# python ./judgelm/llm_judge/eval_model_judgement.py \
# --gt-answer-file-path /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k_gpt4.jsonl \
# --sequential-pred-answer-file-path /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result_lrm14b \
# --reversed-pred-answer-file-path /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result_lrm14b_new \
# --save-path /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result-metrics_lrm14b.json


# # JudgeLRM-8B h100 2,3
# w/o reference & w/o reverse
# python ./judgelm/llm_judge/gen_model_judgement.py \
# --model-path "/shared/hdd/nuochen/models/JudgeLRM-8B" \
# --model-id lrm8b \
# --question-file /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k.jsonl \
# --answer-file /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result_lrm8b \
# --num-gpus-per-model 1 \
# --num-gpus-total 2 \
# --temperature 0.2 \
# --if-reverse 0 \
# --if-fast-eval 0 \
# --judge-format judgelrm

# # # w/o reference & w/ reverse
# python ./judgelm/llm_judge/gen_model_judgement.py \
# --model-path "/shared/hdd/nuochen/models/JudgeLRM-8B" \
# --model-id lrm8b \
# --question-file /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k.jsonl \
# --answer-file /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result_lrm8b_new \
# --num-gpus-per-model 1 \
# --num-gpus-total 2 \
# --temperature 0.2 \
# --if-reverse 1 \
# --if-fast-eval 0 \
# --judge-format judgelrm

# python ./judgelm/llm_judge/eval_model_judgement.py \
# --gt-answer-file-path /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k_gpt4.jsonl \
# --sequential-pred-answer-file-path /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result_lrm8b \
# --reversed-pred-answer-file-path /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result_lrm8b_new \
# --save-path /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result-metrics_lrm8b.json


# # JudgeLRM-4B a100 6,7 (reversea6k03)
# # w/o reference & w/o reverse
# python ./judgelm/llm_judge/gen_model_judgement.py \
# --model-path "/shared/hdd/nuochen/models/JudgeLRM-4B" \
# --model-id lrm4b \
# --question-file /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k.jsonl \
# --answer-file /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result_lrm4b \
# --num-gpus-per-model 1 \
# --num-gpus-total 2 \
# --temperature 0.2 \
# --if-reverse 0 \
# --if-fast-eval 0 \
# --judge-format judgelrm

# # # w/o reference & w/o reverse
# python ./judgelm/llm_judge/gen_model_judgement.py \
# --model-path "/shared/hdd/nuochen/models/JudgeLRM-4B" \
# --model-id lrm4b \
# --question-file /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k.jsonl \
# --answer-file /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result_lrm4b_new_2500 \
# --num-gpus-per-model 1 \
# --num-gpus-total 2 \
# --temperature 0.2 \
# --if-reverse 1 \
# --if-fast-eval 0 \
# --judge-format judgelrm
# #--question-begin 2800

# python ./judgelm/llm_judge/eval_model_judgement.py \
# --gt-answer-file-path /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k_gpt4.jsonl \
# --sequential-pred-answer-file-path /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result_lrm4b \
# --reversed-pred-answer-file-path /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result_lrm4b_new \
# --save-path /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result-metrics_lrm4b.json


# JudgeLRM-3B a100 3,5
# # w/o reference & w/o reverse
# python ./judgelm/llm_judge/gen_model_judgement.py \
# --model-path "/shared/hdd/nuochen/models/JudgeLRM-3B" \
# --model-id lrm3b \
# --question-file /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k.jsonl \
# --answer-file /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result_lrm3b \
# --num-gpus-per-model 1 \
# --num-gpus-total 2 \
# --temperature 0.2 \
# --if-reverse 0 \
# --if-fast-eval 0 \
# --judge-format judgelrm

# # # w/o reference & w/ reverse
# python ./judgelm/llm_judge/gen_model_judgement.py \
# --model-path "/shared/hdd/nuochen/models/JudgeLRM-3B" \
# --model-id lrm3b \
# --question-file /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k.jsonl \
# --answer-file /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result_lrm3b_new \
# --num-gpus-per-model 1 \
# --num-gpus-total 1 \
# --temperature 0.2 \
# --if-reverse 1 \
# --if-fast-eval 0 \
# --judge-format judgelrm

# python ./judgelm/llm_judge/eval_model_judgement.py \
# --gt-answer-file-path /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k_gpt4.jsonl \
# --sequential-pred-answer-file-path /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result_lrm3b \
# --reversed-pred-answer-file-path /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result_lrm3b_new \
# --save-path /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result-metrics_lrm3b.json


# JudgeLRM-7B a100 1,2
# # # w/o reference & w/o reverse
# python ./judgelm/llm_judge/gen_model_judgement.py \
# --model-path "/shared/hdd/nuochen/models/JudgeLRM-7B" \
# --model-id lrm7b \
# --question-file /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k.jsonl \
# --answer-file /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result_lrm7b \
# --num-gpus-per-model 1 \
# --num-gpus-total 1 \
# --temperature 0.2 \
# --if-reverse 0 \
# --if-fast-eval 0 \
# --judge-format judgelrm

# # w/o reference & w/ reverse
# python ./judgelm/llm_judge/gen_model_judgement.py \
# --model-path "/shared/hdd/nuochen/models/JudgeLRM-7B" \
# --model-id lrm7b \
# --question-file /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k.jsonl \
# --answer-file /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result_lrm7b_new \
# --num-gpus-per-model 1 \
# --num-gpus-total 1 \
# --temperature 0.2 \
# --if-reverse 1 \
# --if-fast-eval 0 \
# --judge-format judgelrm

# python ./judgelm/llm_judge/eval_model_judgement.py \
# --gt-answer-file-path /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k_gpt4.jsonl \
# --sequential-pred-answer-file-path /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result_lrm7b \
# --reversed-pred-answer-file-path /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result_lrm7b_new \
# --save-path /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result-metrics_lrm7b.json


# # JudgeLRM-8B h100 2,3
# # w/o reference & w/o reverse
# python ./judgelm/llm_judge/gen_model_judgement.py \
# --model-path "/shared/hdd/nuochen/models/JudgeLRM-8B" \
# --model-id lrm8b \
# --question-file /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k.jsonl \
# --answer-file /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result_lrm8bref \
# --num-gpus-per-model 1 \
# --num-gpus-total 2 \
# --temperature 0.1 \
# --if-reverse 0 \
# --if-fast-eval 0 \
# --judge-format judgelrm \
# --reference-file /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k_references.jsonl


# # # w/o reference & w/ reverse
python ./judgelm/llm_judge/gen_model_judgement.py \
--model-path "/shared/hdd/nuochen/models/JudgeLRM-8B" \
--model-id lrm8b \
--question-file /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k.jsonl \
--answer-file /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result_lrm8b_newref \
--num-gpus-per-model 1 \
--num-gpus-total 2 \
--temperature 0.1 \
--if-reverse 1 \
--if-fast-eval 0 \
--judge-format judgelrm \
--reference-file /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k_references.jsonl


# python ./judgelm/llm_judge/eval_model_judgement.py \
# --gt-answer-file-path /shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k_gpt4.jsonl \
# --sequential-pred-answer-file-path /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result_lrm8bref \
# --reversed-pred-answer-file-path /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result_lrm8b_newref \
# --save-path /shared/hdd/nuochen/models/GRPO_logic_KK/actor/result-metrics_lrm8bref.json
