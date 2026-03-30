#!/bin/bash


source ~/miniconda3/etc/profile.d/conda.sh
conda init
conda activate judgelm

MODEL_PATH="/user/JudgeLM/output/Qwen2.5-3B-Instruct-evaluator-dpo-fixed/checkpoint-9000/"
INPUT_PATH="/user/Logic-RL/testset-v1_update.json"
OUTPUT_PATH="/user/PandaLM/data/results_qwen253binstruct-evaluator-dpo-probability.json"


CUDA_VISIBLE_DEVICES=0,2 python /user/JudgeLM/scripts/test_dpo_probability.py \
--model_path "$MODEL_PATH" \
--input_path "$INPUT_PATH" \
--output_path "$OUTPUT_PATH" \
--seed 42

