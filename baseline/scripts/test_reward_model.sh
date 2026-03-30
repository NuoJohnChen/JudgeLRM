#!/bin/bash


source ~/miniconda3/etc/profile.d/conda.sh
conda init
conda activate judgelm

echo "Testing Reward Model..."

python scripts/test_reward_model.py \
  --model_path "/user/JudgeLM/output/Qwen2.5-3B-reward-model-checkpoint-30000" \
  --base_model_path "models/Qwen2.5-3B" \
  --data_path "/user/Logic-RL/testset-v1_update.json" \
  --output_path "/user/JudgeLM/output/reward_model_results.json" \
  --num_samples 1000 \
  --device cuda

echo "Testing completed!"
