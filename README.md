# JudgeLRM:

## Overview

JudgeLRM is a family of judgment-oriented Large Language Models (LLMs) designed to enhance evaluative reasoning through reinforcement learning (RL) with judge-wise, outcome-driven rewards. It demonstrates that judgment is inherently a reasoning-intensive task and addresses the limitations of supervised fine-tuning (SFT) in pair-wise evaluation. Notably, JudgeLRM-3B surpasses GPT-4, and JudgeLRM-7B outperforms DeepSeek-R1.

## 🛠️ Environment

```
# Recommended Python version: 3.9.21
pip install -r requirements.txt
```

## 📂 Data Preprocess
To preprocess the data for training:

```
python src/examples/data_preprocess/judgelrm.py
```

## 🚀 Train JudgeLRM
```
# Training using GRPO
bash src/scripts/judgelrm_grpo7b_{n}gpu.sh

# Inference after training
python pandalm/utils/judgelrm_inference.py
```

## ⚖️ Inference & Evaluation
### General Inference

See `pandalm/utils` for specific scripts.

```
python pandalm/utils/judgelrm_{qwen3_}inference.py
python pandalm/calculate_result.py
```

### Bias Test

```
bash JudgeLM/scripts/step4eval_judge_on_judgelm_benchmark_rl.sh
```

### Reasoning Analysis

```
# Calculate reasoning rate
python data/markreasoning.py

# Calculate reasoning ability stats
python data/mark_reasoning_countabaility.py
python data/count_reasoning_countabaility.py
```

## 📉 Baselines
<details>

<summary><b>Click to expand all Baseline implementations</b></summary>

First, navigate to the baseline source directory:

```
cd baseline/src
```

### Baseline 1: DPO-ANSWER (Direct Preference Optimization)

```
bash train_dpo_fixed.sh
python convert_dpo_to_reward.py
bash test_reward_model.sh
```

### Baseline 2: CLS-RM (Classification Reward Model)

```
bash train_reward_model.sh
bash test_reward_model.sh
```

### Baseline 3: BRADLEY-TERRY (Pairwise Preference Model)

```
python train_bt_reward.py
python test_bt_reward.py
```

### Baseline 4: CROSS-BT (Single-Input Pairwise Bradley-Terry)

```
python train_bt_cross_encoder.py
python test_crossencoderbt.py
```

### Baseline 5: SFT-THINK / SFT-Distill-R1-Think *(Supervised Fine-Tuning with Structure)*

```
bash train_sft_think.sh
python eval_sft_think.py
```

### Baseline 6: DPO-RC (SPIN with R_content)

```
bash run_spin.sh
```

### Baseline: Single Judge

```
bash pandalm/utils/judgelrm_single_inference.py
```

For other inference scripts regarding baselines, please check `baseline/inference`.

</details>
