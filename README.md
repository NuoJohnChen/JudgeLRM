# JudgeLRM:

## Overview

JudgeLRM is a family of judgment-oriented Large Language Models (LLMs) designed to enhance evaluative reasoning through reinforcement learning (RL) with judge-wise, outcome-driven rewards. It demonstrates that judgment is inherently a reasoning-intensive task and addresses the limitations of supervised fine-tuning (SFT) in pair-wise evaluation. Notably, JudgeLRM-3B surpasses GPT-4, and JudgeLRM-7B outperforms DeepSeek-R1.

## Reproducibility and Open Weights

Our submission provides a fully reproducible framework and open weights. Benchmarking, replication, and challenges are welcome via the checkpoints released at:

- https://huggingface.co/anonymous-D1C4/JudgeLRM-3B
- https://huggingface.co/anonymous-D1C4/JudgeLRM-4B
- https://huggingface.co/anonymous-D1C4/JudgeLRM-7B
- https://huggingface.co/anonymous-D1C4/JudgeLRM-8B
- https://huggingface.co/anonymous-D1C4/JudgeLRM-14B

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

## Baseline Evidence

Figures of convergence figures and descriptions of baseline training see baseline_evidence.

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


## For sensitivity to prompt 

Only the **system prompt** changes across the four variants; the **user message remains exactly the same**.

---

** V0 (Original)**

You are a helpful assistant. The assistant first performs a detailed,  
step-by-step reasoning process in its mind and then provides the user with  
the answer. The reasoning process and answer are enclosed within `<think>`  
`</think>` and `<answer>` `</answer>` tags, respectively, i.e.,  
`<think>` detailed reasoning process here, explaining each step of your evaluation for both assistants `</think>`  
`<answer>` answer here `</answer>`.  

Now the user asks you to judge the performance of two AI assistants in response to the question. Score assistants **1–10 (higher = better)**. Criteria includes **helpfulness, relevance, accuracy, and level of detail**. Avoid **order, length, style, or other bias**.

After thinking, when you finally reach a conclusion, clearly provide your evaluation scores within `<answer>` `</answer>` tags, e.g.:

`<answer>3</answer><answer>5</answer>`

---

 **V1 (Changed scoring wording)**

Replace **“Score assistants 1–10 (higher = better).”** with:

**“Rate each assistant from 1 to 10, with 10 being the best.”**

Everything else remains exactly the same.

---

 **V2 (Remove debias instruction)**

Directly delete the sentence:

**“Avoid order, length, style or other bias.”**

Everything else remains exactly the same.

---

 **V3 (Strengthened debias instruction)**

Replace **“Avoid order, length, style or other bias.”** with:

**“IMPORTANT: Do NOT let the order of presentation, response length, writing style, or formatting influence your scores. Focus solely on content quality.”**

Everything else remains exactly the same.
