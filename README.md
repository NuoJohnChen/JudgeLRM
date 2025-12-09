# JudgeLRM: Large Reasoning Models as a Judge <a href='https://arxiv.org/abs/2504.00050'><img src='https://img.shields.io/badge/arXiv-2504.00050-b31b1b.svg'></a> &nbsp;



<p align="center">

  📃 <a href="https://arxiv.org/abs/2504.00050" target="_blank">[Paper]</a> • 💻 <a href="https://github.com/NuoJohnChen/JudgeLRM" target="_blank">[Github]</a> • 🤗 <a href="https://huggingface.co/nuojohnchen/JudgeLRM-7B" target="_blank">[Models]</a> • <img src="https://osspicgo.oss-cn-shanghai.aliyuncs.com/hf_space_icon.svg"> <a href="https://huggingface.co/spaces/nuojohnchen/JudgeLRMDemo" target="_blank">[Playground]</a>

</p>

## Overview

JudgeLRM is a family of judgment-oriented Large Language Models (LLMs) designed to enhance evaluative reasoning through reinforcement learning (RL) with judge-wise, outcome-driven rewards. It demonstrates that judgment is inherently a reasoning-intensive task and addresses the limitations of supervised fine-tuning (SFT) in pair-wise evaluation. Notably, JudgeLRM-3B surpasses GPT-4, and JudgeLRM-7B outperforms DeepSeek-R1.

[Explore](https://huggingface.co/spaces/nuojohnchen/JudgeLRMDemo) JudgeLRM’s reasoning capabilities and detailed comparisons by testing it against other Hugging Face models with your own questions!

## 🛠️ Environment

### For Qwen2.5 Environment

```
# Recommended Python version: 3.9.21
pip install -r requirements.txt
```

### For Qwen3 Environment

```
# Recommended Python version: 3.10.18
pip install -r requirements_qwen3.txt

# Overwrite src/verl with qwen3 specific source
cp -r src/verl_qwen3/* src/verl/ 
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

## 🙌 Acknowledgements

- [LogicRL](https://github.com/Unakar/Logic-RL) 🔗
- [JudgeLM](https://github.com/Jiayi-Pan/TinyZero) 🔗
- [PandaLM](https://github.com/AlphaPav/mem-kk-logic) 🔗

---

## 🖊️ Citation
If you find this repo useful for your research, please consider citing our paper:

```
@misc{nuo2025judgelrm,
      title={JudgeLRM: Large Reasoning Models as a Judge}, 
      author={Nuo Chen, Zhiyuan Hu, Qingyun Zou, Jiaying Wu, Qian Wang, Bryan Hooi, Bingsheng He},
      year={2025},
      eprint={2504.00050},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2504.00050}, 
}
```
