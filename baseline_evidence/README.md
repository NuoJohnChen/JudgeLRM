## Directory Structure

```
baseline_evidence/
├── loss_curves/                # Loss curve plots
│   ├── CLS_RM_loss.png
│   ├── BT_Reward_loss.png
│   ├── Cross_BT_loss.png
│   ├── SFT_Think_loss.png
│   └── SFT_base_loss.png
├── trainer_states/             # HuggingFace trainer_state.json (full loss history)
│   ├── CLS_RM_trainer_state.json
│   ├── BT_Reward_trainer_state.json
│   ├── Cross_BT_trainer_state.json
│   ├── SFT_Think_trainer_state.json
│   └── SFT_base_trainer_state.json
├── train_scripts/              # Training scripts (py + sh)
│   ├── CLS_RM_train_reward_model.py
│   ├── CLS_RM_step1train_reward_model.sh    # 3 epoch, Qwen2.5-3B
│   ├── BT_Reward_train_bt_reward.py         # no shell script, HF default 3 epoch
│   ├── Cross_BT_train_bt_cross_encoder.py   # example shows 2 epoch
│   ├── SFT_Think_train_sft_think.py
│   ├── SFT_Think_step1train_sft_think.sh    # 2 epoch, Qwen2.5-3B-Instruct
│   ├── SFT_step1train.sh                    # 2 epoch
│   └── util_convert_dpo_to_reward.py
└── wandb_configs/              # wandb config/summary/output.log per run
```

## Convergence Summary

| Method | Output Dir | Set Epoch | Trained To | Steps Done | Converged? |
|--------|-----------|-----------|------------|------------|------------|
| CLS-RM | Qwen2.5-3B-reward-model | 3 | 3.0 ep (100%) | 37317/37317 | YES - plateau at ~1.44 |
| BT-Reward | Qwen2.5-3B-bt-reward | 3 | 1.83 ep (61%) | 22800/37317 | YES - plateau at ~0.45 |
| Cross-BT | Qwen2.5-3B-bt-cross | 2 | 2.0 ep (100%) | 6220/6220 | YES - plateau at ~0.49 |
| SFT-Think | Qwen2.5-3B-Instruct-sft-think | 2 | 0.67 ep (34%) | 8400/24912 | YES - plateau at ~0.42 |
| SFT-base | Qwen2.5-3B-Instruct-evaluator | 2 | 0.29 ep (14%) | 7200/49822 | YES - plateau at ~0.54 |