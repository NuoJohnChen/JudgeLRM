#!/usr/bin/env python3
"""
Bradley–Terry pairwise reward-model training.

Trains a reward head r_θ(x) on top of a base LM encoder (AutoModel) with loss:
  L = BCEWithLogits(r(c) - r(r), target_p)
where target_p ∈ (0,1) is either 1 (hard preference) or a soft label from
scores via target_p = sigmoid(k * (s_c - s_r)).

Input data (jsonl per line): expects either JudgeLM DPO-style items:
{
  "chosen": [{"question": str, "answer": str}],
  "rejected": [{"question": str, "answer": str}],
  "chosen_score": float,    # optional
  "rejected_score": float   # optional
}

Prompts are built in Qwen 2.5 chat format.
"""

import sys
from pathlib import Path
import json
from dataclasses import dataclass, field
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments


def build_qwen_prompt(question: str, answer: str) -> str:
    return (
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n{answer}<|im_end|>"
    )


class BTRewardHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        # hidden_state: [B, H]
        return self.proj(hidden_state).squeeze(-1)


class BTRewardModel(nn.Module):
    def __init__(self, base_model: AutoModel, hidden_size: int):
        super().__init__()
        self.base = base_model
        self.head = BTRewardHead(hidden_size)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # [B, T, H]
        seq_lens = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(last_hidden.size(0), device=last_hidden.device)
        last_states = last_hidden[batch_indices, seq_lens]
        return last_states

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Compute rewards for chosen and rejected
        ch_h = self.encode(batch["chosen_input_ids"], batch["chosen_attention_mask"])  # [B, H]
        rj_h = self.encode(batch["rejected_input_ids"], batch["rejected_attention_mask"])  # [B, H]
        r_ch = self.head(ch_h)  # [B]
        r_rj = self.head(rj_h)  # [B]

        logits = r_ch - r_rj  # [B]
        targets = batch["targets"]  # [B], in (0,1)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        return {"loss": loss, "logits": logits, "r_ch": r_ch, "r_rj": r_rj}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="/path/to/models/Qwen2.5-3B")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to DPO-style pairwise data jsonl"})
    soft_label_k: float = field(default=0.5, metadata={"help": "k for sigmoid(k*(s_c-s_r)) soft label. Set 0 for hard label=1."})


@dataclass
class BTTrainingArguments(transformers.TrainingArguments):
    remove_unused_columns: bool = field(default=False)
    model_max_length: int = field(default=2048)
    cache_dir: Optional[str] = field(default=None)


class BTDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, soft_label_k: float):
        self.tokenizer = tokenizer
        self.k = soft_label_k
        self.items = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    ch = obj.get("chosen", [{}])[0]
                    rj = obj.get("rejected", [{}])[0]
                    q1, a1 = str(ch.get("question", "")).strip(), str(ch.get("answer", "")).strip()
                    q2, a2 = str(rj.get("question", "")).strip(), str(rj.get("answer", "")).strip()
                    if not q1 or not a1 or not q2 or not a2:
                        continue
                    s_c = obj.get("chosen_score", None)
                    s_r = obj.get("rejected_score", None)
                    self.items.append({
                        "q_c": q1, "a_c": a1, "q_r": q2, "a_r": a2,
                        "s_c": s_c, "s_r": s_r,
                    })
                except Exception:
                    continue

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        it = self.items[idx]
        p_c = build_qwen_prompt(it["q_c"], it["a_c"])  # chosen
        p_r = build_qwen_prompt(it["q_r"], it["a_r"])  # rejected
        tok_c = self.tokenizer(p_c, truncation=True, max_length=self.tokenizer.model_max_length,
                               padding="max_length", return_tensors="pt")
        tok_r = self.tokenizer(p_r, truncation=True, max_length=self.tokenizer.model_max_length,
                               padding="max_length", return_tensors="pt")
        chosen_input_ids = tok_c.input_ids.squeeze(0)
        chosen_attention_mask = tok_c.attention_mask.squeeze(0)
        rejected_input_ids = tok_r.input_ids.squeeze(0)
        rejected_attention_mask = tok_r.attention_mask.squeeze(0)
        # target probability
        if isinstance(it.get("s_c"), (int, float)) and isinstance(it.get("s_r"), (int, float)) and self.k > 0:
            diff = float(it["s_c"]) - float(it["s_r"])
            target = torch.tensor(1.0 / (1.0 + torch.exp(torch.tensor(-self.k * diff))), dtype=torch.float32)
        else:
            target = torch.tensor(1.0, dtype=torch.float32)  # chosen preferred
        return {
            "chosen_input_ids": chosen_input_ids,
            "chosen_attention_mask": chosen_attention_mask,
            "rejected_input_ids": rejected_input_ids,
            "rejected_attention_mask": rejected_attention_mask,
            "targets": target,
        }


class BTCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        batch = {}
        for key in [
            "chosen_input_ids", "chosen_attention_mask",
            "rejected_input_ids", "rejected_attention_mask",
        ]:
            batch[key] = torch.stack([f[key] for f in features])
        batch["targets"] = torch.stack([f["targets"] for f in features])
        return batch


class BTTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else getattr(outputs, "loss", None)
        if loss is None:
            raise ValueError("Model must return a dict with 'loss' or an object with .loss")
        return (loss, outputs) if return_outputs else loss


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, BTTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print(f"Loading base encoder from {model_args.model_name_or_path}")
    base = AutoModel.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
        cache_dir=training_args.cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hidden_size = base.config.hidden_size
    model = BTRewardModel(base, hidden_size)

    print("Preparing dataset (pairwise)...")
    train_dataset = BTDataset(data_args.data_path, tokenizer, data_args.soft_label_k)

    trainer = BTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=BTCollator(tokenizer),
    )

    print("Starting BT training...")
    trainer.train()
    print("Saving model...")
    trainer.save_model(training_args.output_dir)
    torch.save(model.head.state_dict(), f"{training_args.output_dir}/bt_reward_head.bin")
    if trainer.is_world_process_zero():
        print("Done.")


if __name__ == "__main__":
    train()


