#!/usr/bin/env python3
"""
Cross-encoder Bradley–Terry training where (x, y1, y2) is ingested as a
single sequence. The model directly predicts Δs = s(y1) - s(y2) and is
trained with BCE-with-logits, i.e. a single-input BT experiment.

Example usage:

python scripts/train_bt_cross_encoder.py \
  --model_name_or_path /shared/ssd/models/Qwen2.5-3B \
  --data_path /shared/hdd/nuochen/JudgeLM/data/judgelm_100k.jsonl \
  --output_dir /shared/hdd/nuochen/JudgeLM/output/Qwen2.5-3B-bt-cross \
  --per_device_train_batch_size 16 \
  --learning_rate 5e-6 \
  --num_train_epochs 2 \
  --bf16 True
"""

import json
import random
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoModel, AutoTokenizer, Trainer


PAIR_TEMPLATE = (
    "<|im_start|>system\n"
    "You are a strict evaluator. Compare two candidate answers and decide "
    "which one better addresses the user.\n"
    "<|im_end|>\n"
    "<|im_start|>user\n{question}<|im_end|>\n"
    "<|im_start|>assistant\n"
    "Response A:\n{answer_a}\n\n"
    "Response B:\n{answer_b}\n\n"
    "Which response is better? Answer with only 'A' or 'B'.\n"
    "<|im_end|>"
)


def build_pair_prompt(question: str, answer_a: str, answer_b: str) -> str:
    return PAIR_TEMPLATE.format(question=question, answer_a=answer_a, answer_b=answer_b)


class CrossBTHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_state).squeeze(-1)


class CrossBTModel(nn.Module):
    def __init__(self, base: AutoModel):
        super().__init__()
        self.base = base
        self.head = CrossBTHead(base.config.hidden_size)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        seq_lens = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(hidden.size(0), device=hidden.device)
        eos_states = hidden[batch_indices, seq_lens]
        return eos_states

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pooled = self.encode(batch["input_ids"], batch["attention_mask"])
        logits = self.head(pooled)
        labels = batch["labels"]
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        return {"loss": loss, "logits": logits}


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="/shared/ssd/models/Qwen2.5-3B")


@dataclass
class DataArguments:
    data_path: str = field(metadata={"help": "Path to JudgeLM-style pairwise jsonl"})
    random_swap: bool = field(default=True, metadata={"help": "Randomly flip winner/loser order to debias position"})
    soft_label_k: float = field(
        default=0.5,
        metadata={"help": "k in sigmoid(k*(score_a-score_b)) for soft labels; set 0 to disable"},
    )


@dataclass
class CrossBTTrainingArguments(transformers.TrainingArguments):
    remove_unused_columns: bool = field(default=False)
    model_max_length: int = field(default=2048)
    cache_dir: Optional[str] = field(default=None)


class SingleInputPairDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        random_swap: bool,
        soft_label_k: float,
    ):
        self.tokenizer = tokenizer
        self.random_swap = random_swap
        self.soft_label_k = soft_label_k
        self.items = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ch = obj.get("chosen", [{}])[0]
                rj = obj.get("rejected", [{}])[0]
                q_win, a_win = str(ch.get("question", "")).strip(), str(ch.get("answer", "")).strip()
                q_lose, a_lose = str(rj.get("question", "")).strip(), str(rj.get("answer", "")).strip()
                if not q_win or not a_win or not q_lose or not a_lose:
                    continue
                if q_win != q_lose:
                    # fallback to chosen prompt if they differ
                    question = q_win or q_lose
                else:
                    question = q_win
                self.items.append(
                    {
                        "question": question,
                        "winner": a_win,
                        "loser": a_lose,
                        "winner_score": obj.get("chosen_score"),
                        "loser_score": obj.get("rejected_score"),
                    }
                )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.items[idx]
        winner_first = True
        if self.random_swap:
            winner_first = random.random() > 0.5

        winner_score = item.get("winner_score")
        loser_score = item.get("loser_score")

        if winner_first:
            ans_a, ans_b = item["winner"], item["loser"]
            score_a, score_b = winner_score, loser_score
            hard_label = 1.0
        else:
            ans_a, ans_b = item["loser"], item["winner"]
            score_a, score_b = loser_score, winner_score
            hard_label = 0.0

        label = hard_label
        if (
            self.soft_label_k > 0
            and isinstance(score_a, (int, float))
            and isinstance(score_b, (int, float))
        ):
            diff = float(score_a) - float(score_b)
            label = float(torch.sigmoid(torch.tensor(self.soft_label_k * diff)))

        prompt = build_pair_prompt(item["question"], ans_a, ans_b)
        tok = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": tok.input_ids.squeeze(0),
            "attention_mask": tok.attention_mask.squeeze(0),
            "labels": torch.tensor(label, dtype=torch.float32),
        }


class CrossBTCollator:
    def __call__(self, features):
        batch = {
            "input_ids": torch.stack([f["input_ids"] for f in features]),
            "attention_mask": torch.stack([f["attention_mask"] for f in features]),
            "labels": torch.stack([f["labels"] for f in features]),
        }
        return batch


class CrossBTTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(inputs)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, CrossBTTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print(f"Loading base model from {model_args.model_name_or_path}")
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

    model = CrossBTModel(base)

    print("Building dataset with single-input (x, y1, y2) sequences...")
    dataset = SingleInputPairDataset(
        data_args.data_path,
        tokenizer,
        data_args.random_swap,
        data_args.soft_label_k,
    )

    trainer = CrossBTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=CrossBTCollator(),
    )

    print("Starting cross-encoder BT training...")
    trainer.train()

    print("Saving final checkpoint...")
    trainer.save_model(training_args.output_dir)
    if trainer.is_world_process_zero():
        print("Done.")


if __name__ == "__main__":
    train()

