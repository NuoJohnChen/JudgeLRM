#!/usr/bin/env python3
"""
SFT training with custom target format (sft_think):

Output format to learn:
### Response:
<think>
{explanation}
</think>
<answer>{score1}</answer><answer>{score2}</answer>

Input prompt includes System + Question + two answers (and optional reference) and ends with "### Response:".
Only the target block is supervised (masked CE on assistant part only).
"""

import json
import argparse
import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer, TrainingArguments as HFTrainingArguments
from transformers.trainer_pt_utils import LabelSmoother


IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="/path/to/models/Qwen2.5-3B-Instruct")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to train jsonl (JudgeLM-100K-think style)"})
    swap_aug_ratio: float = field(default=-1.0)
    ref_drop_ratio: float = field(default=-1.0)


# Use HF's built-in TrainingArguments directly to avoid type resolution issues


def _tokenize(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    tokenized_list = [
        tokenizer(
            s,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=tokenizer.model_max_length,
        )
        for s in strings
    ]
    input_ids = [t.input_ids[0] for t in tokenized_list]
    input_lens = [t.input_ids.ne(tokenizer.pad_token_id).sum().item() for t in tokenized_list]
    return {"input_ids": input_ids, "input_lens": input_lens}


def build_system_prompt(use_reference: bool) -> str:
    base = (
        "You are a careful evaluator. Given a question and two assistant answers, "
        "first write your reasoning enclosed in <think>...</think>, then output two scores "
        "each in an <answer> tag (first for Assistant 1, second for Assistant 2). Scores are integers 1-10."
    )
    if use_reference:
        base += " You may also use the provided reference to support your reasoning."
    return base


def build_source_block(question: str, a1: str, a2: str, use_reference: bool, reference: Optional[str]) -> str:
    sys_prompt = build_system_prompt(use_reference)
    body = [
        sys_prompt,
        "[Question]",
        question,
        "",
        "[The Start of Assistant 1's Answer]",
        a1,
        "[The End of Assistant 1's Answer]",
        "",
        "[The Start of Assistant 2's Answer]",
        a2,
        "[The End of Assistant 2's Answer]",
    ]
    if use_reference and reference:
        body += ["", "[Reference]", reference]
    body += ["", "### Response:"]
    return "\n".join(body)


def parse_text_into_scores_and_explanation(text: str) -> (str, int, int):
    # text's first line like: "8 6", rest is explanation
    lines = text.splitlines()
    if not lines:
        return "", 5, 5
    first = lines[0].strip()
    rest = "\n".join(lines[1:]).strip()
    s1, s2 = 5, 5
    try:
        parts = first.split()
        if len(parts) >= 2:
            s1 = int(float(parts[0]))
            s2 = int(float(parts[1]))
    except Exception:
        pass
    return rest, s1, s2


class SFTThinkDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, swap_aug_ratio: float, ref_drop_ratio: float):
        self.tokenizer = tokenizer
        self.swap_aug_ratio = swap_aug_ratio
        self.ref_drop_ratio = ref_drop_ratio

        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get('text') == 'error':
                    continue
                self.data.append(obj)

    def __len__(self):
        return len(self.data)

    def _maybe_swap(self, d: Dict) -> Dict:
        if self.swap_aug_ratio >= -0.5 and np.random.rand() < self.swap_aug_ratio:
            d = dict(d)
            d['answer1_body'], d['answer2_body'] = d['answer2_body'], d['answer1_body']
            if 'score' in d and isinstance(d['score'], list) and len(d['score']) >= 2:
                d['score'] = d['score'][::-1]
            if 'text' in d and isinstance(d['text'], str):
                # swap first two numbers in header line if present
                t = d['text']
                nl = t.find('\n')
                if nl != -1:
                    hdr = t[:nl]
                    rest = t[nl+1:]
                    parts = hdr.split()
                    if len(parts) >= 2:
                        hdr = f"{parts[1]} {parts[0]}" + (" " + " ".join(parts[2:]) if len(parts) > 2 else "")
                        d['text'] = hdr + "\n" + rest
            return d
        return d

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        d0 = self.data[i]
        use_reference = False
        d = dict(d0)
        if self.ref_drop_ratio >= -0.5 and np.random.rand() < self.ref_drop_ratio:
            # use reference-enhanced version if available
            if 'text_w_reference' in d and 'reference' in d:
                d = dict(d)
                d['text'] = d.get('text_w_reference', d.get('text', ''))
                use_reference = True
        d = self._maybe_swap(d)

        question = d.get('question_body', '').strip()
        a1 = d.get('answer1_body', '').strip()
        a2 = d.get('answer2_body', '').strip()
        reference = d.get('reference', {}).get('text') if isinstance(d.get('reference'), dict) else None
        if not question or not a1 or not a2:
            question = question or "(empty)"

        source = build_source_block(question, a1, a2, use_reference, reference)
        # target
        raw_text = d.get('text', '').strip()
        explanation, s1, s2 = parse_text_into_scores_and_explanation(raw_text)
        target = (
            "<think>\n" + explanation + "\n</think>\n" +
            f"<answer>{s1}</answer><answer>{s2}</answer>"
        )
        example = source + "\n" + target + ("" if self.tokenizer.eos_token is None else self.tokenizer.eos_token)

        # tokenize
        example_tok = _tokenize([example], self.tokenizer)
        source_tok = _tokenize([source], self.tokenizer)
        input_ids = example_tok["input_ids"][0]
        label = input_ids.clone()
        source_len = source_tok["input_lens"][0]
        label[:source_len] = IGNORE_TOKEN_ID
        return dict(input_ids=input_ids, labels=label)


@dataclass
class DataCollator:
    tokenizer: transformers.PreTrainedTokenizer
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [inst["input_ids"] for inst in instances]
        labels = [inst["labels"] for inst in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_TOKEN_ID)
        return dict(input_ids=input_ids, labels=labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id))


def train():
    ap = argparse.ArgumentParser()
    # model/data args
    ap.add_argument('--model_name_or_path', type=str, required=True)
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--swap_aug_ratio', type=float, default=-1.0)
    ap.add_argument('--ref_drop_ratio', type=float, default=-1.0)
    # training args (subset)
    ap.add_argument('--output_dir', type=str, required=True)
    ap.add_argument('--num_train_epochs', type=float, default=1.0)
    ap.add_argument('--per_device_train_batch_size', type=int, default=1)
    ap.add_argument('--gradient_accumulation_steps', type=int, default=1)
    ap.add_argument('--learning_rate', type=float, default=2e-5)
    ap.add_argument('--weight_decay', type=float, default=0.0)
    ap.add_argument('--warmup_ratio', type=float, default=0.0)
    ap.add_argument('--lr_scheduler_type', type=str, default='cosine')
    ap.add_argument('--save_strategy', type=str, default='steps')
    ap.add_argument('--save_steps', type=int, default=200)
    ap.add_argument('--save_total_limit', type=int, default=5)
    ap.add_argument('--logging_steps', type=int, default=10)
    ap.add_argument('--bf16', type=str, default='False')
    ap.add_argument('--tf32', type=str, default='False')
    ap.add_argument('--model_max_length', type=int, default=2048)
    ap.add_argument('--report_to', type=str, default='none')
    ap.add_argument('--cache_dir', type=str, default=None)
    ap.add_argument('--run_name', type=str, default='sft_think')
    args = ap.parse_args()

    # normalize booleans
    def to_bool(x):
        if isinstance(x, bool):
            return x
        return str(x).lower() in ('1','true','yes','y')

    bf16 = to_bool(args.bf16)
    tf32 = to_bool(args.tf32)

    model_args = ModelArguments(model_name_or_path=args.model_name_or_path)
    data_args = DataArguments(data_path=args.data_path, swap_aug_ratio=args.swap_aug_ratio, ref_drop_ratio=args.ref_drop_ratio)
    training_args = HFTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        bf16=bf16,
        tf32=tf32,
        report_to=args.report_to,
        optim='adamw_torch',
    )
    # attach extra
    training_args.cache_dir = args.cache_dir
    training_args.model_max_length = args.model_max_length

    print(f"Load model: {model_args.model_name_or_path}")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    model.config.use_cache = False

    # rope scaling if needed
    orig_ctx_len = getattr(model.config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        factor = math.ceil(training_args.model_max_length / orig_ctx_len)
        model.config.rope_scaling = {"type": "linear", "factor": factor}

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = model.to(torch.bfloat16) if training_args.bf16 else model.to(torch.float16)

    dataset = SFTThinkDataset(data_path=data_args.data_path, tokenizer=tokenizer,
                              swap_aug_ratio=data_args.swap_aug_ratio, ref_drop_ratio=data_args.ref_drop_ratio)
    collator = DataCollator(tokenizer=tokenizer)

    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, train_dataset=dataset, data_collator=collator)
    trainer.train()
    model.config.use_cache = True
    trainer.save_model(training_args.output_dir)
    if trainer.is_world_process_zero():
        print("Training done.")


if __name__ == "__main__":
    train()


