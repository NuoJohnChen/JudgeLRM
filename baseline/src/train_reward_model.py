#!/usr/bin/env python3
"""
Reward Model训练脚本
基于回归头直接学习分数预测
"""

import sys
from pathlib import Path
import torch
import transformers
from transformers import (
    AutoModel,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import json
import math
import torch.nn as nn
import torch.nn.functional as F

# Add root path to use absolute import
file = Path(__file__).resolve()
root = file.parents[2]
sys.path.append(str(root))

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")

@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "Do not remove unused columns (for custom collator)."}
    )

class RewardModel(nn.Module):
    """Reward Model: 在基础模型上添加一个回归头来预测分数"""
    def __init__(self, base_model, hidden_size):
        super().__init__()
        self.base_model = base_model
        
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, input_ids, attention_mask, labels=None):
        
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = hidden_states.shape[0]
        
        
        last_hidden_states = hidden_states[torch.arange(batch_size), sequence_lengths]
        
        
        rewards = self.reward_head(last_hidden_states).squeeze(-1)
        
        loss = None
        if labels is not None:
            
            loss = F.mse_loss(rewards, labels)
            
        return {
            'loss': loss,
            'rewards': rewards,
            'logits': rewards
        }

class RewardDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        self.tokenizer = tokenizer
        
        print("Loading data...")
        self.data = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    item = json.loads(line)
                    
                    # 1) {"question", "answer", "score"}
                    # 2) {"prompt", "score"} 其中prompt已是qwen模板
                    if not isinstance(item, dict):
                        continue
                    if 'prompt' in item and 'score' in item and isinstance(item['score'], (int, float)):
                        if not str(item['prompt']).strip():
                            continue
                        self.data.append({'prompt': item['prompt'], 'score': float(item['score'])})
                        continue
                    required_fields = ['question', 'answer', 'score']
                    if all(field in item for field in required_fields):
                        if not str(item['question']).strip() or not str(item['answer']).strip():
                            continue
                        if not isinstance(item['score'], (int, float)):
                            continue
                        self.data.append({'question': item['question'], 'answer': item['answer'], 'score': float(item['score'])})
                except json.JSONDecodeError:
                    continue
        
        print(f"Loaded {len(self.data)} valid examples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        item = self.data[i]
        
        
        if 'prompt' in item:
            prompt = item['prompt']
        else:
            prompt = (
                f"<|im_start|>user\n{item['question']}<|im_end|>\n"
                f"<|im_start|>assistant\n{item['answer']}<|im_end|>"
            )
        
        tokens = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": tokens.input_ids.squeeze(0),
            "attention_mask": tokens.attention_mask.squeeze(0),
            "labels": torch.tensor(item['score'], dtype=torch.float32)
        }

class RewardCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        if not features:
            raise ValueError("Empty batch")
        
        batch = {}
        for key in ["input_ids", "attention_mask", "labels"]:
            tensors = [f[key] for f in features]
            batch[key] = torch.stack(tensors)
        
        return batch

class RewardTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=labels
        )
        
        if isinstance(outputs, dict):
            loss = outputs.get('loss')
        else:
            loss = getattr(outputs, 'loss', None)
        if loss is None:
            raise ValueError("RewardModel forward must return a dict with 'loss' or an object with .loss")
        
        if return_outputs:
            return loss, outputs
        return loss

def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    print(f"Loading base model from {model_args.model_name_or_path}")
    base_model = AutoModel.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16
    )
    
    
    # Set RoPE scaling factor if needed (some models support it)
    orig_ctx_len = getattr(base_model.config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = math.ceil(training_args.model_max_length / orig_ctx_len)
        base_model.config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    
    hidden_size = base_model.config.hidden_size
    model = RewardModel(base_model, hidden_size)
    
    print("Preparing dataset...")
    train_dataset = RewardDataset(data_args.data_path, tokenizer)
    
    print("Initializing trainer...")
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=RewardCollator(tokenizer)
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving model...")
    trainer.save_model(training_args.output_dir)
    
    
    torch.save(model.reward_head.state_dict(), f"{training_args.output_dir}/reward_head.bin")
    
    if trainer.is_world_process_zero():
        print("Training completed!")
        print("Training args:")
        print(training_args)

if __name__ == "__main__":
    train()
