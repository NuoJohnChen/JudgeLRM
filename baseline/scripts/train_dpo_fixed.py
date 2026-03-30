#!/usr/bin/env python3
"""
修复后的DPO训练脚本
实现正确的DPO loss计算
"""

import sys
from pathlib import Path
import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import json
import math
import torch.nn.functional as F

# Add root path to use absolute import
file = Path(__file__).resolve()
root = file.parents[2]
sys.path.append(str(root))

# 移除JudgeLM依赖，使用通用的对话格式

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
    beta: float = field(
        default=0.1,
        metadata={"help": "DPO temperature parameter."}
    )

class DPODataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        self.tokenizer = tokenizer
        # 使用简单的对话格式，不依赖特定模板
        
        print("Loading data...")
        self.data = []
        self.valid_indices = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    item = json.loads(line)
                    # 验证数据格式
                    if not isinstance(item, dict) or 'chosen' not in item or 'rejected' not in item:
                        continue
                    if not all(k in item['chosen'][0] for k in ['question', 'answer']):
                        continue
                    if not all(k in item['rejected'][0] for k in ['question', 'answer']):
                        continue
                    
                    # 验证数据内容
                    if not item['chosen'][0]['question'] or not item['chosen'][0]['answer']:
                        continue
                    if not item['rejected'][0]['question'] or not item['rejected'][0]['answer']:
                        continue
                    
                    self.data.append(item)
                    self.valid_indices.append(i)
                except json.JSONDecodeError:
                    continue
        
        print(f"Loaded {len(self.data)} valid examples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        item = self.data[i]
        chosen = item['chosen'][0]
        rejected = item['rejected'][0]
        
        # 使用Qwen的对话格式
        # Format chosen example
        chosen_prompt = f"<|im_start|>user\n{chosen['question']}<|im_end|>\n<|im_start|>assistant\n{chosen['answer']}<|im_end|>"
        chosen_tokens = self.tokenizer(
            chosen_prompt,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Format rejected example
        rejected_prompt = f"<|im_start|>user\n{rejected['question']}<|im_end|>\n<|im_start|>assistant\n{rejected['answer']}<|im_end|>"
        rejected_tokens = self.tokenizer(
            rejected_prompt,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # 确保所有张量都是一维的
        chosen_input_ids = chosen_tokens.input_ids.squeeze(0)
        chosen_attention_mask = chosen_tokens.attention_mask.squeeze(0)
        rejected_input_ids = rejected_tokens.input_ids.squeeze(0)
        rejected_attention_mask = rejected_tokens.attention_mask.squeeze(0)
        
        return {
            "input_ids": chosen_input_ids,
            "attention_mask": chosen_attention_mask,
            "rejected_input_ids": rejected_input_ids,
            "rejected_attention_mask": rejected_attention_mask,
        }

class DPOCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        if not features:
            raise ValueError("Empty batch")
        
        batch = {}
        for key in ["input_ids", "attention_mask", "rejected_input_ids", "rejected_attention_mask"]:
            tensors = [f[key] for f in features]
            batch[key] = torch.stack(tensors)
        
        return batch

def compute_logps(model, input_ids, attention_mask):
    """计算模型对序列的对数概率"""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    
    # 计算每个token的对数概率
    log_probs = F.log_softmax(logits, dim=-1)
    
    # 计算序列的对数概率
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    shift_attention_mask = attention_mask[..., 1:].contiguous()
    
    # 计算每个位置的对数概率
    log_probs = F.log_softmax(shift_logits, dim=-1)
    
    # 选择对应token的对数概率
    selected_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    
    # 只计算非padding位置的概率
    selected_log_probs = selected_log_probs * shift_attention_mask
    
    # 计算序列的总对数概率
    sequence_log_probs = selected_log_probs.sum(dim=-1)
    
    return sequence_log_probs

def dpo_loss(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta=0.1):
    """计算DPO loss"""
    # 计算策略模型和参考模型的log ratio
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    
    # 计算DPO loss
    logits = beta * (pi_logratios - ref_logratios)
    
    # 使用log sigmoid来避免数值不稳定
    losses = -F.logsigmoid(logits)
    
    return losses.mean()

class DPOTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reference_model = None
        self.beta = getattr(self.args, 'beta', 0.1)
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 获取原始模型（如果是DDP包装的）
        if hasattr(model, 'module'):
            original_model = model.module
        else:
            original_model = model
            
        # 如果还没有加载参考模型，使用当前模型的副本作为参考
        if self.reference_model is None:
            # 创建参考模型的副本，并冻结参数
            self.reference_model = AutoModelForCausalLM.from_pretrained(
                original_model.config._name_or_path,
                torch_dtype=original_model.dtype,
                device_map={"": original_model.device}  # 参考模型放在与策略模型相同的设备上
            )
            # 冻结参考模型的参数
            for param in self.reference_model.parameters():
                param.requires_grad = False
        
        # 获取chosen和rejected序列
        chosen_input_ids = inputs["input_ids"]
        chosen_attention_mask = inputs["attention_mask"]
        rejected_input_ids = inputs["rejected_input_ids"]
        rejected_attention_mask = inputs["rejected_attention_mask"]
            
        # 计算策略模型的对数概率（需要梯度）
        policy_chosen_logps = compute_logps(original_model, chosen_input_ids, chosen_attention_mask)
        policy_rejected_logps = compute_logps(original_model, rejected_input_ids, rejected_attention_mask)
        
        # 计算参考模型的对数概率（不需要梯度）
        with torch.no_grad():
            reference_chosen_logps = compute_logps(self.reference_model, chosen_input_ids, chosen_attention_mask)
            reference_rejected_logps = compute_logps(self.reference_model, rejected_input_ids, rejected_attention_mask)
        
        # 计算DPO loss
        loss = dpo_loss(
            policy_chosen_logps, policy_rejected_logps,
            reference_chosen_logps, reference_rejected_logps,
            beta=self.beta
        )
        
        if return_outputs:
            return loss, {"policy_chosen_logps": policy_chosen_logps, "policy_rejected_logps": policy_rejected_logps}
        return loss

def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    print(f"Loading model from {model_args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16
    )
    model.config.use_cache = False
    
    # Set RoPE scaling factor if needed
    orig_ctx_len = getattr(model.config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = math.ceil(training_args.model_max_length / orig_ctx_len)
        model.config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    
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
    
    print("Preparing dataset...")
    train_dataset = DPODataset(data_args.data_path, tokenizer)
    
    print("Initializing trainer...")
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=DPOCollator(tokenizer)
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving model...")
    trainer.save_model(training_args.output_dir)
    
    if trainer.is_world_process_zero():
        print("Training completed!")
        print("Training args:")
        print(training_args)

if __name__ == "__main__":
    train()
