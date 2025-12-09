import sys
from pathlib import Path
import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import json
import math

# Add root path to use absolute import
file = Path(__file__).resolve()
root = file.parents[2]
sys.path.append(str(root))

from judgelm.conversation import SeparatorStyle
from judgelm.model.model_adapter import get_conversation_template

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

class DPODataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.conv = get_conversation_template("vicuna")
        
        #print("Loading data...")
        self.data = []
        self.valid_indices = []  # store indices of valid data
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    item = json.loads(line)
                    
                    if not isinstance(item, dict) or 'chosen' not in item or 'rejected' not in item:
                        #print(f"Skipping line {i}: Missing chosen or rejected")
                        continue
                    if not all(k in item['chosen'] for k in ['question', 'answer', 'score']):
                        #print(f"Skipping line {i}: Missing keys in chosen")
                        continue
                    if not all(k in item['rejected'] for k in ['question', 'answer', 'score']):
                        #print(f"Skipping line {i}: Missing keys in rejected")
                        continue
                    
                    
                    if not item['chosen']['question'] or not item['chosen']['answer']:
                        #print(f"Skipping line {i}: Empty chosen question or answer")
                        continue
                    if not item['rejected']['question'] or not item['rejected']['answer']:
                        #print(f"Skipping line {i}: Empty rejected question or answer")
                        continue
                    
                    self.data.append(item)
                    self.valid_indices.append(i)
                except json.JSONDecodeError as e:
                    #print(f"Error parsing line {i}: {str(e)}")
                    continue
        
        #print(f"\nLoaded {len(self.data)} valid examples out of {i+1} total lines")
        if len(self.data) > 0:
            print("\nFirst example structure:")
            print(json.dumps(self.data[0], indent=2))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i >= len(self.data):
            raise IndexError(f"Index {i} out of range")
            
        item = self.data[i]
        
        
        #print(f"\n=== Processing sample {i} (original index: {self.valid_indices[i]}) ===")
        #print(f"Item keys: {item.keys()}")
        #print(f"Chosen keys: {item['chosen'].keys()}")
        #print(f"Rejected keys: {item['rejected'].keys()}")
        
        
        if not isinstance(item, dict) or 'chosen' not in item or 'rejected' not in item:
            raise ValueError(f"Invalid data format at index {i}")
        
        chosen = item['chosen']
        rejected = item['rejected']
        
        # Validate the format of chosen and rejected
        for key in ['question', 'answer', 'score']:
            if key not in chosen or key not in rejected:
                raise ValueError(f"Missing key '{key}' in data at index {i}")
        
        # Format chosen example
        chosen_prompt = f"Human: {chosen['question']}\n\nAssistant: {chosen['answer']}"
        #print(f"\nChosen prompt: {chosen_prompt[:100]}...")  # only print first 100 chars
        chosen_tokens = self.tokenizer(
            chosen_prompt,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Format rejected example
        rejected_prompt = f"Human: {rejected['question']}\n\nAssistant: {rejected['answer']}"
        #print(f"Rejected prompt: {rejected_prompt[:100]}...")
        rejected_tokens = self.tokenizer(
            rejected_prompt,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        
        chosen_input_ids = chosen_tokens.input_ids.squeeze(0)
        chosen_attention_mask = chosen_tokens.attention_mask.squeeze(0)
        rejected_input_ids = rejected_tokens.input_ids.squeeze(0)
        rejected_attention_mask = rejected_tokens.attention_mask.squeeze(0)
        
        
        chosen_input_ids = chosen_input_ids.cpu()
        chosen_attention_mask = chosen_attention_mask.cpu()
        rejected_input_ids = rejected_input_ids.cpu()
        rejected_attention_mask = rejected_attention_mask.cpu()
        
        
        chosen_input_ids = chosen_input_ids.long()
        chosen_attention_mask = chosen_attention_mask.long()
        rejected_input_ids = rejected_input_ids.long()
        rejected_attention_mask = rejected_attention_mask.long()
        
        
        chosen_score = torch.tensor(float(chosen['score']), dtype=torch.float)
        rejected_score = torch.tensor(float(rejected['score']), dtype=torch.float)
        
        
        result = {
            "input_ids": chosen_input_ids,
            "attention_mask": chosen_attention_mask,
            "labels": chosen_input_ids.clone(),
            "rejected_input_ids": rejected_input_ids,
            "rejected_attention_mask": rejected_attention_mask,
            "rejected_labels": rejected_input_ids.clone(),
            "chosen_score": chosen_score,
            "rejected_score": rejected_score
        }
        
        
        #print(f"\nReturned keys: {result.keys()}")
        # for k, v in result.items():
        #     if isinstance(v, torch.Tensor):
        #         print(f"{k}: shape={v.shape}, dtype={v.dtype}")
        
        return result

class DPOCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        if not features:
            raise ValueError("Empty batch")
            
        
        #print("\n=== DPOCollator Debug Info ===")
        #print(f"Number of features in batch: {len(features)}")
        #print(f"First feature keys: {features[0].keys()}")
        
        
        required_keys = [
            "input_ids", "attention_mask", "labels",
            "rejected_input_ids", "rejected_attention_mask", "rejected_labels",
            "chosen_score", "rejected_score"
        ]
        
        
        valid_features = []
        for i, feature in enumerate(features):
            missing_keys = [key for key in required_keys if key not in feature]
            if missing_keys:
                #print(f"Skipping feature {i} due to missing keys: {missing_keys}")
                continue
            valid_features.append(feature)
        
        if not valid_features:
            raise ValueError("No valid features in batch")
        
        try:
            
            batch = {}
            for key in required_keys:
                try:
                    tensors = [f[key] for f in valid_features]
                    #print(f"Stacking {key} with {len(tensors)} tensors")
                    #print(f"First tensor shape: {tensors[0].shape}")
                    
                    # For scalar tensors (e.g., scores), add a dimension with unsqueeze(0)
                    if tensors[0].dim() == 0:
                        tensors = [t.unsqueeze(0) for t in tensors]
                    
                    # For sequence tensors (e.g., input_ids), ensure they are 2D
                    if tensors[0].dim() == 1:
                        tensors = [t.unsqueeze(0) for t in tensors]
                    
                    # Stack tensors to ensure final shape is [batch_size, seq_len]
                    batch[key] = torch.stack(tensors).squeeze(1)  # remove extra dimension
                    #print(f"Successfully stacked {key} with shape: {batch[key].shape}")
                except Exception as e:
                    print(f"Error stacking {key}: {str(e)}")
                    print(f"Available keys in first feature: {valid_features[0].keys()}")
                    raise
            
            return batch
        except Exception as e:
            #print("\nError in collator:")
            #print(f"Error type: {type(e)}")
            #print(f"Error message: {str(e)}")
            #print("\nFirst feature content:")
            # for k, v in valid_features[0].items():
            #     if isinstance(v, torch.Tensor):
            #         print(f"{k}: shape={v.shape}, dtype={v.dtype}")
            #     else:
            #         print(f"{k}: type={type(v)}")
            raise

class DPOTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Get model outputs for chosen and rejected sequences
        chosen_outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"]
        )
        rejected_outputs = model(
            input_ids=inputs["rejected_input_ids"],
            attention_mask=inputs["rejected_attention_mask"],
            labels=inputs["rejected_labels"]
        )
        
        # Calculate DPO loss
        chosen_loss = chosen_outputs.loss
        rejected_loss = rejected_outputs.loss
        
        # DPO loss is the difference between chosen and rejected losses
        loss = chosen_loss - rejected_loss
        
        if return_outputs:
            return loss, {"chosen_outputs": chosen_outputs, "rejected_outputs": rejected_outputs}
        return loss

def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    #print(f"Loading model from {model_args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16
    )
    model.config.use_cache = False
    
    # New: enable static graph
    if hasattr(model, '_set_static_graph'):
        #print('Setting static graph for DDP+gradient_checkpointing...')
        model._set_static_graph()
    
    # Set RoPE scaling factor if needed
    orig_ctx_len = getattr(model.config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = math.ceil(training_args.model_max_length / orig_ctx_len)
        model.config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    
    #print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    #print("Preparing dataset...")
    train_dataset = DPODataset(data_args.data_path, tokenizer)
    
    
    #print("Dataset size:", len(train_dataset))
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        #print("Sample keys:", sample.keys())
        #print("Sample shapes:")
        # for k, v in sample.items():
        #     if isinstance(v, torch.Tensor):
        #         print(f"{k}: {v.shape}")
    
    #print("Initializing trainer...")
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=DPOCollator(tokenizer)
    )
    
    #print("Starting training...")
    trainer.train()
    
    #print("Saving model...")
    trainer.save_model(training_args.output_dir)
    
    if trainer.is_world_process_zero():
        print("Training completed!")
        print("Training args:")
        print(training_args)

if __name__ == "__main__":
    train()