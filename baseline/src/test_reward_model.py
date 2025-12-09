#!/usr/bin/env python3
"""
Reward Model test script
Evaluate a trained reward model on PandaLM data
"""

import sys
from pathlib import Path
import torch
import json
import argparse
from transformers import AutoTokenizer, AutoModel
from safetensors.torch import load_file as load_safetensors
import torch.nn as nn
import torch.nn.functional as F

# Add root path to use absolute import
file = Path(__file__).resolve()
root = file.parents[2]
sys.path.append(str(root))

class RewardModel(nn.Module):
    """Reward Model: add a regression head on top of the base model to predict scores"""
    def __init__(self, base_model, hidden_size):
        super().__init__()
        self.base_model = base_model
        # Use a deeper regression head
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, input_ids, attention_mask):
        # Get outputs from base model
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Use hidden state of the last non-padding token
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = hidden_states.shape[0]
        
        # Get hidden state of the last valid token in each sequence
        last_hidden_states = hidden_states[torch.arange(batch_size), sequence_lengths]
        # Align dtype with reward_head weights to avoid BF16/FP32 mismatch
        target_dtype = self.reward_head[0].weight.dtype
        if last_hidden_states.dtype != target_dtype:
            last_hidden_states = last_hidden_states.to(target_dtype)
        
        # Predict score via reward head
        rewards = self.reward_head(last_hidden_states).squeeze(-1)
        
        return rewards

class RewardModelTester:
    def __init__(self, model_path, device="cuda", base_model_path=None):
        self.device = device
        print(f"Loading model from {model_path}")
        
        # Choose base and tokenizer source (checkpoint may lack config.json)
        base_path = base_model_path if base_model_path else model_path
        
        # Load base model (AutoModel for last_hidden_state)
        self.base_model = AutoModel.from_pretrained(
            base_path,
            torch_dtype=torch.bfloat16,
            device_map=None
        ).to(self.device)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_path,
            padding_side="right",
            use_fast=False,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Build reward model
        hidden_size = self.base_model.config.hidden_size
        self.model = RewardModel(self.base_model, hidden_size)
        
        # Load reward head weights: support full model dir or Trainer checkpoint dir
        reward_head_path = Path(model_path) / "reward_head.bin"
        if reward_head_path.exists():
            print(f"Loading reward head from {reward_head_path}")
            state_dict = torch.load(str(reward_head_path), map_location=device)
            self.model.reward_head.load_state_dict(state_dict, strict=False)
        else:
            # Try extracting reward_head.* from checkpoint files
            ckpt_sft = Path(model_path) / "model.safetensors"
            ckpt_pt = Path(model_path) / "pytorch_model.bin"
            loaded = False
            if ckpt_sft.exists():
                print(f"Extracting reward head from {ckpt_sft}")
                sd = load_safetensors(str(ckpt_sft), device='cpu')
                head_sd = {k.replace('reward_head.', ''): v for k, v in sd.items() if k.startswith('reward_head.')}
                if head_sd:
                    self.model.reward_head.load_state_dict(head_sd, strict=False)
                    loaded = True
            if (not loaded) and ckpt_pt.exists():
                print(f"Extracting reward head from {ckpt_pt}")
                sd = torch.load(str(ckpt_pt), map_location='cpu')
                head_sd = {k.replace('reward_head.', ''): v for k, v in sd.items() if k.startswith('reward_head.')}
                if head_sd:
                    self.model.reward_head.load_state_dict(head_sd, strict=False)
                    loaded = True
            if not loaded:
                print("Warning: No reward head weights found in checkpoint; using random initialization")
        
        self.model.to(device)
        self.model.eval()
    
    def build_prompt(self, instruction, input_text, response):
        """Build Qwen 2.5 chat template prompt"""
        if input_text:
            question = instruction + '\n' + input_text
        else:
            question = instruction
        
        prompt = (
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n{response}<|im_end|>"
        )
        return prompt
    
    def calculate_reward(self, instruction, input_text, response):
        """Compute reward score for a single answer"""
        prompt = self.build_prompt(instruction, input_text, response)
        
        # Tokenize
        tokens = self.tokenizer(
            prompt,
            truncation=True,
            max_length=2048,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = tokens.input_ids.to(self.device)
        attention_mask = tokens.attention_mask.to(self.device)
        
        # Compute reward
        with torch.no_grad():
            reward = self.model(input_ids, attention_mask)
        
        return reward.item()
    
    def evaluate_pandalm_data(self, data_path, output_path, num_samples=None):
        """Evaluate PandaLM data"""
        print(f"Loading PandaLM data from {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if num_samples:
            data = data[:num_samples]
        
        results = []
        correct_predictions = 0
        total_predictions = 0
        y_true_all, y_pred_all = [], []
        # Over-total metric: treat unlabeled samples as incorrect
        y_true_total, y_pred_total = [], []
        
        print(f"Evaluating {len(data)} samples...")
        
        for i, item in enumerate(data):
            if i % 100 == 0:
                print(f"Processing {i}/{len(data)}")
            
            instruction = item.get('instruction', '')
            input_text = item.get('input', '')
            # Support field names: output1/output2 or response1/response2
            response1 = item.get('output1') or item.get('response1') or ''
            response2 = item.get('output2') or item.get('response2') or ''
            
            if not response1 or not response2:
                continue
            
            # Compute reward for both responses
            reward1 = self.calculate_reward(instruction, input_text, response1)
            reward2 = self.calculate_reward(instruction, input_text, response2)
            
            
            predicted_better = 1 if reward1 > reward2 else 2
            # Support multiple label fields (common in PandaLM: annotator1/2/3 or label)
            actual_better = item.get('better')
            if actual_better is None:
                actual_better = item.get('label')
            if actual_better is None:
                actual_better = item.get('winner')
            
            if actual_better is None:
                votes = []
                for k in ('annotator1', 'annotator2', 'annotator3'):
                    v = item.get(k)
                    if isinstance(v, (int, float)) and int(v) in (1, 2):
                        votes.append(int(v))
                if len(votes) >= 2:
                    ones = sum(1 for v in votes if v == 1)
                    twos = sum(1 for v in votes if v == 2)
                    if ones > twos:
                        actual_better = 1
                    elif twos > ones:
                        actual_better = 2
                    else:
                        actual_better = None  # ignore ties
            
            if isinstance(actual_better, str) and actual_better.isdigit():
                actual_better = int(actual_better)
            
            if actual_better in (1, 2):
                if predicted_better == actual_better:
                    correct_predictions += 1
                total_predictions += 1
                y_true_all.append(actual_better)
                y_pred_all.append(predicted_better)
                y_true_total.append(actual_better)
                y_pred_total.append(predicted_better)
            else:
                
                opp = 1 if predicted_better == 2 else 2
                y_true_total.append(opp)
                y_pred_total.append(predicted_better)
            
            correct_flag = None
            if actual_better in (1, 2):
                correct_flag = (predicted_better == actual_better)

            result = {
                'instruction': instruction,
                'input': input_text,
                'response1': response1,
                'response2': response2,
                'reward1': reward1,
                'reward2': reward2,
                'predicted_better': predicted_better,
                'actual_better': actual_better,
                'correct': correct_flag
            }
            results.append(result)
        
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        print(f"\nEvaluation Results:")
        print(f"Total samples: {len(data)}")
        print(f"Valid predictions: {total_predictions}")
        print(f"Correct predictions: {correct_predictions}")
        print(f"Accuracy (valid): {accuracy:.4f}")
        # Print precision/recall/f1 (valid and over-total)
        try:
            from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
            if total_predictions > 0:
                precision = precision_score(y_true_all, y_pred_all, average='macro', zero_division=0)
                recall = recall_score(y_true_all, y_pred_all, average='macro', zero_division=0)
                f1 = f1_score(y_true_all, y_pred_all, average='macro', zero_division=0)
                print(f"Precision (valid): {precision:.4f}")
                print(f"Recall (valid): {recall:.4f}")
                print(f"F1 (valid): {f1:.4f}")
            # over total
            acc_over_total = accuracy_score(y_true_total, y_pred_total) if len(y_true_total) > 0 else 0.0
            prec_over_total = precision_score(y_true_total, y_pred_total, average='macro', zero_division=0) if len(y_true_total) > 0 else 0.0
            rec_over_total = recall_score(y_true_total, y_pred_total, average='macro', zero_division=0) if len(y_true_total) > 0 else 0.0
            f1_over_total = f1_score(y_true_total, y_pred_total, average='macro', zero_division=0) if len(y_true_total) > 0 else 0.0
            print(f"Accuracy (over total {len(data)}): {acc_over_total:.4f}")
            print(f"Precision (over total): {prec_over_total:.4f}")
            print(f"Recall (over total): {rec_over_total:.4f}")
            print(f"F1 (over total): {f1_over_total:.4f}")
        except Exception:
            pass
        
        
        with open(output_path, 'w', encoding='utf-8') as f:
            
            payload = {
                'accuracy_valid': accuracy,
                'total_samples': len(data),
                'valid_predictions': total_predictions,
                'correct_predictions': correct_predictions,
                'results': results
            }
            try:
                from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
                if total_predictions > 0:
                    payload['precision'] = precision_score(y_true_all, y_pred_all, average='macro', zero_division=0)
                    payload['recall'] = recall_score(y_true_all, y_pred_all, average='macro', zero_division=0)
                    payload['f1'] = f1_score(y_true_all, y_pred_all, average='macro', zero_division=0)
                payload['accuracy_over_total'] = accuracy_score(y_true_total, y_pred_total) if len(y_true_total) > 0 else 0.0
                payload['precision_over_total'] = precision_score(y_true_total, y_pred_total, average='macro', zero_division=0) if len(y_true_total) > 0 else 0.0
                payload['recall_over_total'] = recall_score(y_true_total, y_pred_total, average='macro', zero_division=0) if len(y_true_total) > 0 else 0.0
                payload['f1_over_total'] = f1_score(y_true_total, y_pred_total, average='macro', zero_division=0) if len(y_true_total) > 0 else 0.0
            except Exception:
                pass
            json.dump(payload, f, ensure_ascii=False, indent=2)
        
        print(f"Results saved to {output_path}")
        return accuracy

def main():
    parser = argparse.ArgumentParser(description="Test Reward Model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained reward model or checkpoint dir")
    parser.add_argument("--base_model_path", type=str, default=None, help="Optional: base model to load tokenizer/backbone (for checkpoints)")
    parser.add_argument("--data_path", type=str, default="/user/Logic-RL/testset-v1_update.json", help="Path to PandaLM test data")
    parser.add_argument("--output_path", type=str, default="/user/JudgeLM/output/reward_model_results.json", help="Path to save results")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to evaluate (None for all)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    
    tester = RewardModelTester(args.model_path, args.device, base_model_path=args.base_model_path)
    
    
    accuracy = tester.evaluate_pandalm_data(args.data_path, args.output_path, args.num_samples)
    
    print(f"\nFinal Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
