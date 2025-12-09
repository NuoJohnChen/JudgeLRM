#!/usr/bin/env python3
"""
Test script for Bradley–Terry pairwise reward model checkpoints.

Loads a base AutoModel backbone and a BT reward head from a checkpoint directory
then evaluates on PandaLM-style pair data to produce Accuracy/Precision/Recall/F1.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file as load_safetensors
from transformers import AutoModel, AutoTokenizer
from typing import Dict


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
        return self.proj(hidden_state).squeeze(-1)


class BTRewardTester:
    def __init__(self, ckpt_dir: str, base_model_path: str, device: str = "cuda"):
        self.device = device
        print(f"Loading base model from {base_model_path}")
        self.base = AutoModel.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map=None,
        ).to(self.device)
        self.hidden_size = self.base.config.hidden_size
        self.head = BTRewardHead(self.hidden_size).to(self.device)

        print(f"Loading tokenizer from {base_model_path}")
        self.tok = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        # Load head weights from checkpoint dir
        ckpt = Path(ckpt_dir)
        head_bin = ckpt / 'bt_reward_head.bin'
        loaded = False
        if head_bin.exists():
            print(f"Loading BT head from {head_bin}")
            sd = torch.load(str(head_bin), map_location='cpu')
            self.head.load_state_dict(sd, strict=False)
            loaded = True
        else:
            # Try to extract from full model weights
            sft_path = ckpt / 'model.safetensors'
            pt_path = ckpt / 'pytorch_model.bin'
            if sft_path.exists():
                print(f"Extracting BT head from {sft_path}")
                sd = load_safetensors(str(sft_path), device='cpu')
                head_sd = {k.replace('head.', ''): v for k, v in sd.items() if k.startswith('head.')}
                if head_sd:
                    self.head.load_state_dict(head_sd, strict=False)
                    loaded = True
            if (not loaded) and pt_path.exists():
                print(f"Extracting BT head from {pt_path}")
                sd = torch.load(str(pt_path), map_location='cpu')
                head_sd = {k.replace('head.', ''): v for k, v in sd.items() if k.startswith('head.')}
                if head_sd:
                    self.head.load_state_dict(head_sd, strict=False)
                    loaded = True
        if not loaded:
            print("Warning: BT head weights not found; using random init (metrics may be meaningless)")

        self.base.eval()
        self.head.eval()

    def encode(self, prompt: str) -> torch.Tensor:
        toks = self.tok(prompt, truncation=True, max_length=2048, padding='max_length', return_tensors='pt')
        input_ids = toks.input_ids.to(self.device)
        attn = toks.attention_mask.to(self.device)
        with torch.no_grad():
            outs = self.base(input_ids=input_ids, attention_mask=attn)
        last_hidden = outs.last_hidden_state
        seq_lens = attn.sum(dim=1) - 1
        batch_indices = torch.arange(last_hidden.size(0), device=last_hidden.device)
        last_states = last_hidden[batch_indices, seq_lens]
        # dtype align with head
        target_dtype = self.head.proj[0].weight.dtype
        if last_states.dtype != target_dtype:
            last_states = last_states.to(target_dtype)
        return last_states

    def score(self, question: str, answer: str) -> float:
        prompt = build_qwen_prompt(question, answer)
        feats = self.encode(prompt)
        with torch.no_grad():
            r = self.head(feats)
        return float(r.item())

    def evaluate(self, data_path: str, num_samples: int | None = None) -> Dict:
        print(f"Loading eval data from {data_path}")
        data = json.load(open(data_path, 'r', encoding='utf-8'))
        if num_samples:
            data = data[:num_samples]
        y_true, y_pred = [], []
        y_true_total, y_pred_total = [], []
        total = len(data)
        valid = 0
        results = []
        for i, it in enumerate(data):
            if i % 100 == 0:
                print(f"Processing {i}/{len(data)}")
            instr = it.get('instruction', '')
            inp = it.get('input', '')
            q = instr + ("\n" + inp if inp else "")
            a1 = it.get('output1') or it.get('response1') or ''
            a2 = it.get('output2') or it.get('response2') or ''
            if not a1 or not a2:
                continue
            r1 = self.score(q, a1)
            r2 = self.score(q, a2)
            pred = 1 if r1 > r2 else 2
            # majority vote from annotators if available
            gt = it.get('better') or it.get('label') or it.get('winner')
            if gt is None:
                votes = []
                for k in ('annotator1', 'annotator2', 'annotator3'):
                    v = it.get(k)
                    if isinstance(v, (int, float)) and int(v) in (1, 2):
                        votes.append(int(v))
                if len(votes) >= 2:
                    ones = sum(1 for v in votes if v == 1)
                    twos = sum(1 for v in votes if v == 2)
                    if ones > twos:
                        gt = 1
                    elif twos > ones:
                        gt = 2
            if isinstance(gt, str) and gt.isdigit():
                gt = int(gt)
            if gt in (1, 2):
                y_true.append(gt)
                y_pred.append(pred)
                valid += 1
                y_true_total.append(gt)
                y_pred_total.append(pred)
                is_correct = (pred == gt)
            else:
                
                opp = 1 if pred == 2 else 2
                y_true_total.append(opp)
                y_pred_total.append(pred)
                is_correct = None

            results.append({
                'idx': it.get('idx'),
                'instruction': instr,
                'input': inp,
                'response1': a1,
                'response2': a2,
                'reward1': r1,
                'reward2': r2,
                'predicted_better': pred,
                'actual_better': gt,
                'correct': is_correct,
            })
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        acc = accuracy_score(y_true, y_pred) if valid > 0 else 0.0
        prec = precision_score(y_true, y_pred, average='macro', zero_division=0) if valid > 0 else 0.0
        rec = recall_score(y_true, y_pred, average='macro', zero_division=0) if valid > 0 else 0.0
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0) if valid > 0 else 0.0

        # Over-total metric: treat unlabeled samples as incorrect (assign opposite label)
        acc_over_total = accuracy_score(y_true_total, y_pred_total) if total > 0 else 0.0
        prec_over_total = precision_score(y_true_total, y_pred_total, average='macro', zero_division=0) if total > 0 else 0.0
        rec_over_total = recall_score(y_true_total, y_pred_total, average='macro', zero_division=0) if total > 0 else 0.0
        f1_over_total = f1_score(y_true_total, y_pred_total, average='macro', zero_division=0) if total > 0 else 0.0
        print("\nEvaluation Metrics:")
        print(f"Total samples (dataset): {total}")
        print(f"Valid predictions: {valid}")
        print(f"Accuracy (on valid): {acc:.4f}")
        print(f"Accuracy (over total {total}): {acc_over_total:.4f}")
        print(f"Precision (valid): {prec:.4f}")
        print(f"Recall (valid): {rec:.4f}")
        print(f"F1 (valid): {f1:.4f}")
        print(f"Precision (over total): {prec_over_total:.4f}")
        print(f"Recall (over total): {rec_over_total:.4f}")
        print(f"F1 (over total): {f1_over_total:.4f}")
        return {
            'total_samples': total,
            'valid': valid,
            'accuracy_valid': acc,
            'accuracy_over_total': acc_over_total,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'precision_over_total': prec_over_total,
            'recall_over_total': rec_over_total,
            'f1_over_total': f1_over_total,
            'results': results,
        }


def main():
    ap = argparse.ArgumentParser(description="Test BT reward checkpoint")
    ap.add_argument('--ckpt_dir', required=True, help='Checkpoint directory')
    ap.add_argument('--base_model', required=True, help='Base model path (e.g., models/Qwen2.5-3B)')
    ap.add_argument('--data_path', default='/user/Logic-RL/testset-v1_update.json')
    ap.add_argument('--num_samples', type=int, default=None)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--output_path', type=str, default=None, help='If set, save metrics json to this path')
    args = ap.parse_args()

    tester = BTRewardTester(args.ckpt_dir, args.base_model, device=args.device)
    metrics = tester.evaluate(args.data_path, num_samples=args.num_samples)
    if args.output_path:
        outp = Path(args.output_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open('w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"Saved metrics to {outp}")


if __name__ == '__main__':
    main()


