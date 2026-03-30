#!/usr/bin/env python3
"""
DPO模型概率比较测试脚本
通过比较模型对两个回答的概率来选择更好的回答
"""

import argparse
import json
import os
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 移除JudgeLM依赖，使用通用的概率比较方法


class DPOProbabilityProvider:
    """DPO模型概率比较提供者"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.load_model()
        
    def load_model(self):
        """加载DPO模型和tokenizer"""
        print(f"Loading DPO model from {self.model_path}")
        
        # 加载tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        except:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)
            
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            load_in_8bit=False,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        
        # 设置pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model.eval()
        print("DPO model loaded successfully")
        
    def build_prompt(self, instruction, input_text, response):
        """构建单个回答的prompt"""
        if input_text:
            question = instruction + '\n' + input_text
        else:
            question = instruction
            
        # 使用Qwen的对话格式
        prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
        return prompt
        
    def calculate_log_probability(self, prompt):
        """计算prompt的对数概率"""
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)
        
        # 计算logits
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
        # 计算对数概率
        # 使用log_softmax来避免数值不稳定
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # 计算整个序列的平均对数概率
        # 排除padding tokens
        valid_tokens = attention_mask.sum(dim=1)
        total_log_prob = 0
        
        for i in range(input_ids.size(0)):
            seq_len = valid_tokens[i].item()
            # 计算每个token的对数概率
            for j in range(seq_len - 1):  # 排除最后一个token（没有下一个token预测）
                token_id = input_ids[i, j + 1].item()
                token_log_prob = log_probs[i, j, token_id].item()
                total_log_prob += token_log_prob
                
        # 返回平均对数概率
        return total_log_prob / (valid_tokens.sum().item() - input_ids.size(0))
        
    def compare_responses(self, instruction, input_text, resp1, resp2):
        """比较两个回答的概率"""
        # 构建prompts
        prompt1 = self.build_prompt(instruction, input_text, resp1)
        prompt2 = self.build_prompt(instruction, input_text, resp2)
        
        # 计算对数概率
        log_prob1 = self.calculate_log_probability(prompt1)
        log_prob2 = self.calculate_log_probability(prompt2)
        
        # 比较概率
        if log_prob1 > log_prob2:
            result = 1  # resp1更好
        elif log_prob1 < log_prob2:
            result = 2  # resp2更好
        else:
            result = 0  # 平局
            
        return {
            "log_prob1": log_prob1,
            "log_prob2": log_prob2,
            "result": result,
            "probability1": np.exp(log_prob1),
            "probability2": np.exp(log_prob2)
        }
        
    def inference_batch(self, data_list):
        """批量推理"""
        results = []
        
        for i, item in enumerate(tqdm(data_list, desc="DPO概率比较")):
            comparison = self.compare_responses(
                instruction=item["instruction"],
                input_text=item.get("input", ""),
                resp1=item["response1"],
                resp2=item["response2"]
            )
            
            # 添加原始数据和比较结果
            result = item.copy()
            # 为后续与标签对齐，缺失时补充顺序索引为 idx
            if 'idx' not in result:
                result['idx'] = i
            result.update(comparison)
            results.append(result)
            
        return results


def main():
    parser = argparse.ArgumentParser(description="DPO模型概率比较测试")
    parser.add_argument("--model_path", type=str, required=True, help="DPO模型路径")
    parser.add_argument("--input_path", type=str, required=True, help="输入数据路径")
    parser.add_argument("--output_path", type=str, required=True, help="输出结果路径")
    parser.add_argument("--testset_path", type=str, default=None, help="可选：带有idx/label的测试集，用于计算Accuracy/Precision/Recall/F1")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 加载数据
    print(f"Loading data from {args.input_path}")
    with open(args.input_path, 'r') as f:
        data_list = json.load(f)
    
    # 创建推理提供者
    provider = DPOProbabilityProvider(args.model_path)
    
    # 执行推理
    print("Starting DPO probability comparison...")
    results = provider.inference_batch(data_list)
    
    # 保存结果
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {args.output_path}")
    
    # 统计结果
    total = len(results)
    resp1_wins = sum(1 for r in results if r["result"] == 1)
    resp2_wins = sum(1 for r in results if r["result"] == 2)
    ties = sum(1 for r in results if r["result"] == 0)
    
    print(f"\n=== 概率比较结果统计 ===")
    print(f"总样本数: {total}")
    print(f"Response1获胜: {resp1_wins} ({resp1_wins/total*100:.2f}%)")
    print(f"Response2获胜: {resp2_wins} ({resp2_wins/total*100:.2f}%)")
    print(f"平局: {ties} ({ties/total*100:.2f}%)")
    
    # 显示概率分布
    log_probs1 = [r["log_prob1"] for r in results]
    log_probs2 = [r["log_prob2"] for r in results]
    print(f"\n=== 概率分布 ===")
    print(f"Response1平均对数概率: {np.mean(log_probs1):.4f} ± {np.std(log_probs1):.4f}")
    print(f"Response2平均对数概率: {np.mean(log_probs2):.4f} ± {np.std(log_probs2):.4f}")

    # 计算分类指标（如提供了带标签的测试集）
    if args.testset_path is not None:
        try:
            with open(args.testset_path, 'r', encoding='utf-8') as f:
                testset = json.load(f)
            idx_to_label = {it['idx']: it['label'] for it in testset if isinstance(it, dict) and 'idx' in it and 'label' in it}
            y_true, y_pred = [], []
            for r in results:
                idx = r.get('idx')
                pred = r.get('result')
                if idx in idx_to_label and pred in (1, 2):
                    y_true.append(idx_to_label[idx])
                    y_pred.append(pred)
            if y_true:
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
                recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
                f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
                print("\nEvaluation Metrics:")
                print(f"Accuracy: {accuracy:.4f}")
                print(f"Precision: {precision:.4f}")
                print(f"Recall: {recall:.4f}")
                print(f"F1: {f1:.4f}")
            else:
                print("\nNo matched labels found; skipped metrics.")
        except Exception as e:
            print(f"\nFailed to compute metrics: {e}")


if __name__ == "__main__":
    main()
