#!/usr/bin/env python3
"""
Inference script for SFT-think judge model.

Given a question (and optional input) and two responses, the model should produce:

### Response:
<think>
...explanation...
</think>
<answer>S1</answer><answer>S2</answer>

This script builds the same prompt used in training, generates the output,
parses the two <answer> scores, and determines which response is better.
"""

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm


def build_prompt(question: str, a1: str, a2: str, reference: str | None = None) -> str:
    sys_prompt = (
        "You are a careful evaluator. Given a question and two assistant answers, "
        "first write your reasoning enclosed in <think>...</think>, then output two scores "
        "each in an <answer> tag (first for Assistant 1, second for Assistant 2). Scores are integers 1-10."
    )
    parts = [
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
    if reference:
        parts += ["", "[Reference]", reference]
    parts += ["", "### Response:"]
    return "\n".join(parts)


def parse_answers(text: str):
    nums = [int(m) for m in re.findall(r"<answer>\s*([0-9]+)\s*</answer>", text)]
    if len(nums) >= 2:
        return nums[0], nums[1]
    return None, None


def fallback_extract_scores(text: str):
    
    region = text.split('<answer>', 1)[0]
    # 1) 优先匹配“pair-wise ... are X Y”或“scores ... are X ... Y”类表达
    m = re.search(r"(?:pair[-\s]*wise|scores?)\b[^\d]{0,50}(?:are|is|:)\s*(\d{1,2})\D+(\d{1,2})",
                  region, flags=re.IGNORECASE)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if 1 <= a <= 10 and 1 <= b <= 10:
            return a, b
    # 2) 移除范围表达（如 1-10），避免把 1 和 10 当作真实分数
    region_clean = re.sub(r"\b\d{1,2}\s*-\s*\d{1,2}\b", "", region)
    # 3) 提取所有 1..10 的整数，取“最后两个”作为最终分数（更贴近结论位置）
    nums = [int(n) for n in re.findall(r"\b(\d{1,2})\b", region_clean)]
    nums = [n for n in nums if 1 <= n <= 10]
    if len(nums) >= 2:
        return nums[-2], nums[-1]
    return None, None


class StopAfterTwoAnswers(StoppingCriteria):
    def __init__(self, tokenizer: AutoTokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.close_tag = "</answer>"

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # decode only the newly generated tail for efficiency could be added, this is simple and safe
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return text.count(self.close_tag) >= 2


def main():
    ap = argparse.ArgumentParser(description="SFT-think judge inference")
    ap.add_argument("--model_path", type=str, default="/disk2/user/JudgeLM/output/Qwen2.5-3B-Instruct-sft-think")
    ap.add_argument("--input_path", type=str, default="/user/Logic-RL/testset-v1_update.json")
    ap.add_argument("--output_path", type=str, default="/user/PandaLM/data/results_sft_think.json")
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--testset_path", type=str, default=None, help="Optional testset json with idx/label to compute metrics")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    data = json.load(open(args.input_path, 'r', encoding='utf-8'))
    results = []
    for i, item in enumerate(tqdm(data, desc="SFT-think inference")):
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        question = instruction + ("\n" + input_text if input_text else "")
        resp1 = item.get('output1') or item.get('response1') or ""
        resp2 = item.get('output2') or item.get('response2') or ""
        if not resp1 or not resp2:
            continue
        prompt = build_prompt(question, resp1, resp2)
        # Anchor the required format to reduce free-form preface
        prompt_with_anchor = prompt + "\n<think>\n"
        inputs = tokenizer(prompt_with_anchor, return_tensors='pt').to(args.device)
        gen_cfg = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature if args.temperature > 0 else 0.2,
            top_p=args.top_p if args.temperature > 0 else 0.9,
            num_beams=1,
            do_sample=True,
            early_stopping=True,
            repetition_penalty=1.1,
        )
        stopping = StoppingCriteriaList([StopAfterTwoAnswers(tokenizer)])
        with torch.no_grad():
            out = model.generate(**inputs, generation_config=gen_cfg, stopping_criteria=stopping)
        full_text = tokenizer.decode(out[0], skip_special_tokens=True)
        gen = full_text[len(prompt_with_anchor):].strip()
        s1, s2 = parse_answers(gen)
        
        if (s1 is None or s2 is None) or (s1 == 5 and s2 == 5):
            fb1, fb2 = fallback_extract_scores(gen)
            if fb1 is not None and fb2 is not None:
                s1, s2 = fb1, fb2
        if s1 is None or s2 is None:
            winner = 0
        else:
            winner = 1 if s1 > s2 else (2 if s2 > s1 else 0)
        res = {
            "idx": item.get('idx', i),
            "instruction": instruction,
            "input": input_text,
            "response1": resp1,
            "response2": resp2,
            "generation": gen,
            "score1": s1,
            "score2": s2,
            "result": winner,
        }
        results.append(res)

        # Print first five generations in-loop for quick inspection
        if i < 5:
            print(f"[Preview] idx={res['idx']}, score1={res['score1']}, score2={res['score2']}, result={res['result']}")
            print(gen[:400].replace('\n',' '))
            print('---')

    # (Kept previews printed during loop)

    # Optional metrics
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

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(results)} results to {out_path}")


if __name__ == "__main__":
    main()


