#!/usr/bin/env python3
"""
Generate with SFT-think model and parse <answer> scores for PandaLM-style pairs.
"""

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_answers(text: str):
    # find <answer>x</answer>
    vals = [int(m) for m in re.findall(r"<answer>\s*([0-9]+)\s*</answer>", text)]
    if len(vals) >= 2:
        return vals[0], vals[1]
    return None, None


def build_prompt(question: str, a1: str, a2: str, reference: str | None = None):
    sys_prompt = (
        "You are a careful evaluator. Given a question and two assistant answers, first write your reasoning enclosed in "
        "<think>...</think>, then output two scores each in an <answer> tag (first for Assistant 1, second for Assistant 2)."
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    data = json.load(open(args.data_path, 'r', encoding='utf-8'))
    results = []
    for i, it in enumerate(data):
        if i % 100 == 0:
            print(f"{i}/{len(data)}")
        q = it.get('instruction', '')
        inp = it.get('input', '')
        if inp:
            q = q + '\n' + inp
        # Support multiple field names: output1/output2 or response1/response2
        a1 = it.get('output1') or it.get('response1') or ''
        a2 = it.get('output2') or it.get('response2') or ''
        if not a1 or not a2:
            continue
        prompt = build_prompt(q, a1, a2)
        inputs = tok(prompt, return_tensors='pt').to(args.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        gen = tok.decode(out[0], skip_special_tokens=True)[len(prompt):].strip()
        s1, s2 = parse_answers(gen)
        item_res = {
            'idx': it.get('idx', i),
            'question': q,
            'a1': a1,
            'a2': a2,
            'generation': gen,
            'score1': s1,
            'score2': s2,
        }
        results.append(item_res)
        # Print first five for quick inspection
        if len(results) <= 5:
            print(f"[Preview] idx={item_res['idx']}, score1={item_res['score1']}, score2={item_res['score2']}")
            print(gen[:400].replace('\n',' '))
            print('---')

    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()


