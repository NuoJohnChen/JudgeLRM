#!/usr/bin/env python3
"""
Convert DPO pairwise jsonl into single-sample reward jsonl.
Input format (per line):
{
  "chosen": [{"question": str, "answer": str}],
  "rejected": [{"question": str, "answer": str}],
  "chosen_score": float,
  "rejected_score": float
}

Output format (per line):
{ "prompt": "<|im_start|>user\n{Q}<|im_end|>\n<|im_start|>assistant\n{A}<|im_end|>", "score": float }
Two lines per input example (one for chosen, one for rejected).
"""

import argparse
import json
from pathlib import Path

def to_qwen_prompt(question: str, answer: str) -> str:
    return (
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n{answer}<|im_end|>"
    )


def convert(input_path: str, output_path: str) -> None:
    input_file = Path(input_path)
    output_file = Path(output_path)
    assert input_file.exists(), f"Input not found: {input_file}"

    total_in = 0
    total_out = 0

    with input_file.open('r', encoding='utf-8') as fin, output_file.open('w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total_in += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            try:
                chosen = obj["chosen"][0]
                rejected = obj["rejected"][0]
                cs = float(obj["chosen_score"])  # may raise
                rs = float(obj["rejected_score"])  # may raise
                cq = str(chosen["question"]).strip()
                ca = str(chosen["answer"]).strip()
                rq = str(rejected["question"]).strip()
                ra = str(rejected["answer"]).strip()
                if not cq or not ca or not rq or not ra:
                    continue
            except Exception:
                continue

            # chosen sample
            prompt_c = to_qwen_prompt(cq, ca)
            fout.write(json.dumps({"prompt": prompt_c, "score": cs}, ensure_ascii=False) + "\n")
            total_out += 1

            # rejected sample
            prompt_r = to_qwen_prompt(rq, ra)
            fout.write(json.dumps({"prompt": prompt_r, "score": rs}, ensure_ascii=False) + "\n")
            total_out += 1

    print(f"Converted {total_in} input lines into {total_out} reward samples → {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Convert DPO jsonl to reward jsonl (Qwen prompts)")
    parser.add_argument("--input", required=True, help="Path to DPO jsonl input")
    parser.add_argument("--output", required=True, help="Path to reward jsonl output")
    args = parser.parse_args()
    convert(args.input, args.output)


if __name__ == "__main__":
    main()
