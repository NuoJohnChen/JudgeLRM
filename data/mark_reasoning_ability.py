import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm


CLIENT = OpenAI(
    base_url="https://api.openai.com/v1",
    api_key=os.environ.get("OPENAI_API_KEY", ""),
)

# MODEL = "gpt-5.1-2025-11-13"#"gpt-5-chat-latest"
MODEL = "gpt-5-mini-2025-08-07"#"gpt-5-nano-2025-08-07"#"deepseek-v3-250324"#"gpt-5-nano-2025-08-07"#"deepseek-v3-250324" #"gpt-4.1-mini"#"deepseek-v3-250324"#"gpt-5.1-2025-11-13"#"deepseek-v3.1"#"deepseek-v3.1"#"gpt-5.1-2025-11-13"#"gpt-5-chat-latest"#
MAX_WORKERS = 128#32


DEDUCTION_PROMPT = """
Task: Extract **Non-Trivial** Deductive Reasoning Steps
You are a severe logic auditor. Your goal is to separate "performative reasoning" (mimicking the style of thought) from "functional reasoning" (actual logical problem solving).

We define Valid Deduction strictly as:

1. **Dialectical Hypothesis (Branching)**:
   - The model must propose a scenario *specifically to test its validity against alternatives*.
   - EXCLUDE: Linear assumptions like "Assume the standard definition applies" which are just setting context.
   - INCLUDE: "If we assume A, then B follows, but B contradicts the prompt. Therefore..." (Reductio ad absurdum).
   - INCLUDE: "Let's consider Case 1 vs Case 2." (Exhaustive Proof).

2. **Negative Verification (Self-Correction)**:
   - The sentence must explicitly identify a **flaw, error, or oversight** in the model's own previous thinking.
   - EXCLUDE: "Double checking... everything looks correct." (This is SFT mimicry/hallucination support).
   - INCLUDE: "Wait, I made a mistake in the calculation here."
   - INCLUDE: "Actually, looking closer at the prompt, I misinterpreted 'X'."

**Extraction Rules:**
1. **The "Wait" Test**: If the sentence functions as a "Stop & Turn" signal (changing the direction of thought), extract it. If it is a "Go Ahead" signal (confirming the current path), IGNORE it.
2. **Substance over Form**: Do not extract sentences based on keywords like "assume" or "check". Extract them only if they contain the *content* of the counter-argument or the error found.
3. **Outcome Dependency**: Only extract steps that actually impacted the final judgment.

**OUTPUT FORMAT (STRICTLY FOLLOW THIS):**
1. Output a JSON array of objects.
2. Immediately after the JSON array (on a new line), output the total_count.

Example Output:
[
  {
    "category": "NegativeVerification",
    "sentence": "Wait, I calculated the integral wrong, it should be x^2.",
    "impact": "Fixed a calculation error that would have led to the wrong score."
  }
]
"total_count": 1

Input Thinking:
< thinking text >

Your Output:
"""

INDUCTION_PROMPT = """
Task: Extract **Substantive** Inductive Reasoning (Generalization)
You are a logic analyst specializing in identifying inductive leaps. Your goal is to distinguish between "summary statements" (restating facts) and "inductive generalizations" (synthesizing new rules from examples).

We define Valid Induction strictly as:
**The "Evidence-to-Rule" Leap**: The model must explicitly observe **multiple distinct specific instances** (e.g., Case A and Case B, or Example 1 and Example 2) and synthesize them into a **new general rule, trend, or pattern** that was not explicitly stated in the prompt.

**Extraction Rules (Strict Filter):**
1. **Require Specific Evidence**: Only extract if the text shows the model looking at *at least two* specific data points/examples/cases in the context *before* deriving the rule.
2. **Exclude Transitive Logic**: Do NOT extract simple forward reasoning (e.g., "A implies B, B implies C, therefore A implies C"). This is deduction, not induction.
3. **Exclude Summaries**: Do NOT extract sentences starting with "In summary," or "Generally," if they merely repeat what was just said without synthesizing a *new* abstract rule.

**OUTPUT FORMAT (STRICTLY FOLLOW THIS):**
1. Output a JSON array of objects.
2. Immediately after the JSON array (on a new line), output the total_count.

Example Output:
[
  {
    "sentence": "Seeing that Model A failed on the math question and Model B failed on the code question, it seems both struggle with formal logic.",
    "evidence_base": "Observed failure in Math case and Code case."
  }
]
"total_count": 1

Input Thinking:
< thinking text >

Your Output:
"""

ABDUCTION_PROMPT = """
Task: Extract **Explanatory** Abductive Reasoning (Backward Inference)
You are a reasoning auditor. Your task is to identify moments where the model performs **Inference to the Best Explanation**.

We define Valid Abduction strictly as:
**The "Surprise-to-Explanation" Loop**: The model starts from a specific **observation, anomaly, or surprising result** (the Effect) and works backward to propose a **plausible cause or hypothesis** (the Cause) that accounts for it.

**Extraction Rules (Strict Filter):**
1. **Directionality Check (Backward vs. Forward)**:
   - **REJECT** Forward Reasoning: "Rule X applies, so Result Y must be true." (This is Deduction).
   - **ACCEPT** Backward Reasoning: "We observe Result Y. This is unexpected. The most likely cause is Rule X."
2. **The "Why" Factor**: Extract sentences where the model asks "Why did this happen?" or "What accounts for this discrepancy?" and then answers it.
3. **Exclude Simple Causality**: Do not extract "Because A, B happened." Only extract if the model is *inferring* A from observing B.

**Template Matching (Strict)**:
- Only use "template" match_type if the sentence strictly follows the logical structure of: "Observation O exists -> Hypothesis H explains O".

**OUTPUT FORMAT (STRICTLY FOLLOW THIS):**
1. Output a JSON array of objects.
2. Immediately after the JSON array (on a new line), output the total_count.

Example Output:
[
  {
    "sentence": "The model output is empty, which suggests it might have triggered a safety filter.",
    "match_type": "clear_abduction_non_template",
    "matched_pattern": "",
    "observation_trigger": "Empty model output"
  }
]
"total_count": 1

Input Thinking:
< thinking text >

Your Output:
"""



def extract_think_text(output_text: str) -> str | None:
    if not output_text:
        return None
    match = re.search(r"<think>(.*?)</think>", output_text, re.S)
    if not match:
        return None
    return match.group(1).strip()


def call_openai(prompt: str) -> str:
    response = CLIENT.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


def build_prompt(template: str, thinking: str) -> str:
    return template.replace("< thinking text >", thinking)


def process_item(idx_item_pair):
    idx, item = idx_item_pair
    think_text = extract_think_text(item.get("output", ""))
    result = {
        "deduction": None,
        "induction": None,
        "abduction": None,
    }

    if not think_text:
        return idx, result

    deduction_prompt = build_prompt(DEDUCTION_PROMPT, think_text)
    induction_prompt = build_prompt(INDUCTION_PROMPT, think_text)
    abduction_prompt = build_prompt(ABDUCTION_PROMPT, think_text)

    try:
        result["deduction"] = call_openai(deduction_prompt)
    except Exception as exc:
        result["deduction"] = f"ERROR: {exc}"

    try:
        result["induction"] = call_openai(induction_prompt)
    except Exception as exc:
        result["induction"] = f"ERROR: {exc}"

    try:
        result["abduction"] = call_openai(abduction_prompt)
    except Exception as exc:
        result["abduction"] = f"ERROR: {exc}"

    return idx, result


def main():
    parser = argparse.ArgumentParser(description="Analyze reasoning capabilities within <think> sections.")
    parser.add_argument(
        "--input_path",
        default="/shared/hdd/nuochen/PandaLM/data/results_judgelrm3bthinktag.json",
        help="Path to the JSON results file containing <think> outputs.",
    )
    parser.add_argument(
        "--output_path",
        default="/shared/hdd/nuochen/PandaLM/data/results_judgelrm3bthinktag_reasoning_analysis.json",
        help="Path to save the annotated JSON results.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="If set, only the first N samples will be processed (useful for quick tests).",
    )
    args = parser.parse_args()

    with open(args.input_path, "r") as f:
        data = json.load(f)

    if args.max_samples is not None:
        if args.max_samples < 0:
            raise ValueError("--max_samples must be non-negative")
        data = data[: args.max_samples]
        print(f"[INFO] Limiting to first {len(data)} samples (max_samples={args.max_samples}).")

    analysis_results = [None] * len(data)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(process_item, (idx, item))
            for idx, item in enumerate(data)
        ]
        for future in tqdm(as_completed(futures), total=len(data)):
            idx, analysis = future.result()
            analysis_results[idx] = analysis

    for item, analysis in zip(data, analysis_results):
        item["reasoning_analysis"] = analysis

    with open(args.output_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Analysis completed. Results saved to {args.output_path}")


if __name__ == "__main__":
    main()