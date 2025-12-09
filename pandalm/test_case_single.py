from transformers import pipeline, GenerationConfig
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

json_input=  {
    "idx": 549,
    "motivation_app": "Google Docs",
    "cmp_key": "cerebras-gpt-6.7B_pythia-6.9b",
    "instruction": "Include important study notes and key points that someone should know about the given subject.",
    "input": "history of the USA",
    "response1": "The history of the United States is one of the most influential and influential countries in the world. It has had a profound impact on the world and continues to shape the world today. The United States has had a long and storied history, from its founding in 1775 to the present day. It has had a tumultuous and tumultuous history, from the Revolutionary War to the Civil War and beyond. The United States has had a tumultuous and tumultuous history, from the Revolutionary War to the Civil War and beyond. The United States has had a long and storied history, from its founding in 1775 to the present day. It has had a profound impact on the world and continues to shape the world today.",
    "response2": "1. The United States of America was founded in 1776.\n2. The Declaration of Independence was signed in 1776.\n3. The Constitution of the United States of America was signed in 1787.\n4. The Civil War began in 1861.\n5. The Emancipation Proclamation was issued in 1863.\n6. The 13th Amendment was ratified in 1865.\n7. The 14th Amendment was ratified in 1868.\n8. The 15th Amendment was ratified in 1870.\n9. The 16th Amendment was ratified in 1913.\n10. The 17th Amendment was ratified in 1913.\n11. The 18th Amendment was ratified in 1919.\n12. The 19th Amendment was ratified in 1920.\n13. The 20th Amendment was ratified in 1933.\n14. The 21st Amendment was ratified in 1933.\n15. The 22nd Amendment was ratified in",
    "annotator1": 2,
    "annotator2": 2,
    "annotator3": 2,
    "label": 2,
    "needed_reasoning_rate1-10": 7,
    "rate_explanation": "The task requires evaluating the quality of responses based on their adherence to the instruction to include important study notes and key points about the history of the USA. Response1 is repetitive and lacks specific details, while Response2 provides a clear, concise list of key historical events. The reasoning needed to judge these responses involves assessing clarity, specificity, and relevance to the instruction, which is moderately complex.\n----------------------------------------"
}


question = json_input.get("instruction", "").strip()+"\n"+json_input.get("input", "").strip()
answer_1 = json_input.get("response1", "").strip()
answer_2 = json_input.get("response2", "").strip()

def build_single_score_prompt(question_text, answer_text):
    return """<|im_start|>system
You are a helpful assistant. The assistant first performs a detailed, step-by-step reasoning process in its mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. Now the user asks you to judge ONE AI assistant's response to the question. Provide a single score from 1-10 (higher=better). Criteria include helpfulness, relevance, accuracy, and level of detail. Avoid bias from order, length, or style. After thinking, provide ONLY the numeric score within <answer> </answer> tags.
<|im_end|>
<|im_start|>user
[Question]
{question}

[Assistant's Answer]
{answer}
<|im_end|>
<|im_start|>assistant
<think>""".format(question=question_text, answer=answer_text)

def extract_score_from_output(text):
    print("####")
    print(text)
    print("####")
    match = re.search(r"<answer>\s*([0-9]+(?:\.[0-9]+)?)\s*</answer>", text)
    if match:
        score_str = match.group(1)
        try:
            # Prefer integer if it looks like an integer
            return int(score_str) if score_str.isdigit() else float(score_str)
        except ValueError:
            return None
    return None

# model_path ="/home/nuochen/.cache/huggingface/hub/models--zhiyuanhucs--qwen25_7b_zhiyuan_function_rm_323-step-600/snapshots/fbf74d02832769bf78bc44d6c68cc0e9750c5a12/"#"/shared/hdd/nuochen/models/judge_7B_step1000"#"/home/nuochen/.cache/huggingface/hub/models--zhiyuanhucs--qwen25_7b_zhiyuan_function_rm_323-step-600/snapshots/fbf74d02832769bf78bc44d6c68cc0e9750c5a12/"
model_path="/shared/hdd/nuochen/models/Qwen3-8B-Instruct-think-evaluator"#"/shared/hdd/nuochen/models/JudgeLRM-8B"

device = "cuda:5"


tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map=device,
    torch_dtype="auto"
)


generation_config = GenerationConfig(
    max_new_tokens=4096,
)


def score_single_answer(question_text, answer_text):
    single_prompt = build_single_score_prompt(question_text, answer_text)
    inputs_local = tokenizer(single_prompt, return_tensors="pt").to(device)
    outputs_local = model.generate(
        **inputs_local,
        generation_config=generation_config
    )
    decoded = tokenizer.decode(outputs_local[0], skip_special_tokens=False)
    score = extract_score_from_output(decoded)
    return score, decoded

# Score response1 and response2 separately
score1, raw1 = score_single_answer(question, answer_1)
score2, raw2 = score_single_answer(question, answer_2)

print(f"response1 score: {score1}")
print(f"response2 score: {score2}")

# Optional: print raw generations for debugging
# print(raw1)
# print(raw2)

# Check GPU usage
print(f"Model on GPU: {next(model.parameters()).is_cuda}")