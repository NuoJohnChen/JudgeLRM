import argparse
import torch
import transformers
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer
import json
import logging
import os
import random
import re
import sys
from typing import Dict

from openai import OpenAI
from tqdm import tqdm

from common import *  # assume conv_judge_pair and related variables are defined in common

try:
    import torch
except ImportError:
    torch = None


def seed_everything(seed):
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)


class PandaLMBatchInferenceProvider(object):
    """
    Evaluate batch responses with PandaLM
    """

    def __init__(self, client: OpenAI, model_name: str, max_retries: int = 5) -> None:
        super().__init__()
        self.client = client
        self.model_name = model_name
        self.max_retries = max_retries
        self.prepared = []
        self.pattern = re.compile(
            r"<unk>|<pad>|<s>|</s>|\[PAD\]|<\|endoftext\|>|\[UNK\]|\[CLS\]|\[MASK\]|<\|startofpiece\|>|<\|endofpiece\|>|\[gMASK\]|\[sMASK\]"
        )

    def build_judgelrm_prompt(
        self, instruction, input, resp1, resp2, result=None, explain=None, ref=None
    ):
        resp1 = self.pattern.sub("", resp1.strip()).strip()
        resp2 = self.pattern.sub("", resp2.strip()).strip()
        # conv = conv_judge_pair.copy()
        # template = conv.prompt_template
        # conv.sep = "\n"

        # rsp = f"### Response 1:\n{resp1}\n\n### Response 2:\n{resp2}"

        template = """<|im_start|>system
You are a helpful assistant. The assistant first performs a detailed, step-by-step reasoning process in its mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> detailed reasoning process here, explaining each step of your evaluation for both assistants </think><answer> answer here </answer>. Now the user asks you to judge the performance of two AI assistants in response to the question. Score assistants 1-10 (higher=better). Criteria includes helpfulness, relevance, accuracy, and level of detail. Avoid order, length, style or other bias. After thinking, when you finally reach a conclusion, clearly provide your evaluation scores within <answer> </answer> tags, i.e. for example,<answer>3</answer><answer>5</answer>
<|im_end|>
<|im_start|>user
[Question]
{question}

[Assistant 1's Answer]
{answer_1}

[Assistant 2's Answer]
{answer_2}
<|im_end|>
<|im_start|>assistant
<think>"""
        if input:
            input_sequence = template.format(
                question=instruction + '\n' + input,
                answer_1=resp1,
                answer_2=resp2
            )
            # print(input_sequence)
        else:
            input_sequence = template.format(
                question=instruction,
                answer_1=resp1,
                answer_2=resp2
            )
        if result:
            output_sequence = (
                f"{result}\n\n### Reason: {explain}\n\n### Reference: {ref}\n"
            )
            return input_sequence, output_sequence
        else:
            return input_sequence

    def build_pandalm_prompt(
        self, instruction, input, resp1, resp2, result=None, explain=None, ref=None
    ):
        resp1 = self.pattern.sub("", resp1.strip()).strip()
        resp2 = self.pattern.sub("", resp2.strip()).strip()
        rsp = f"### Response 1:\n{resp1}\n\n### Response 2:\n{resp2}"
        if input:
            input_sequence = f"Below are two responses for a given task. The task is defined by the Instruction with an Input that provides further context. Evaluate the responses and generate a reference answer for the task.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n{rsp}\n\n### Evaluation:\n"
        else:
            input_sequence = f"Below are two responses for a given task. The task is defined by the Instruction. Evaluate the responses and generate a reference answer for the task.\n\n### Instruction:\n{instruction}\n\n{rsp}\n\n### Evaluation:\n"
        if result:
            output_sequence = (
                f"{result}\n\n### Reason: {explain}\n\n### Reference: {ref}\n"
            )
            return input_sequence, output_sequence
        else:
            return input_sequence

    def parse_pandalm_response(self, text):
        sp = text.strip().split("\n")
        if sp[0] in ["1", "2"]:
            return int(sp[0])
        elif sp[0].lower() == "tie":
            return 0
        else:
            return 0
        
    def parse_judge_lrm_response(self, text):
        import re
        answers = re.findall(r'<answer>(\d+)</answer>', text)
        
        
        if len(answers) >= 2:
            num1 = int(answers[0])
            num2 = int(answers[1])
            
            if num1 > num2:
                return 1
            elif num1 < num2:
                return 2
            else:
                return 0

    def smart_tokenizer_and_embedding_resize(
        self,
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
    ):
        """Resize tokenizer and embedding.

        Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
        """
        num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))

        if num_new_tokens > 0:
            input_embeddings = model.get_input_embeddings().weight.data
            output_embeddings = model.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True
            )
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True
            )

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

    def preprocess_input(self, instruction, input, response1, response2):
        prompt = self.build_judgelrm_prompt(instruction, input, response1, response2)
        self.prepared.append(prompt)

    def postprocess_output(self, text):
        text = text.strip()
        if "<think>" in text and "</think>" in text:
            text = text.split("</think>", 1)[1]
        return self.pattern.sub("", text.strip()).strip()

    def filter_special_token(self, text):
        return self.pattern.sub("", text.strip()).strip()

    def inference(
        self,
        temperature=0,
        top_p=1,
        max_new_tokens=2048,
    ):
        generated = []

        for prompt in tqdm(self.prepared):
            last_error = None
            for _ in range(self.max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {
                                "role": "user",
                                "content": prompt,
                            }
                        ],
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_new_tokens,
                    )
                    output = response.choices[0].message.content
                    resp = self.postprocess_output(output)
                    resp = self.filter_special_token(resp)
                    generated.append(resp)
                    break
                except Exception as exc:
                    last_error = exc
                    continue
            else:
                raise RuntimeError(f"Failed to generate response after retries: {last_error}")

        return generated


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO
    )

    parser = argparse.ArgumentParser(description="PandaLM batch inference script")
    parser.add_argument("-s", "--seed", type=int, default=2023)
    parser.add_argument("-m", "--model_name", default="deepseek-r1")
    parser.add_argument(
        "-i",
        "--input_path",
        default="/shared/hdd/nuochen/Logic-RL/testset-v1_update.json",
    )
    parser.add_argument("-o", "--output_path", default="/shared/hdd/nuochen/PandaLM/data/results_r1.json")
    parser.add_argument(
        "--api_base",
        default="https://api.openai.com/v1",
        help="Deepseek-compatible OpenAI base url",
    )
    parser.add_argument(
        "--api_key",
        default=None,
        help="API key for the Deepseek endpoint (falls back to DEEPSEEK_API_KEY env)",
    )

    args = parser.parse_args()

    logging.info(args)

    seed_everything(args.seed)

    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("API key not provided. Pass --api_key or set DEEPSEEK_API_KEY.")

    logging.info(f"Preparing client for model {args.model_name}")
    client = OpenAI(
        base_url=args.api_base,
        api_key=api_key,
    )
    handler = PandaLMBatchInferenceProvider(
        client=client,
        model_name=args.model_name,
    )
    with open(args.input_path) as f:
        input_data = json.load(f)

    results = []
    for item in tqdm(input_data):
        handler.preprocess_input(
            instruction=item["instruction"],
            input=item.get("input", ""),
            response1=item["response1"],
            response2=item["response2"],
        )
    generated = handler.inference()

    for item, output in zip(input_data, generated):
        item["output"] = output
        item["result"] = handler.parse_judge_lrm_response(output)
        results.append(item)

    # print(results)
    if args.output_path:
        with open(args.output_path, "w") as f:
            json.dump(results, f)
    else:
        print(results)
