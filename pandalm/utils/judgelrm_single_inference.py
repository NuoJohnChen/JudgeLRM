import argparse
import torch
import transformers
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer, AutoConfig
import json
import sys
import logging
from typing import Union, Dict
from tqdm import tqdm
import re, random
import os
from common import *  # assume conv_judge_pair and related variables are defined in common

import logging

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def load_fsdp_checkpoint(model_path, base_model_path):
    """
    Load FSDP checkpoint and merge weights
    """
    print(f"Start loading FSDP checkpoint: {model_path}")
    config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    
    # Choose device_map based on available GPUs
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        device_map = "auto"  # use auto mode for multi-GPU
        print(f"Detected multi-GPU ({torch.cuda.device_count()} cards), using auto mode")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
        )
    else:
        # Single GPU: load to CPU then move to GPU
        print(f"Single-GPU environment, load to CPU then move to GPU")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map=None,  # load to CPU first
            trust_remote_code=True,
        )
        # Explicitly move to GPU
        model = model.to("cuda:0")
        print(f"Model moved to: {next(model.parameters()).device}")

    fsdp_files = [f for f in os.listdir(model_path) if f.startswith("model_world_size_")]
    fsdp_files.sort()
print(f"Found FSDP files: {fsdp_files}")

    state_dict = {}
    for fsdp_file in fsdp_files:
        fsdp_path = os.path.join(model_path, fsdp_file)
        print(f"Loading {fsdp_file}...")
        checkpoint = torch.load(fsdp_path, map_location="cpu", weights_only=False)
        for key, value in checkpoint.items():
            if not isinstance(value, torch.Tensor):
                continue
            if hasattr(value, '_local_tensor'):
                value = value._local_tensor
            elif hasattr(value, 'to_local'):
                value = value.to_local()
            if key in state_dict and isinstance(value, torch.Tensor) and isinstance(state_dict[key], torch.Tensor):
                try:
                    state_dict[key] = torch.cat([state_dict[key], value], dim=0)
                except Exception:
                    state_dict[key] = value
            else:
                state_dict[key] = value

print("Loading merged weights into model...")
    model.load_state_dict(state_dict, strict=False)
    return model


class PandaLMBatchInferenceProvider(object):
    """
    Evaluate batch responses with PandaLM
    """

    def __init__(self, model_path, base_model_path=None) -> None:
        super().__init__()
        # Resolve paths; support huggingface subdir and FSDP checkpoint
        if "huggingface" in model_path:
            tokenizer_path = model_path
            model_path_for_tokenizer = model_path
        else:
            huggingface_path = os.path.join(model_path, "huggingface")
            if os.path.exists(huggingface_path):
                fsdp_files = [f for f in os.listdir(model_path) if f.startswith("model_world_size_")]
                if fsdp_files:
                    if base_model_path and os.path.exists(base_model_path):
                        print(f"Detected FSDP checkpoint: {fsdp_files}")
                        print(f"Using base model as tokenizer: {base_model_path}")
                        tokenizer_path = base_model_path
                        model_path_for_tokenizer = model_path
                    else:
                        raise ValueError("FSDP checkpoint detected but no valid --base path provided")
                else:
                    tokenizer_path = huggingface_path
                    model_path_for_tokenizer = huggingface_path
            else:
                tokenizer_path = model_path
                model_path_for_tokenizer = model_path

        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False, trust_remote_code=True)

        # Use custom load for FSDP checkpoint; otherwise regular load
        fsdp_files_root = [f for f in os.listdir(model_path) if f.startswith("model_world_size_")]
        if fsdp_files_root and base_model_path:
            print("Detected FSDP checkpoint, using custom load method")
            model = load_fsdp_checkpoint(model_path, base_model_path)
        else:
            # Choose device_map based on available GPUs
            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                device_map = "auto"  # use auto mode for multi-GPU
                print(f"Detected multi-GPU ({torch.cuda.device_count()} cards), using auto mode")
                model = AutoModelForCausalLM.from_pretrained(
                    model_path_for_tokenizer,
                    load_in_8bit=False,
                    torch_dtype=torch.bfloat16,
                    device_map=device_map,
                    trust_remote_code=True,
                )
            else:
                # Single GPU: load to CPU then move to GPU
                print(f"Single-GPU environment, load to CPU then move to GPU")
                model = AutoModelForCausalLM.from_pretrained(
                    model_path_for_tokenizer,
                    load_in_8bit=False,
                    torch_dtype=torch.bfloat16,
                    device_map=None,  # load to CPU first
                    trust_remote_code=True,
                )
                # Explicitly move to GPU
                model = model.to("cuda:0")
                print(f"Model moved to: {next(model.parameters()).device}")
        if tokenizer.pad_token is None:
            self.smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
        tokenizer.add_special_tokens(
            {
                "eos_token": "</s>",
                "bos_token": "</s>",
                "unk_token": "</s>",
            }
        )
        self.tokenizer = tokenizer

        model.config.pad_token_id = self.tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
        model.eval()

        # Debug info
        print(f"Model device: {next(model.parameters()).device}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"Current CUDA device: {torch.cuda.current_device()}")
            print(f"Visible CUDA device count: {torch.cuda.device_count()}")

        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)

        self.model = model
        self.prepared = []
        self.prepared_map = []  # record which sample/response each prepared item belongs to
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

    def build_single_score_prompt(self, instruction, input, answer):
        answer = self.pattern.sub("", answer.strip()).strip()
        template = """<|im_start|>system
You are a helpful assistant. The assistant first performs a detailed, step-by-step reasoning process in its mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. Now the user asks you to judge ONE AI assistant's response to the question. Provide a single score from 1-10 (higher=better). Criteria include helpfulness, relevance, accuracy, and level of detail. Avoid bias from order, length, or style. After thinking, provide ONLY the numeric score within <answer> </answer> tags.
<|im_end|>
<|im_start|>user
[Question]
{question}

[Assistant's Answer]
{answer}
<|im_end|>
<|im_start|>assistant
<think>"""
        question_text = instruction + ("\n" + input if input else "")
        return template.format(question=question_text, answer=answer)

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
        answers = re.findall(r'<answer>\s*(\d+)\s*</answer>', text)
        
        try:
            
            if len(answers) >= 2:
                num1 = float(answers[0])
            num2 = float(answers[1])
            
            if num1 > num2:
                return 1
            elif num1 < num2:
                return 2
            else:
                return 0
        except:
            print("bad output:",text)
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
        # Build two single-score prompts for one sample
        prompt1 = self.build_single_score_prompt(instruction, input, response1)
        prompt2 = self.build_single_score_prompt(instruction, input, response2)
        self.prepared.append(self.tokenizer(prompt1, return_tensors="pt", padding=True))
        self.prepared_map.append({"resp_index": 1})
        self.prepared.append(self.tokenizer(prompt2, return_tensors="pt", padding=True))
        self.prepared_map.append({"resp_index": 2})

    def postprocess_output(self, text):
        # print(text)
        try:
            text = text.strip().split("<|im_start|>assistant")[1].strip().split("</think>")[1].strip()
            self.pattern.sub("", text.strip()).strip()
        except:
            print("bad output:",text)
            text = "<answer>0</answer>"
        
        return text

    def filter_special_token(self, text):
        return self.pattern.sub("", text.strip()).strip()

    def inference(
        self,
        temperature=0.1,
        top_p=1,
        top_k=1,
        num_beams=4,
        max_new_tokens=4096,
        repetition_penalty=1.2,
    ):
        generated = []
        printed_count = 0

        for idx in tqdm(range(len(self.prepared))):
            inputs = self.prepared[idx]
            input_ids = inputs["input_ids"].to(self.model.device)
            generation_config = GenerationConfig(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_beams=num_beams,
                early_stopping=True,
                repetition_penalty=repetition_penalty,
            )
            with torch.no_grad():
                generation_output = self.model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=max_new_tokens,
                )

            for j in range(len(generation_output.sequences)):
                # print("####",generation_output.sequences)
                s = generation_output.sequences[j]
                output = self.tokenizer.decode(s)
                resp = self.postprocess_output(output)
                resp = self.filter_special_token(resp)
                generated.append(resp)
                if printed_count < 20:
                    print(f"generated[{printed_count}] output={resp}")
                    printed_count += 1

        return generated

    def extract_single_score(self, text):
        try:
            matches = re.findall(r"<answer>\s*([0-9]+(?:\.[0-9]+)?)\s*</answer>", text)
            if not matches:
                return 0.0
            score_str = matches[-1]
            return float(score_str)
        except Exception:
            return 0.0


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO
    )

    parser = argparse.ArgumentParser(description="PandaLM batch inference script")
    parser.add_argument("-s", "--seed", type=int, default=2023)
    parser.add_argument("-m", "--model_name", default="/shared/hdd/nuochen/models/judge_7B_step1000")
    parser.add_argument(
        "-i",
        "--input_path",
        default="/shared/hdd/nuochen/Logic-RL/testset-v1_update.json",#"/shared/hdd/nuochen/Logic-RL/testset_with_responses_pandlm_r1_back.json",
    )
    parser.add_argument("-o", "--output_path", default="/shared/hdd/nuochen/PandaLM/data/results_jugdelrm7b_single.json")
    parser.add_argument("--base", default=None, help="Base model path for FSDP checkpoint loading")

    args = parser.parse_args()

    logging.info(args)

    seed_everything(args.seed)

    logging.info(f"Loading model from {args.model_name}")
    handler = PandaLMBatchInferenceProvider(
        model_path=args.model_name,
        base_model_path=args.base,
    )
    with open(args.input_path) as f:
        input_data = json.load(f)

    results = []
    for item in tqdm(input_data):
        handler.preprocess_input(
            instruction=item["instruction"],
            input=item["input"],
            response1=item["response1"],
            response2=item["response2"],
        )
    generated = handler.inference()

    # Each sample has two entries in generated list: resp1, resp2
    for idx, item in enumerate(input_data):
        g1 = generated[2 * idx]
        g2 = generated[2 * idx + 1]
        out1 = handler.extract_single_score(g1)
        out2 = handler.extract_single_score(g2)
        item["output1"] = out1
        item["output2"] = out2
        if out1 > out2:
            item["result"] = 1
        elif out1 < out2:
            item["result"] = 2
        else:
            item["result"] = 0
        results.append(item)

    # print(results)
    # Print first 10 items of output1, output2, result
    for i in range(min(10, len(results))):
        r = results[i]
        print(f"idx={i} output1={r['output1']} output2={r['output2']} result={r['result']}")

    if args.output_path:
        with open(args.output_path, "w") as f:
            json.dump(results, f)
    else:
        print(results)
