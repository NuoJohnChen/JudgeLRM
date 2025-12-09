#!/usr/bin/env python3
"""
Batch inference script for JudgeLM models.

Supports two model types:
1. JudgeLRM (generative model) - original PandaLM-style inference
2. Cross-Encoder BT (discriminative model) - trained with train_bt_cross_encoder.py

Example usage for Cross-Encoder BT:
    python scripts/judgelrm_inf_crossencoderbt.py \
        --model_type cross_bt \
        --model_name /shared/hdd/nuochen/JudgeLM/output/Qwen2.5-3B-bt-cross/checkpoint-6220 \
        --base_model_path /shared/ssd/models/Qwen2.5-3B \
        --input_path /path/to/test.json \
        --output_path /path/to/results.json \
        --debias

Example usage for JudgeLRM:
    python scripts/judgelrm_inf_crossencoderbt.py \
        --model_type judgelrm \
        --model_name /path/to/judgelrm/model \
        --input_path /path/to/test.json \
        --output_path /path/to/results.json
"""

import argparse
import torch
import transformers
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer, AutoModel
import json
import sys
import logging
from typing import Union, Dict
from tqdm import tqdm
import re, random
import torch.nn as nn
import torch.nn.functional as F

import logging

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


# Cross-encoder BT model classes (from train_bt_cross_encoder.py)
PAIR_TEMPLATE = (
    "<|im_start|>system\n"
    "You are a strict evaluator. Compare two candidate answers and decide "
    "which one better addresses the user.\n"
    "<|im_end|>\n"
    "<|im_start|>user\n{question}<|im_end|>\n"
    "<|im_start|>assistant\n"
    "Response A:\n{answer_a}\n\n"
    "Response B:\n{answer_b}\n\n"
    "Which response is better? Answer with only 'A' or 'B'.\n"
    "<|im_end|>"
)


def build_pair_prompt(question: str, answer_a: str, answer_b: str) -> str:
    return PAIR_TEMPLATE.format(question=question, answer_a=answer_a, answer_b=answer_b)


class CrossBTHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_state).squeeze(-1)


class CrossBTModel(nn.Module):
    def __init__(self, base: AutoModel):
        super().__init__()
        self.base = base
        self.head = CrossBTHead(base.config.hidden_size)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        seq_lens = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(hidden.size(0), device=hidden.device)
        eos_states = hidden[batch_indices, seq_lens]
        return eos_states

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        pooled = self.encode(input_ids, attention_mask)
        # Ensure pooled and head have matching dtypes
        head_dtype = next(self.head.parameters()).dtype
        if pooled.dtype != head_dtype:
            pooled = pooled.to(dtype=head_dtype)
        logits = self.head(pooled)
        return logits


class CrossBTInferenceProvider(object):
    """
    Inference provider for Cross-Encoder BT model.
    Loads checkpoint from train_bt_cross_encoder.py and performs pairwise comparison.
    """

    def __init__(self, model_path, base_model_path=None, debias=True):
        super().__init__()
        # Load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        except:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        self.tokenizer = tokenizer
        self.debias = debias  # If True, average predictions from both orderings
        
        # Load base model config (use base_model_path if provided, otherwise try checkpoint or model_path)
        import os
        if base_model_path is None:
            # Check if checkpoint has config.json
            if os.path.exists(os.path.join(model_path, "config.json")):
                base_model_path = model_path
            else:
                # Need to specify base_model_path if checkpoint doesn't have config
                raise ValueError(
                    f"Checkpoint at {model_path} doesn't contain config.json. "
                    "Please specify --base_model_path to the original base model (e.g., /shared/ssd/models/Qwen2.5-3B)"
                )
        
        print(f"Loading base model config from {base_model_path}")
        # Load base model to get the architecture (but we'll load weights from checkpoint)
        # Don't use device_map="auto" here since we'll load weights from checkpoint
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base = AutoModel.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
        )
        
        # Create CrossBTModel
        model = CrossBTModel(base)
        
        # Load checkpoint weights (checkpoint contains full model: base + head)
        checkpoint_paths = [
            os.path.join(model_path, "model.safetensors"),
            os.path.join(model_path, "pytorch_model.bin"),
            os.path.join(model_path, "bt_reward_head.bin"),  # fallback: head only
        ]
        
        loaded = False
        for ckpt_path in checkpoint_paths:
            if os.path.exists(ckpt_path):
                print(f"Loading weights from {ckpt_path}")
                try:
                    if ckpt_path.endswith("bt_reward_head.bin"):
                        # Only head weights (fallback case)
                        model.head.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
                        loaded = True
                        break
                    else:
                        # Full model state dict (normal case)
                        if ckpt_path.endswith(".safetensors"):
                            try:
                                from safetensors.torch import load_file
                                state_dict = load_file(ckpt_path)
                            except ImportError:
                                print("safetensors not installed, skipping .safetensors file")
                                continue
                        else:
                            state_dict = torch.load(ckpt_path, map_location="cpu")
                        
                        # Check if checkpoint contains full model (base + head) or just head
                        has_base = any(k.startswith("base.") for k in state_dict.keys())
                        has_head = any(k.startswith("head.") for k in state_dict.keys())
                        
                        if has_base and has_head:
                            # Full model checkpoint: load everything
                            print("Checkpoint contains full model (base + head), loading all weights...")
                            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                            if missing_keys:
                                print(f"Warning: Missing keys: {missing_keys[:5]}...")
                            if unexpected_keys:
                                print(f"Warning: Unexpected keys: {unexpected_keys[:5]}...")
                            loaded = True
                            break
                        elif has_head:
                            # Only head weights
                            print("Checkpoint contains only head weights, loading head...")
                            head_state_dict = {k.replace("module.head.", "head."): v 
                                             for k, v in state_dict.items() 
                                             if k.startswith("head.") or k.startswith("module.head.")}
                            model.head.load_state_dict(head_state_dict, strict=True)
                            loaded = True
                            break
                        else:
                            print(f"Warning: Checkpoint doesn't contain expected keys (base.* or head.*)")
                            continue
                except Exception as e:
                    print(f"Failed to load from {ckpt_path}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        if not loaded:
            raise RuntimeError(
                f"Could not load checkpoint weights from {model_path}! "
                "Please check that the checkpoint contains model.safetensors or pytorch_model.bin"
            )
        
        # Ensure head has the same dtype as base model
        base_dtype = next(model.base.parameters()).dtype
        if next(model.head.parameters()).dtype != base_dtype:
            print(f"Converting head weights from {next(model.head.parameters()).dtype} to {base_dtype}")
            model.head = model.head.to(dtype=base_dtype)
        
        # Move model to device and set eval mode
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not hasattr(model.base, "device") or next(model.base.parameters()).device.type != device.type:
            model = model.to(device)
        model.eval()
        
        self.model = model
        self.device = device
        self.prepared = []

    def build_prompt(self, instruction, input, response1, response2):
        """Build prompt using the same template as training."""
        question = instruction
        if input:
            question = f"{instruction}\n{input}"
        return build_pair_prompt(question, response1, response2)

    def preprocess_input(self, instruction, input, response1, response2):
        """Preprocess input and store tokenized result."""
        prompt = self.build_prompt(instruction, input, response1, response2)
        tok = self.tokenizer(
            prompt,
            truncation=True,
            max_length=2048,
            padding="max_length",
            return_tensors="pt",
        )
        self.prepared.append({
            "input_ids": tok.input_ids,
            "attention_mask": tok.attention_mask,
            "instruction": instruction,
            "input": input,
            "response1": response1,
            "response2": response2,
        })

    def inference(self, debias=True):
        """Run inference on prepared inputs."""
        results = []
        
        for idx in tqdm(range(len(self.prepared))):
            item = self.prepared[idx]
            input_ids = item["input_ids"].to(self.device)
            attention_mask = item["attention_mask"].to(self.device)
            
            with torch.no_grad():
                logit = self.model(input_ids, attention_mask).item()
            
            if debias:
                # Also compute reverse order for debiasing
                prompt_reverse = self.build_prompt(
                    item.get("instruction", ""),
                    item.get("input", ""),
                    item["response2"],
                    item["response1"],
                )
                tok_reverse = self.tokenizer(
                    prompt_reverse,
                    truncation=True,
                    max_length=2048,
                    padding="max_length",
                    return_tensors="pt",
                )
                input_ids_rev = tok_reverse.input_ids.to(self.device)
                attention_mask_rev = tok_reverse.attention_mask.to(self.device)
                
                with torch.no_grad():
                    logit_reverse = self.model(input_ids_rev, attention_mask_rev).item()
                
                # Average: if original says A>B (logit>0) and reverse says B>A (logit_reverse<0),
                # then final = (logit - logit_reverse) / 2
                # If logit > 0, A is better; if logit < 0, B is better
                # After reverse, if logit_reverse > 0, B is better in reverse order
                # So debiased: (logit - logit_reverse) / 2
                debiased_logit = (logit - logit_reverse) / 2.0
                prob = torch.sigmoid(torch.tensor(debiased_logit)).item()
            else:
                prob = torch.sigmoid(torch.tensor(logit)).item()
                debiased_logit = logit
            
            # Result: 1 if response1 is better, 2 if response2 is better, 0 if tie
            if prob > 0.5:
                result = 1  # Response A (response1) is better
            elif prob < 0.5:
                result = 2  # Response B (response2) is better
            else:
                result = 0  # Tie
            
            results.append({
                "logit": debiased_logit if debias else logit,
                "prob": prob,
                "result": result,
            })
        
        return results


class PandaLMBatchInferenceProvider(object):
    """
    Evaluate batch responses with PandaLM
    """

    def __init__(self, model_path) -> None:
        super().__init__()
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        except:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=False,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
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

        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)

        self.model = model
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
        prompt = self.build_judgelrm_prompt(instruction, input, response1, response2)
        self.prepared.append(self.tokenizer(prompt, return_tensors="pt", padding=True))

    def postprocess_output(self, text):
        # print(text)
        try:
            text = text.strip().split("<|im_start|>assistant")[1].strip().split("</think>")[1].strip()
            self.pattern.sub("", text.strip()).strip()
        except:
            print("bad output:",text)
            text = "<answer>0</answer><answer>0</answer>"
        
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

        return generated


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO
    )

    parser = argparse.ArgumentParser(description="Batch inference script for JudgeLM models")
    parser.add_argument("-s", "--seed", type=int, default=2023)
    parser.add_argument("-m", "--model_name", default="/shared/hdd/nuochen/models/judge_7B_step1000")
    parser.add_argument(
        "-i",
        "--input_path",
        default="/shared/hdd/nuochen/Logic-RL/testset-v1_update.json",
    )
    parser.add_argument("-o", "--output_path", default="/shared/hdd/nuochen/PandaLM/data/results_jugdelrm7b.json")
    parser.add_argument("--model_type", type=str, default="judgelrm", choices=["judgelrm", "cross_bt"],
                        help="Model type: 'judgelrm' for JudgeLRM (generative), 'cross_bt' for Cross-Encoder BT")
    parser.add_argument("--base_model_path", type=str, default=None,
                        help="Base model path for cross_bt (if different from checkpoint)")
    parser.add_argument("--debias", action="store_true", default=True,
                        help="Use debiasing (average both orderings) for cross_bt")

    args = parser.parse_args()

    logging.info(args)

    seed_everything(args.seed)

    logging.info(f"Loading {args.model_type} model from {args.model_name}")
    
    if args.model_type == "cross_bt":
        handler = CrossBTInferenceProvider(
            model_path=args.model_name,
            base_model_path=args.base_model_path,
            debias=args.debias,
        )
    else:
        handler = PandaLMBatchInferenceProvider(
            model_path=args.model_name,
        )
    
    with open(args.input_path) as f:
        input_data = json.load(f)

    results = []
    for item in tqdm(input_data):
        handler.preprocess_input(
            instruction=item.get("instruction", ""),
            input=item.get("input", ""),
            response1=item["response1"],
            response2=item["response2"],
        )
    
    if args.model_type == "cross_bt":
        generated = handler.inference(debias=args.debias)
        for idx, item in enumerate(input_data):
            item["output"] = f"logit={generated[idx]['logit']:.4f}, prob={generated[idx]['prob']:.4f}"
            item["result"] = generated[idx]["result"]
            item["logit"] = generated[idx]["logit"]
            item["prob"] = generated[idx]["prob"]
            results.append(item)
    else:
        generated = handler.inference()
        for idx, item in enumerate(input_data):
            item["output"] = generated[idx]
            item["result"] = handler.parse_judge_lrm_response(generated[idx])
            results.append(item)

    if args.output_path:
        with open(args.output_path, "w") as f:
            json.dump(results, f, indent=2)
        logging.info(f"Results saved to {args.output_path}")
    else:
        print(results)
