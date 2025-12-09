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
    
    # First load base model config and structure
    config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Find all FSDP checkpoint files
    fsdp_files = [f for f in os.listdir(model_path) if f.startswith("model_world_size_")]
    fsdp_files.sort()
    
    print(f"Found FSDP files: {fsdp_files}")
    
    # Load and merge weights
    state_dict = {}
    for fsdp_file in fsdp_files:
        fsdp_path = os.path.join(model_path, fsdp_file)
        print(f"Loading {fsdp_file}...")
        # Use weights_only=False to load checkpoints containing DTensor
        checkpoint = torch.load(fsdp_path, map_location="cpu", weights_only=False)
        
        # Process weights in FSDP checkpoint
        for key, value in checkpoint.items():
            # Skip non-tensor data
            if not isinstance(value, torch.Tensor):
                continue
                
            # If distributed tensor, try to extract local tensor
            if hasattr(value, '_local_tensor'):
                value = value._local_tensor
            elif hasattr(value, 'to_local'):
                value = value.to_local()
            
            # Merge weights
            if key in state_dict:
                # If key exists, merge (assuming sharded weights)
                if isinstance(value, torch.Tensor) and isinstance(state_dict[key], torch.Tensor):
                    # Simple concat; real merge logic may need more care
                    try:
                        state_dict[key] = torch.cat([state_dict[key], value], dim=0)
                    except:
                        # If concat fails, fall back to latest value
                        state_dict[key] = value
                else:
                    state_dict[key] = value
            else:
                state_dict[key] = value
    
    # Load merged weights into model
    print("Loading merged weights into model...")
    model.load_state_dict(state_dict, strict=False)
    
    # Ensure model on GPU
    if torch.cuda.is_available():
        print(f"Move model to GPU device: {torch.cuda.current_device()}")
        model = model.cuda()
    else:
        print("Warning: CUDA not available, model will run on CPU")
    
    return model


class PandaLMBatchInferenceProvider(object):
    """
    Evaluate batch responses with PandaLM
    """

    def __init__(self, model_path, base_model_path=None) -> None:
        super().__init__()
        
        # Support checkpoint-style model loading
        if "huggingface" in model_path:
            # If huggingface subdir, use directly
            tokenizer_path = model_path
            model_path_for_tokenizer = model_path
        else:
            # If checkpoint root, try to find huggingface subdir
            import os
            huggingface_path = os.path.join(model_path, "huggingface")
            if os.path.exists(huggingface_path):
                # Check for FSDP checkpoint
                fsdp_files = [f for f in os.listdir(model_path) if f.startswith("model_world_size_")]
                if fsdp_files:
                    print(f"Detected FSDP checkpoint, model files: {fsdp_files}")
                    if base_model_path and os.path.exists(base_model_path):
                        print(f"Using base model as tokenizer: {base_model_path}")
                        print(f"Using trained model: {model_path}")
                        tokenizer_path = base_model_path  # tokenizer uses base model
                        model_path_for_tokenizer = model_path  # model uses trained checkpoint
                    else:
                        print("Error: FSDP checkpoint detected but no valid base model path provided")
                        raise ValueError("FSDP checkpoint requires --base to specify base model path")
                else:
                    tokenizer_path = huggingface_path
                    model_path_for_tokenizer = huggingface_path
            else:
                tokenizer_path = model_path
                model_path_for_tokenizer = model_path
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        except:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False, trust_remote_code=True)
        
        # Check if this is an FSDP checkpoint
        fsdp_files = [f for f in os.listdir(model_path) if f.startswith("model_world_size_")]
        if fsdp_files and base_model_path:
            print(f"Detected FSDP checkpoint, using custom load method")
            model = load_fsdp_checkpoint(model_path, base_model_path)
        else:
            # Determine device
            if torch.cuda.is_available():
                device = f"cuda:{torch.cuda.current_device()}"
                print(f"Using device: {device}")
            else:
                device = "cpu"
                print("Warning: CUDA unavailable, using CPU")
            
            model = AutoModelForCausalLM.from_pretrained(
                model_path_for_tokenizer,
                load_in_8bit=False,
                torch_dtype=torch.bfloat16,
                device_map=device if torch.cuda.is_available() else None,
                trust_remote_code=True,  # enable trust_remote_code
            )
            
            # If device_map fails, explicitly move to GPU
            if torch.cuda.is_available():
                if hasattr(model, 'parameters'):
                    model_device = next(model.parameters()).device
                    if model_device.type == 'cpu':
                        print(f"Model detected on CPU, explicitly moving to GPU: {device}")
                        model = model.to(device)
        
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

        # Ensure model on GPU before torch.compile
        if torch.cuda.is_available() and hasattr(model, 'parameters'):
            model_device = next(model.parameters()).device
            if model_device.type == 'cpu':
                device = f"cuda:{torch.cuda.current_device()}"
                print(f"Moving model to GPU before torch.compile: {device}")
                model = model.to(device)

        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)
            # After torch.compile, check device again
            if torch.cuda.is_available() and hasattr(model, 'parameters'):
                model_device = next(model.parameters()).device
                if model_device.type == 'cpu':
                    device = f"cuda:{torch.cuda.current_device()}"
                    print(f"Model on CPU after torch.compile; moving back to GPU: {device}")
                    model = model.to(device)

        self.model = model
        self.prepared = []
        self.pattern = re.compile(
            r"<unk>|<pad>|<s>|</s>|\[PAD\]|<\|endoftext\|>|\[UNK\]|\[CLS\]|\[MASK\]|<\|startofpiece\|>|<\|endofpiece\|>|\[gMASK\]|\[sMASK\]"
        )
        
        # Check model device
        if hasattr(self.model, 'parameters'):
            model_device = next(self.model.parameters()).device
            print(f"Model loaded on device: {model_device}")
            if torch.cuda.is_available() and model_device.type == 'cpu':
                print("Warning: Model on CPU while CUDA available; inference may be slow.")
                # Last attempt to move to GPU
                device = f"cuda:{torch.cuda.current_device()}"
                print(f"Attempting to move model to GPU: {device}")
                self.model = self.model.to(device)
                model_device = next(self.model.parameters()).device
                print(f"Device after move: {model_device}")

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
        import re
        try:
            # Keep full assistant output, including <think> block
            if "<|im_start|>assistant" in text:
                assistant_part = text.split("<|im_start|>assistant", 1)[1]
            else:
                assistant_part = text

            # Trim to <|im_end|> or end of text
            if "<|im_end|>" in assistant_part:
                assistant_part = assistant_part.split("<|im_end|>", 1)[0]

            assistant_part = assistant_part.strip()

            # Ensure at least two <answer> tags; otherwise return default
            answers = re.findall(r'<answer>\s*(\d+)\s*</answer>', assistant_part)
            if len(answers) < 2:
                print("bad output: did not find two <answer> tags, text:", text[:500])
                return "<answer>0</answer><answer>0</answer>"

            return assistant_part
        except Exception as e:
            print(f"bad output: extraction error {e}, text: {text[:500]}")
            return "<answer>0</answer><answer>0</answer>"

    def filter_special_token(self, text):
        return self.pattern.sub("", text.strip()).strip()

    def inference(
        self,
        temperature=0.1,
        top_p=1,
        top_k=1,
        num_beams=4,
        max_new_tokens=3072,
        repetition_penalty=1.2,
    ):
        generated = []

        print(f"\n{'='*100}")
        print(f"Starting inference on {len(self.prepared)} samples")
        print(f"Using GPU: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU device: {torch.cuda.get_device_name()}")
            print(f"Model device: {next(self.model.parameters()).device}")
            print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"{'='*100}")

        
        if hasattr(self.model, 'device'):
            model_device = self.model.device
        elif hasattr(self.model, 'parameters'):
            model_device = next(self.model.parameters()).device
        else:
            model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        for idx in tqdm(range(len(self.prepared))):
            inputs = self.prepared[idx]
            input_ids = inputs["input_ids"].to(model_device)
            generation_config = GenerationConfig(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_beams=num_beams,
                early_stopping=True,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                stop_strings=["<|im_end|>"],
            )
            with torch.no_grad():
                generation_output = self.model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    tokenizer=self.tokenizer,
                )

            for j in range(len(generation_output.sequences)):
                s = generation_output.sequences[j]
                output = self.tokenizer.decode(s)
                
                # For first 10 items, print raw output for debugging
                if idx < 10:
                    print(f"\n=== Raw output #{idx + 1} ===")
                    print(output)
                    print("-" * 80)
                
                resp = self.postprocess_output(output)
                resp = self.filter_special_token(resp)
                generated.append(resp)
                
                # For first 10 items, print parsed result
                if idx < 10:
                    parsed_result = self.parse_judge_lrm_response(resp)
                    print(f"\n=== Parsed result #{idx + 1} ===")
                    print(f"Extracted Output: {resp}")
                    print(f"Parsed Result: {parsed_result} (1=Response1 better, 2=Response2 better, 0=tie)")
                    print("-" * 80)

        return generated


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO
    )

    parser = argparse.ArgumentParser(description="PandaLM batch inference script")
    parser.add_argument("-s", "--seed", type=int, default=2023)
    parser.add_argument("-m", "--model_name", default="/shared/hdd/nuochen/models/GRPO_logic_KK_99/Qwen3-4B/global_step_400/actor")
    parser.add_argument(
        "-i",
        "--input_path",
        default="/shared/hdd/nuochen/Logic-RL/testset-v1_update.json",#"/shared/hdd/nuochen/Logic-RL/testset_with_responses_pandlm_r1_back.json",
    )
    parser.add_argument("-o", "--output_path", default="/shared/hdd/nuochen/PandaLM/data/results_jugdelrm7b.json")
    parser.add_argument("--base", default="/shared/hdd/nuochen/models/Qwen3-4B", help="Base model path for FSDP checkpoint loading")

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
        output = handler.preprocess_input(
            instruction=item["instruction"],
            input=item["input"],
            response1=item["response1"],
            response2=item["response2"],
        )
    generated = handler.inference()
    for idx, item in enumerate(input_data):
        item["output"] = generated[idx]  # assign generated[idx] to "output"
        item["result"] = handler.parse_judge_lrm_response(generated[idx])
        results.append(item)  # append modified item to results
        # results.append([item, generated[idx]])

    # print(results)
    if args.output_path:
        with open(args.output_path, "w") as f:
            json.dump(results, f)
    else:
        print(results)
