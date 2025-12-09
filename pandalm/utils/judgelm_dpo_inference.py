import argparse
import torch
import transformers
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer
import json
import sys
import logging
from typing import Union, Dict
from tqdm import tqdm
import re, random
from common import *  

import logging

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


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
        ).to("cuda")  
        
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

    def build_judgelm_prompt(
        self, instruction, input, resp1, resp2, result=None, explain=None, ref=None
    ):
        resp1 = self.pattern.sub("", resp1.strip()).strip()
        resp2 = self.pattern.sub("", resp2.strip()).strip()
        conv = conv_judge_pair.copy()
        template = conv.prompt_template
        conv.sep = "\n"

        rsp = f"### Response 1:\n{resp1}\n\n### Response 2:\n{resp2}"

        if input:
            input_sequence = conv.system + '\n' + template.format(
                question=instruction + '\n' + input,
                answer_1=resp1,
                answer_2=resp2,
                prompt=conv.prompt
            ) + conv.appendix
        else:
            input_sequence = conv.system + '\n' + template.format(
                question=instruction,
                answer_1=resp1,
                answer_2=resp2,
                prompt=conv.prompt
            ) + conv.appendix
        


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
        
    def parse_judge_lm_response(self, text):
        # print(text)
        try:
            first_line = text.strip().split("\n")[0]
            nums = first_line.strip().split()
            if len(nums) >= 2:
                num1 = float(nums[0])
                num2 = float(nums[1])
                
                if num1 > num2:
                    return 1
                elif num1 < num2:
                    return 2
                else:
                    return 0
        except:
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
        prompt = self.build_judgelm_prompt(instruction, input, response1, response2)
        self.prepared.append(self.tokenizer(prompt, return_tensors="pt", padding=True))

    def postprocess_output(self, text):
        text = text.strip().split("### Response:")[1].strip()
        self.pattern.sub("", text.strip()).strip()
        return text

    def filter_special_token(self, text):
        return self.pattern.sub("", text.strip()).strip()

    def get_logprob(self, instruction, input, response):
        
        if input:
            input_text = instruction + "\n" + input
        else:
            input_text = instruction
            
        
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        response_inputs = self.tokenizer(response, return_tensors="pt").to(self.model.device)
        
        
        with torch.no_grad():
            
            outputs = self.model(**inputs)
            logits = outputs.logits.to(self.model.device)  
            
            
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            
            
            response_ids = response_inputs["input_ids"]
            
            
            response_log_probs = []
            
            max_len = min(log_probs.size(1), response_ids.size(1))
            for i in range(max_len):
                token_id = response_ids[0, i]
                token_log_prob = log_probs[0, i, token_id].item()
                response_log_probs.append(token_log_prob)
            
            
            avg_log_prob = sum(response_log_probs) / len(response_log_probs) if response_log_probs else 0.0
            
        return avg_log_prob

    def inference(self):
        generated = []
        
        for idx in tqdm(range(len(self.prepared))):
            item = self.prepared[idx]
            instruction = item["instruction"]
            input_text = item["input"]
            response1 = item["response1"]
            response2 = item["response2"]
            
            
            response1_logprob = self.get_logprob(instruction, input_text, response1)
            response2_logprob = self.get_logprob(instruction, input_text, response2)
            
            
            if response1_logprob > response2_logprob:
                result = 1
            elif response1_logprob < response2_logprob:
                result = 2
            else:
                result = 0
                
            
            output_dict = {
                "result": result,
                "response1_logprob": response1_logprob,
                "response2_logprob": response2_logprob
            }
            
            generated.append(output_dict)
            
        return generated


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO
    )

    parser = argparse.ArgumentParser(description="PandaLM batch inference script")
    parser.add_argument("-s", "--seed", type=int, default=2023)
    parser.add_argument("-m", "--model_name", default="/shared/hdd/nuochen/models/Qwen2.5-3B-Instruct")
    parser.add_argument(
        "-i",
        "--input_path",
        default="/shared/hdd/nuochen/Logic-RL/testset-v1_update.json",#"/shared/hdd/nuochen/Logic-RL/testset_with_responses_pandlm_r1_back.json",
    )
    parser.add_argument("-o", "--output_path", default="/shared/hdd/nuochen/PandaLM/data/results_jugdelm.json")

    args = parser.parse_args()

    logging.info(args)

    seed_everything(args.seed)

    logging.info(f"Loading model from {args.model_name}")
    handler = PandaLMBatchInferenceProvider(
        model_path=args.model_name,
    )
    
    
    logging.info(f"Model device: {handler.model.device}")
    logging.info(f"Model dtype: {handler.model.dtype}")

    with open(args.input_path) as f:
        input_data = json.load(f)

    results = []
    for item in tqdm(input_data):
        
        handler.prepared.append({
            "instruction": item["instruction"],
            "input": item["input"],
            "response1": item["response1"],
            "response2": item["response2"]
        })
        
    generated = handler.inference()
    
    for idx, item in enumerate(input_data):
        
        item["output"] = generated[idx]
        results.append(item)
    
    print(results)
    if args.output_path:
        with open(args.output_path, "w") as f:
            json.dump(results, f)
    else:
        print(results)
