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
from vllm import LLM, SamplingParams
import torch.multiprocessing as mp

import logging

mp.set_start_method('spawn', force=True)

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
            
        
        self.model = LLM(
            model=model_path,
            dtype="bfloat16",
            tensor_parallel_size=torch.cuda.device_count(),
            gpu_memory_utilization=0.9
        )
        
        self.tokenizer = tokenizer
        self.prepared = []
        self.pattern = re.compile(
            r"<unk>|<pad>|<s>|</s>|\[PAD\]|<\|endoftext\|>|\[UNK\]|\[CLS\]|\[MASK\]|<\|startofpiece\|>|<\|endofpiece\|>|\[gMASK\]|\[sMASK\]"
        )

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
        prompt = self.build_pandalm_prompt(instruction, input, response1, response2)
        
        self.prepared.append(prompt)

    def postprocess_output(self, text):
        
        parts = text.strip().split("### Evaluation:")
        if len(parts) > 1:
            return parts[1].strip()
        
        return text.strip()

    def filter_special_token(self, text):
        return self.pattern.sub("", text.strip()).strip()

    def inference(
        self,
        temperature=0,
        top_p=1,
        top_k=1,
        num_beams=4,
        max_new_tokens=512,
        repetition_penalty=1.2,
    ):
        
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            stop_token_ids=[self.tokenizer.eos_token_id]
        )
        
        
        prompts = self.prepared
        
        
        outputs = self.model.generate(prompts, sampling_params)
        
        
        generated = []
        for idx, output in enumerate(outputs):
            text = output.outputs[0].text
            
            logging.debug(f"Input prompt: {prompts[idx]}")
            logging.debug(f"Raw model output: {text}")
            resp = self.postprocess_output(text)
            resp = self.filter_special_token(resp)
            generated.append(resp)
            
        
        self.prepared = []
        return generated


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO
    )

    parser = argparse.ArgumentParser(description="PandaLM batch inference script")
    parser.add_argument("-s", "--seed", type=int, default=2023)
    parser.add_argument("-m", "--model_name", default="/shared/hdd/nuochen/models/PandaLM-7B-v1")#"/shared/hdd/nuochen/models/JudgeLM-7B-v1.0")
    parser.add_argument(
        "-i",
        "--input_path",
        default="/shared/hdd/nuochen/Logic-RL/testset-v1_update.json",
    )
    parser.add_argument("-o", "--output_path", default="/shared/hdd/nuochen/PandaLM/data/results.json")

    args = parser.parse_args()

    logging.info(args)

    seed_everything(args.seed)

    logging.info(f"Loading model from {args.model_name}")
    handler = PandaLMBatchInferenceProvider(
        model_path=args.model_name,
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
        results.append([item, generated[idx]])

    print(results)
    if args.output_path:
        with open(args.output_path, "w") as f:
            json.dump(results, f)
    else:
        print(results)
