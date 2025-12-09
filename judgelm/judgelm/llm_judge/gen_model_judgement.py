"""Generate answers with local models.

"""
import argparse
import json
import os
import time
import re

import shortuuid
import torch
from tqdm import tqdm

import sys
from pathlib import Path # if you haven't already done so
file = Path(__file__).resolve()
root = file.parents[2]
sys.path.append(str(root))
print(sys.path)

from judgelm.llm_judge.common import load_questions, reorg_answer_file, conv_judge_pair, conv_judge_pair_w_reference, KeywordsStoppingCriteria, parse_score, translate_score_to_win_list
from judgelm.model import load_model
from judgelm.utils import extract_jsonl


def run_eval(
    model_path,
    model_id,
    question_file,
    question_begin,
    question_end,
    answer_file,
    max_new_token,
    num_gpus_per_model,
    num_gpus_total,
    max_gpu_memory,
    temperature,
    if_reverse_answers,
    reference_file,
    if_fast_eval,
    judge_format
):
    print("start run_eval")
    questions = load_questions(question_file, question_begin, question_end)
    if reference_file is not None:
        references = load_questions(reference_file, question_begin, question_end)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model) # // 2
    ans_handles = []
    print("start ans_handles append")
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                model_path,
                model_id,
                questions[i : i + chunk_size],
                answer_file,
                max_new_token,
                num_gpus_per_model,
                max_gpu_memory,
                temperature,
                if_reverse_answers,
                references[i : i + chunk_size] if reference_file is not None else None,
                if_fast_eval,
                judge_format,
            )
        )

    if use_ray:
        ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
    model_path,
    model_id,
    questions,
    answer_file,
    max_new_token,
    num_gpus_per_model,
    max_gpu_memory,
    temperature,
    if_reverse_answers,
    references,
    if_fast_eval,
    judge_format="judgelm",
):
    print("start load model")
    # Determine device from Ray-assigned GPU IDs if available
    device_str = "cuda"
    try:
        import ray
        ctx = ray.get_runtime_context()
        gpu_ids = ctx.get_gpu_ids() if hasattr(ctx, "get_gpu_ids") else []
        if gpu_ids:
            physical_id = int(gpu_ids[0])
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
            if visible:
                try:
                    visible_list = [int(x) for x in visible.split(",") if x != ""]
                    if physical_id in visible_list:
                        local_idx = visible_list.index(physical_id)
                        device_str = f"cuda:{local_idx}"
                    else:
                        # fallback to first visible device
                        device_str = "cuda:0"
                except Exception:
                    device_str = "cuda:0"
            else:
                # No masking; use physical id directly
                device_str = f"cuda:{physical_id}"
    except Exception:
        pass
    model, tokenizer = load_model(
        model_path,
        device=device_str,
        num_gpus=num_gpus_per_model,
        max_gpu_memory=max_gpu_memory,
        load_8bit=False,
        cpu_offloading=False,
        debug=False,
    )
    # Determine the model's primary device (works for single device or sharded device_map)
    try:
        model_first_device = next(model.parameters()).device
    except StopIteration:
        model_first_device = torch.device("cuda")

    for q_i, question in tqdm(enumerate(questions)):
        torch.manual_seed(q_i)
        conv = conv_judge_pair.copy() if references is None else conv_judge_pair_w_reference.copy()
        template = conv.prompt_template

        # if fast eval, use the "\n" as the separator (only for default format)
        if if_fast_eval and judge_format == "judgelm":
            conv.sep = "\n"

        # reverse the order of the answers
        if if_reverse_answers:
            temp_answer = question["answer1_body"]
            question["answer1_body"] = question["answer2_body"]
            question["answer2_body"] = temp_answer

        # combine data_sample
        if judge_format == "judgelm":
            if references is None:
                data_sample = conv.system + '\n' + template.format(question=question['question_body'],
                                                                   answer_1=question['answer1_body'],
                                                                   answer_2=question['answer2_body'],
                                                                   prompt=conv.prompt) + conv.appendix
            else:
                data_sample = conv.system + '\n' + template.format(question=question['question_body'],
                                                                   reference=references[q_i]['reference']['text'],
                                                                   answer_1=question['answer1_body'],
                                                                   answer_2=question['answer2_body'],
                                                                   prompt=conv.prompt) + conv.appendix
        else:
            # judgelrm format with <think> and <answer> tags
            user_block = f"[Question]\n{question['question_body']}\n\n[Assistant 1's Answer]\n{question['answer1_body']}\n\n[Assistant 2's Answer]\n{question['answer2_body']}"
            system_block = (
                "<|im_start|>system\n"
                "You are a helpful assistant. The assistant first performs a detailed, step-by-step reasoning process in its mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> detailed reasoning process here, explaining each step of your evaluation for both assistants </think><answer> answer here </answer>. Now the user asks you to judge the performance of two AI assistants in response to the question. Score assistants 1-10 (higher=better). Criteria includes helpfulness, relevance, accuracy, and level of detail. Avoid order, length, style or other bias. After thinking, when you finally reach a conclusion, clearly provide your evaluation scores within <answer> </answer> tags, i.e. for example,<answer>3</answer><answer>5</answer>\n"
                "<|im_end|>\n"
            )
            user_msg = f"<|im_start|>user\n{user_block}\n<|im_end|>\n"
            assistant_prefix = "<|im_start|>assistant\n<think>"
            data_sample = system_block + user_msg + assistant_prefix

        input_ids = tokenizer([data_sample]).input_ids
        input_ids[0][0] = 1

        do_sample = False if temperature < 1e-4 else True
        stopping_criteria = None if judge_format == "judgelrm" else KeywordsStoppingCriteria([conv.sep], tokenizer, torch.as_tensor(input_ids))

        # generate judgements
        output_ids = model.generate(
            torch.as_tensor(input_ids).to(model_first_device),
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_token,
            stopping_criteria=[stopping_criteria] if stopping_criteria is not None else None
        )

        if model.config.is_encoder_decoder:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0][len(input_ids[0]) :]

        output = tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            spaces_between_special_tokens=False,
        )

        if judge_format == "judgelm":
            if conv.sep:
                output = output[: output.find(conv.sep)]
            output = output.strip()
        else:
            # Extract two scores inside <answer> tags and format as "a b" on first line
            try:
                answers = re.findall(r"<answer>\s*([0-9]+(?:\.[0-9]+)?)\s*</answer>", output)
                if len(answers) >= 2:
                    output = f"{answers[0]} {answers[1]}"
                else:
                    # fallback: no valid answers found
                    output = "-1 -1"
            except Exception:
                output = "-1 -1"

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_id = shortuuid.uuid()
            question["pred_id"] = ans_id
            question["pred_text"] = output
            question["pred_model_id"] = model_id
            question["tstamp"] = time.time()
            if references is not None:
                question["reference"] = references[q_i]['reference']['text']
            fout.write(json.dumps(question) + "\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument(
        "--question-file",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=2048,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        # default="37Gib",
        help="Maxmum GPU memory used for model weights per GPU.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="The temperature for sampling.",
    )
    parser.add_argument(
        "--if-reverse-answers",
        type=int,
        default=0,
        help="Whether to reverse the order of the answers.",
    )
    parser.add_argument(
        "--reference-file",
        type=str,
        default=None,
        help="The reference file for evaluation.",
    )
    parser.add_argument(
        "--if-fast-eval",
        type=int,
        default=0,
        help="Whether to use fast evaluation.",
    )
    parser.add_argument(
        "--judge-format",
        type=str,
        default="judgelm",
        choices=["judgelm", "judgelrm"],
        help="Prompt/parse style: 'judgelm' original two-number first-line; 'judgelrm' uses <think>/<answer> and will be parsed to two numbers.",
    )
    args = parser.parse_args()
    args.if_reverse_answers = bool(args.if_reverse_answers)
    args.if_fast_eval = bool(args.if_fast_eval)
    if args.reference_file == 'None':
        args.reference_file = None
    print(f"args: {args}")

    # if use ray
    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray
        # Propagate CUDA_VISIBLE_DEVICES into Ray workers
        env_vars = {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "")}
        ray.init(runtime_env={
            "working_dir": str(root),
            "env_vars": env_vars,
        })

    print(f"Output to {args.answer_file}")
    
    run_eval(
        args.model_path,
        args.model_id,
        args.question_file,
        args.question_begin,
        args.question_end,
        args.answer_file,
        args.max_new_token,
        args.num_gpus_per_model,
        args.num_gpus_total,
        args.max_gpu_memory,
        args.temperature,
        args.if_reverse_answers,
        args.reference_file,
        args.if_fast_eval,
        args.judge_format
    )

    reorg_answer_file(args.answer_file)

    # statistics the judgements
    sequential_pred_answer_file_list = extract_jsonl(args.answer_file)

    sequential_pred_score_list = []
    for sequential_pred_answer_file in sequential_pred_answer_file_list:
        sequential_pred_score_list.append(parse_score(sequential_pred_answer_file['pred_text']))

    # if the score gap is less than T, we consider it as a draw
    T = 0.0
    sequential_pred_win_list = translate_score_to_win_list(sequential_pred_score_list, T)

    # get the number of 1 in sequential_pred_win_list
    win_num = sequential_pred_win_list.count(1)
    tie_num = sequential_pred_win_list.count(0)
    lose_num = sequential_pred_win_list.count(-1)

    # print the win, tie, and lose number, use format {}
    print("Assistant 1's reuslts ---> win_num: {}, tie_num: {}, lose_num: {}".format(win_num, tie_num, lose_num))
    print("Assistant 2's reuslts ---> win_num: {}, tie_num: {}, lose_num: {}".format(lose_num, tie_num, win_num))

