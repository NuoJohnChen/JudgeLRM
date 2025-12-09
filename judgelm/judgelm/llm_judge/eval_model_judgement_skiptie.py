import argparse
import json
import os
import time

import shortuuid
import torch
from tqdm import tqdm

import sys
from pathlib import Path  # if you haven't already done so

file = Path(__file__).resolve()
root = file.parents[2]
sys.path.append(str(root))
print(sys.path)

from judgelm.utils import extract_jsonl


from judgelm.llm_judge.common import parse_score, translate_score_to_win_list


def filt_pred_score_list_by_gt_score_list(gt_score_list, pred_score_list):
    new_gt_score_list = []
    new_pred_score_list = []
    # filter [-1, -1] pairs
    for gt_score, pred_score in zip(gt_score_list, pred_score_list):
        if gt_score[0] == -1 or gt_score[1] == -1:
            continue
        else:
            new_gt_score_list.append(gt_score)
            new_pred_score_list.append(pred_score)

    return new_gt_score_list, new_pred_score_list


def calculate_metrics(gt_answer_file_path, sequential_pred_answer_file_path, reversed_pred_answer_file_path, if_filter_minus_one=True, skip_gt_draws=True, count_draws_correct=False):
    # get file list
    gt_answer_file_list = extract_jsonl(gt_answer_file_path)  # [:1000]
    # check if sequential_pred_answer_file_path is `str`
    if isinstance(sequential_pred_answer_file_path, str):
        sequential_pred_answer_file_list = extract_jsonl(sequential_pred_answer_file_path)  # [:1000]
    elif (isinstance(sequential_pred_answer_file_path, list)):
        sequential_pred_answer_file_list = sequential_pred_answer_file_path
    else:
        pass

    gt_score_list = []
    for gt_answer_file in gt_answer_file_list:
        gt_score_list.append(parse_score(gt_answer_file['text']))
    print(
        "===============================================================================================================")
    sequential_pred_score_list = []
    for sequential_pred_answer_file in sequential_pred_answer_file_list:
        sequential_pred_score_list.append(parse_score(sequential_pred_answer_file['pred_text']))

    gt_score_list_contains_minus_one = gt_score_list.copy()

    if if_filter_minus_one:
        # filter pred by gt
        gt_score_list, sequential_pred_score_list = filt_pred_score_list_by_gt_score_list(gt_score_list,
                                                                                      sequential_pred_score_list)

    # win_list calculate v2
    # if the score gap is less than T, we consider it as a draw
    T = 0.0
    gt_win_list = translate_score_to_win_list(gt_score_list, T)
    sequential_pred_win_list = translate_score_to_win_list(sequential_pred_score_list, T)

    # Count draws on GT after removing -1 pairs, before draw skipping
    gt_total_after_minus1 = len(gt_win_list)
    gt_draws_after_minus1 = sum(1 for w in gt_win_list if w == 0)

    # Will hold indices of non-draw GT items to reuse for reversed predictions
    keep_indices_draws = None

    # Optionally remove items where GT indicates a draw (0)
    if skip_gt_draws and not count_draws_correct:
        keep_indices_draws = [idx for idx, w in enumerate(gt_win_list) if w != 0]
        gt_win_list = [gt_win_list[idx] for idx in keep_indices_draws]
        sequential_pred_win_list = [sequential_pred_win_list[idx] for idx in keep_indices_draws]

    # sklearn.metrics (compute AFTER any draw handling, so flags take effect)
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    if isinstance(reversed_pred_answer_file_path, str):
        reversed_pred_answer_file_list = extract_jsonl(reversed_pred_answer_file_path)
    elif (isinstance(reversed_pred_answer_file_path, list)):
        reversed_pred_answer_file_list = reversed_pred_answer_file_path
    else:
        pass

    reversed_pred_score_list = []
    for reversed_pred_answer_file in reversed_pred_answer_file_list:
        reversed_pred_score_list.append(parse_score(reversed_pred_answer_file['pred_text']))
    _, reversed_pred_score_list = filt_pred_score_list_by_gt_score_list(gt_score_list_contains_minus_one,
                                                                        reversed_pred_score_list)
    reversed_pred_win_list = translate_score_to_win_list(reversed_pred_score_list, T)

    # Apply the same draw-removal mask for reversed predictions
    if skip_gt_draws and not count_draws_correct and keep_indices_draws is not None:
        reversed_pred_win_list = [reversed_pred_win_list[idx] for idx in keep_indices_draws]

    # If counting draws as correct, force predictions to draw (0) where GT is draw
    if count_draws_correct:
        draw_indices = [idx for idx, w in enumerate(gt_win_list) if w == 0]
        for idx in draw_indices:
            sequential_pred_win_list[idx] = 0
            reversed_pred_win_list[idx] = 0

    # Compute classification metrics AFTER handling draws and any skipping
    y_true = gt_win_list
    y_pred = sequential_pred_win_list
    if len(y_true) == 0:
        accuracy = 0.0
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    else:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    metrics_dict = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'gt_total_after_minus1': gt_total_after_minus1,
        'gt_draws_after_minus1': gt_draws_after_minus1
    }

    # from the perspect of before one
    # 1: win, -1: lose, 0: draw

    same = 0
    perfer_before = 0
    perfer_after = 0
    for i in range(len(sequential_pred_win_list)):
        if sequential_pred_win_list[i] == 1:
            sequential_pred_win_list[i] = -1
        elif sequential_pred_win_list[i] == -1:
            sequential_pred_win_list[i] = 1
        else:
            pass
        if sequential_pred_win_list[i] == reversed_pred_win_list[i]:
            same += 1
        elif sequential_pred_win_list[i] - reversed_pred_win_list[i] > 0:  # 1 & 0， 1 & -1， 0 & -1
            perfer_after += 1
            # print(i)
        elif sequential_pred_win_list[i] - reversed_pred_win_list[i] < 0:  # -1 & 0， -1 & 1， 0 & 1
            perfer_before += 1
            # print(i)
        else:
            pass

    # add metrics to dict
    denom = len(sequential_pred_win_list) if len(sequential_pred_win_list) > 0 else 1
    metrics_dict['consistency'] = same / denom
    metrics_dict['perfer_before_rate'] = perfer_before / denom
    metrics_dict['perfer_after_rate'] = perfer_after / denom
    metrics_dict['delta_bias'] = abs(perfer_before - perfer_after) / denom
    metrics_dict['total_bias'] = (perfer_before + perfer_after) / denom
    metrics_dict['total_num'] = len(sequential_pred_win_list)
    metrics_dict['kept_after_skip_draws'] = len(sequential_pred_win_list)
    metrics_dict['removed_draws'] = gt_draws_after_minus1 if skip_gt_draws else 0

    return metrics_dict


if __name__ == '__main__':
    # gt w/o ref
    gt_answer_file_path = "/share/project/lianghuizhu/JudgeLM-Project/JudgeLM/judgelm/data/JudgeLM/judgelm_val_5k_gpt4.jsonl"
    # gt w/ ref
    # gt_answer_file_path = "/share/project/lianghuizhu/JudgeLM-Project/JudgeLM/judgelm/data/JudgeLM/judgelm_val_5k_gpt4_with_reference.jsonl"

    # 33b 100k full model lr 3e-5
    sequential_pred_answer_file_path = "/share/project/lianghuizhu/JudgeLM-Project/JudgeLM/judgements_output/JudgeLM/7b-full-model-pycharm-debug-v2"
    #
    # 33b 100k full model lr 3e-5
    reversed_pred_answer_file_path = "/share/project/lianghuizhu/JudgeLM-Project/JudgeLM/judgements_output/JudgeLM/7b-full-model-pycharm-debug-reverse-v2"

    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-answer-file-path", type=str, default=gt_answer_file_path)
    parser.add_argument("--sequential-pred-answer-file-path", type=str, default=sequential_pred_answer_file_path)
    parser.add_argument("--reversed-pred-answer-file-path", type=str, default=reversed_pred_answer_file_path)
    parser.add_argument("--save-path", type=str, default=None, help="Path to save metrics dict as JSON")
    skip_draws_group = parser.add_mutually_exclusive_group()
    skip_draws_group.add_argument("--skip-gt-draws", dest="skip_gt_draws", action="store_true", help="Exclude ground-truth draws from evaluation")
    skip_draws_group.add_argument("--include-gt-draws", dest="skip_gt_draws", action="store_false", help="Include ground-truth draws in evaluation (default)")
    parser.set_defaults(skip_gt_draws=False)
    parser.add_argument("--count-draws-correct", action="store_true", help="Treat GT draws as correct by forcing predictions to draw on those items")

    args = parser.parse_args()

    metrics_dict = calculate_metrics(args.gt_answer_file_path, args.sequential_pred_answer_file_path,
                                     args.reversed_pred_answer_file_path, skip_gt_draws=args.skip_gt_draws,
                                     count_draws_correct=args.count_draws_correct)

    print(metrics_dict)

    if args.save_path is not None:
        with open(args.save_path, 'w') as f:
            json.dump(metrics_dict, f, indent=4)
        print(f"Metrics saved to {args.save_path}")
