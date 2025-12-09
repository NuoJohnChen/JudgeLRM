import json
import os
from typing import Dict, List, Any

def convert_to_dpo_format(input_file: str, output_file: str):
    """
    将原始数据转换为DPO格式
    DPO格式要求：
    {
        "chosen": {
            "question": str,
            "answer": str,
            "score": float
        },
        "rejected": {
            "question": str,
            "answer": str,
            "score": float
        }
    }
    """
    print(f"Converting {input_file} to DPO format...")
    
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    
    total_items = 0
    valid_items = 0
    error_items = 0
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            total_items += 1
            try:
                item = json.loads(line)
                
                
                if not all(k in item for k in ['question_body', 'answer1_body', 'answer2_body', 'score_w_reference']):
                    print(f"Skipping item {total_items}: Missing required fields")
                    error_items += 1
                    continue
                
                
                scores = item['score_w_reference']
                if not isinstance(scores, list) or len(scores) != 2:
                    print(f"Skipping item {total_items}: Invalid score format")
                    error_items += 1
                    continue
                
                score1, score2 = scores
                
                # Determine chosen and rejected
                if score1 >= score2:
                    chosen = {
                        "question": item['question_body'],
                        "answer": item['answer1_body'],
                        "score": float(score1)
                    }
                    rejected = {
                        "question": item['question_body'],
                        "answer": item['answer2_body'],
                        "score": float(score2)
                    }
                else:
                    chosen = {
                        "question": item['question_body'],
                        "answer": item['answer2_body'],
                        "score": float(score2)
                    }
                    rejected = {
                        "question": item['question_body'],
                        "answer": item['answer1_body'],
                        "score": float(score1)
                    }
                
                
                if not all(chosen.values()) or not all(rejected.values()):
                    print(f"Skipping item {total_items}: Empty values in chosen or rejected")
                    error_items += 1
                    continue
                
                
                dpo_item = {
                    "chosen": chosen,
                    "rejected": rejected
                }
                f_out.write(json.dumps(dpo_item, ensure_ascii=False) + '\n')
                valid_items += 1
                
            except json.JSONDecodeError as e:
                print(f"Error parsing item {total_items}: {str(e)}")
                error_items += 1
                continue
            except Exception as e:
                print(f"Unexpected error processing item {total_items}: {str(e)}")
                error_items += 1
                continue
    
    print(f"\nConversion completed:")
    print(f"Total items processed: {total_items}")
    print(f"Valid items: {valid_items}")
    print(f"Error items: {error_items}")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    input_file = "/shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_train_100k.jsonl"
    output_file = "/shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_train_100k_dpo.jsonl"
    convert_to_dpo_format(input_file, output_file) 