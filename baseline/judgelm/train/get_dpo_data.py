import json
from pathlib import Path
import sys

# Add root path to use absolute import
file = Path(__file__).resolve()
root = file.parents[2]
sys.path.append(str(root))

from judgelm.utils import jlload
def validate_dpo_data(dpo_data):
    """验证DPO数据的质量"""
    print("\nValidating DPO data...")
    total = len(dpo_data)
    valid = 0
    invalid = 0
    empty_chosen = 0
    empty_rejected = 0
    invalid_scores = 0
    
    for i, item in enumerate(dpo_data):
        
        if not isinstance(item, dict) or 'chosen' not in item or 'rejected' not in item or 'chosen_score' not in item or 'rejected_score' not in item:
            print(f"Item {i}: Invalid structure")
            invalid += 1
            continue
            
        chosen = item['chosen'][0]
        rejected = item['rejected'][0]
        
        # Validate chosen
        if not chosen.get('question') or not chosen.get('answer'):
            print(f"Item {i}: Empty chosen question or answer")
            empty_chosen += 1
            invalid += 1
            continue
            
        
        if not rejected.get('question') or not rejected.get('answer'):
            print(f"Item {i}: Empty rejected question or answer")
            empty_rejected += 1
            invalid += 1
            continue
            
        
        if not isinstance(item.get('chosen_score'), (int, float)) or not isinstance(item.get('rejected_score'), (int, float)):
            print(f"Item {i}: Invalid scores")
            invalid_scores += 1
            invalid += 1
            continue
            
        valid += 1
    
    print("\nDPO Data Validation Results:")
    print(f"Total items: {total}")
    print(f"Valid items: {valid}")
    print(f"Invalid items: {invalid}")
    print(f"Empty chosen: {empty_chosen}")
    print(f"Empty rejected: {empty_rejected}")
    print(f"Invalid scores: {invalid_scores}")
    
    return valid == total

def convert_to_dpo_format(input_file, output_file):
    """Convert JudgeLM format to DPO format"""
    print("Loading data...")
    data = jlload(input_file)
    
    dpo_data = []
    error_count = 0
    empty_question_count = 0
    empty_answer_count = 0
    invalid_score_count = 0
    
    for i, item in enumerate(data):
        
        if i % 1000 == 0:
            print(f"Processing item {i}/{len(data)}")
        
        
        if not isinstance(item, dict):
            print(f"Item {i}: Not a dictionary")
            error_count += 1
            continue
            
        
        required_fields = ['question_body', 'answer1_body', 'answer2_body', 'score']
        missing_fields = [field for field in required_fields if field not in item]
        if missing_fields:
            print(f"Item {i}: Missing fields: {missing_fields}")
            error_count += 1
            continue
            
        
        if not item['question_body'].strip():
            print(f"Item {i}: Empty question")
            empty_question_count += 1
            continue
            
        if not item['answer1_body'].strip():
            print(f"Item {i}: Empty answer1")
            empty_answer_count += 1
            continue
            
        if not item['answer2_body'].strip():
            print(f"Item {i}: Empty answer2")
            empty_answer_count += 1
            continue
            
        
        scores = item['score']
        if not isinstance(scores, list) or len(scores) != 2:
            print(f"Item {i}: Invalid scores format: {scores}")
            invalid_score_count += 1
            continue
            
        if not all(isinstance(s, (int, float)) for s in scores):
            print(f"Item {i}: Non-numeric scores: {scores}")
            invalid_score_count += 1
            continue
            
        
        if scores[0] > scores[1]:
            dpo_item = {
                "chosen": [{
                    "question": item['question_body'].strip(),
                    "answer": item['answer1_body'].strip()
                }],
                "rejected": [{
                    "question": item['question_body'].strip(),
                    "answer": item['answer2_body'].strip()
                }],
                "chosen_score": scores[0],
                "rejected_score": scores[1]
            }
        else:
            dpo_item = {
                "chosen": [{
                    "question": item['question_body'].strip(),
                    "answer": item['answer2_body'].strip()
                }],
                "rejected": [{
                    "question": item['question_body'].strip(),
                    "answer": item['answer1_body'].strip()
                }],
                "chosen_score": scores[1],
                "rejected_score": scores[0]
            }
        
        
        if not dpo_item['chosen'][0]['question'] or not dpo_item['chosen'][0]['answer']:
            print(f"Item {i}: Empty chosen question or answer after processing")
            continue
            
        if not dpo_item['rejected'][0]['question'] or not dpo_item['rejected'][0]['answer']:
            print(f"Item {i}: Empty rejected question or answer after processing")
            continue
            
        dpo_data.append(dpo_item)
    
    
    print("\nData Processing Statistics:")
    print(f"Total items processed: {len(data)}")
    print(f"Valid DPO items: {len(dpo_data)}")
    print(f"Error items: {error_count}")
    print(f"Empty questions: {empty_question_count}")
    print(f"Empty answers: {empty_answer_count}")
    print(f"Invalid scores: {invalid_score_count}")
    
    
    if not validate_dpo_data(dpo_data):
        print("\nWarning: Some DPO data items are invalid!")
        return
    
    
    print(f"\nSaving {len(dpo_data)} examples to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in dpo_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Saved DPO format data to {output_file}")

if __name__ == "__main__":
    input_file = "/user/datasets/JudgeLM-100K/judgelm_train_100k.jsonl"
    output_file = "/user/datasets/JudgeLM-100K/judgelm_train_100k_dpo.jsonl"
    convert_to_dpo_format(input_file, output_file)