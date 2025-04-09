import re
from typing import Dict, Tuple, Optional
import numpy as np

def extract_solution(solution_str: str) -> Tuple[Optional[list], str]:
    """Extracts the numbers from model's response string.
    
    Args:
        solution_str: Raw response string from the language model
        
    Returns:
        Tuple containing (extracted_numbers, processed_string)
    """
    # Split response to isolate assistant output
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        print("[Error] Failed to locate model response header")
        return None, solution_str

    # Extract numbers from answer
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    
    if len(matches) < 2:
        print("[Error] Less than 2 answer tags found")
        return None, processed_str
        
    try:
        # Extract both scores
        score_1 = int(matches[0].group(1).strip())
        score_2 = int(matches[1].group(1).strip())
        
        # Validate scores
        if not (0 <= score_1 <= 10) or not (0 <= score_2 <= 10):
            raise ValueError("Scores must be between 0 and 10")
            
        return [score_1, score_2], processed_str
    except Exception as e:
        print(f"  [Error] Invalid format in answer: {str(e)}")
        return None, processed_str

def parse_solution_text_format(solution_text: np.ndarray) -> list:
    """Parses ground truth solution text into expected numbers.
    
    Args:
        solution_text: Formatted solution text from dataset (as ndarray)
        
    Returns:
        List of two numbers from the first line
    """
    print("\n[Ground Truth Parsing]")
    
    try:
        solution_list = solution_text.tolist()
        
        if isinstance(solution_list, list) and len(solution_list) == 2:
            print(f"  Found expected scores: {solution_list}")
            return solution_list
            

        first_line = solution_list[0].split('\n')[0]
        expected_scores = list(map(int, first_line.split()))
        if len(expected_scores) != 2:
            raise ValueError
        print(f"  Found expected scores: {expected_scores}")
        return expected_scores
    except Exception as e:
        print(f"  [Error] Invalid format in solution_text_format: {e}")
        return None

def validate_response_structure(processed_str: str) -> float:
    """Performs comprehensive validation of response structure.
    
    Args:
        processed_str: Processed response string from the model
        
    Returns:
        Float score based on validation results
    """
    print("\n[Structure Validation]")
    
    # Check required tags
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
        'answer_start': ('<answer>', 2),
        'answer_end': ('</answer>', 2)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        
        print(f"  {tag_str}: count={count}, position={pos}")
        
        if count != expected_count:
            print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            return -1.0

    # Verify tag order
    if (positions['think_start'] > positions['think_end'] or
        positions['think_end'] > positions['answer_start'] or
        positions['answer_start'] > positions['answer_end']):
        print("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
        return -1.0
    else:
        print("  Tag sequence validation passed")

    # Validate answer content format
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    
    if len(matches) != 2:
        print("  [Error] Should have exactly 2 answer tags")
        return -1.0
        
    try:
        # Extract both scores
        score_1 = int(matches[0].group(1).strip())
        score_2 = int(matches[1].group(1).strip())
        
        # Validate scores
        if not (0 <= score_1 <= 10) or not (0 <= score_2 <= 10):
            print("  [Error] Scores must be between 0 and 10")
            return -0.5
            
        return 1.0
    except Exception as e:
        print(f"  [Error] Invalid format in answer: {str(e)}")
        return -0.5

def parse_model_answer(answer_text: list, expected_scores: list, reward_ablation: str = 'base_reward') -> float:
    """Validates model's answer against expected numbers.
    
    Args:
        answer_text: List of two numbers from model's answer
        expected_scores: List of two expected numbers
        
    Returns:
        Float score based on validation results
    """
    print("\n[Model Answer Validation]")
    try:

        expected_relation = expected_scores[0] >= expected_scores[1]
        # 检查模型答案的比较关系
        answer_relation = answer_text[0] >= answer_text[1]
        

        if expected_relation == answer_relation:
            print("  Content validation: FULL MATCH")
            score = 2.0
            

            if reward_ablation != 'reward_wo_score':
                abs_diff = abs(expected_scores[0]-answer_text[0]) + abs(expected_scores[1]-answer_text[1])
                
                if abs_diff == 0:
                    score += 1.0
                    print("  Perfect match: +1.0")
                elif abs_diff <= 2:
                    score += 0.6
                    print(f"  Close match (diff={abs_diff}): +0.6")
                    
                pred_diff = abs(answer_text[1]-answer_text[0])
                expected_diff = abs(expected_scores[1]-expected_scores[0])
                if pred_diff >= expected_diff:
                    score += 0.2
                    print(f"  Difference magnitude (pred={pred_diff}, expected={expected_diff}): +0.2")
            return score
        else:
            print("  Content validation: MISMATCH")
            return -1.5
    except:
        print("  [Error] Invalid answer format")
        return -2.0

def compute_score(solution_str: str, 
                 ground_truth: Dict[str, str],
                 format_reward: float = 1.0,
                 answer_reward: float = 1.0,
                 reward_ablation: str = 'base_reward',
                 response_length: int = 0,
                 max_response_length: int = 0):
    """Computes comprehensive score for model response."""
    print("\n" + "="*80)
    print(" Processing New Sample ".center(80, '='))
    
    # Parse ground truth data
    solution_text = ground_truth.get('solution_text_format', np.array([]))
    expected_scores = parse_solution_text_format(solution_text)
    if expected_scores is None:
        return -2.0

    # Extract model answer
    answer_dict, processed_str = extract_solution(solution_str)
    print(f"\n[Model Response]\n{processed_str}")

    # Validate response structure
    format_score = validate_response_structure(processed_str) * format_reward
    print(f"\n  Format validation score: {format_score}")

    # Initialize answer score
    answer_score = 0.0

    if format_score > 0 and answer_dict:
        expected_relation = expected_scores[0] >= expected_scores[1]
        predicted_relation = answer_dict[0] >= answer_dict[1]
        
        print(f"\n[Content Validation]")
        print(f"  Expected: {expected_scores[0]} {'>=' if expected_relation else '<'} {expected_scores[1]}")
        print(f"  Predicted: {answer_dict[0]} {'>=' if predicted_relation else '<'} {answer_dict[1]}")
        

        answer_score = parse_model_answer(answer_dict, expected_scores, reward_ablation) * answer_reward

        if reward_ablation == 'base_reward':
            length_bonus = 0.2 if response_length > 120 else 0.0
            length_penalty = -1.0 if response_length >= max_response_length else 0.0
            answer_score +=  length_bonus + length_penalty
        elif reward_ablation == 'reward_wo_length':
            pass
    else:
        answer_score = -2.0
        print("\n[Content Validation] Skipped due to format errors or missing answer")

    total_score = format_score + answer_score
    print("\n" + "-"*80)
    print(f" Final Score ".center(80, '-'))
    print(f"  Format: {format_score}")
    print(f"  Answer: {answer_score}")
    print(f"  Total: {total_score}")
    print("="*80 + "\n")

    return total_score

