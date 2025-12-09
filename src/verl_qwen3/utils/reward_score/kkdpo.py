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
        # If no standard header found, use the entire string
        print("[Warning] No standard header found, using entire response")
        processed_str = solution_str

    # Extract numbers from answer
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    
    print(f"[Debug] Found {len(matches)} answer tags")
    for i, match in enumerate(matches):
        print(f"  Match {i}: '{match.group(1).strip()}'")
    
    if len(matches) < 2:
        print("[Error] Less than 2 answer tags found")
        return None, processed_str
        
    try:
        # Extract both scores as float
        score_1 = float(matches[0].group(1).strip())
        score_2 = float(matches[1].group(1).strip())

        # Validate scores (float range)
        if not (0.0 <= score_1 <= 10.0) or not (0.0 <= score_2 <= 10.0):
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
        # Normalize to Python list
        if isinstance(solution_text, np.ndarray):
            solution_list = solution_text.tolist()
        elif isinstance(solution_text, list):
            solution_list = solution_text
        elif isinstance(solution_text, str):
            solution_list = [solution_text]
        else:
            print(f"  [Error] Unsupported solution_text_format type: {type(solution_text)}")
            return None

        # If already a list of length 2 with numeric elements, return as-is (keep float)
        if isinstance(solution_list, list) and len(solution_list) == 2 and all(
            isinstance(x, (int, float, np.integer, np.floating)) for x in solution_list
        ):
            casted = [float(x) for x in solution_list]
            print(f"  Found expected scores: {casted}")
            return casted

        
        first_elem = solution_list[0] if len(solution_list) > 0 else ""
        if not isinstance(first_elem, str):
            first_elem = str(first_elem)
        first_line = first_elem.split('\n')[0]
        # Allow float strings like "1.0 5.0" and parse to float
        expected_scores = [float(tok) for tok in first_line.split()]
        if len(expected_scores) != 2:
            raise ValueError("expected two integers in first line")
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
    
    # Check tags (make <think> optional; <answer> must appear exactly twice)
    think_start_count = processed_str.count('<think>')
    think_end_count = processed_str.count('</think>')
    answer_start_count = processed_str.count('<answer>')
    answer_end_count = processed_str.count('</answer>')

    print(f"  <think>: count={think_start_count}, position={processed_str.find('<think>')}")
    print(f"  </think>: count={think_end_count}, position={processed_str.find('</think>')}")
    print(f"  <answer>: count={answer_start_count}, position={processed_str.find('<answer>')}")
    print(f"  </answer>: count={answer_end_count}, position={processed_str.find('</answer>')}")

    # Validate counts
    if answer_start_count != 2 or answer_end_count != 2:
        print("  [Error] Should have exactly 2 <answer> and 2 </answer> tags")
        return -1.0
    # Make <think> tags more lenient - allow mismatches or only closing tag
    if think_start_count > 1 or think_end_count > 1:
        print("  [Error] <think> and </think> must appear at most 1 time each")
        return -1.0
    # Allow cases where only </think> appears or where counts don't match
    if think_start_count > 0 and think_end_count > 0 and think_start_count != think_end_count:
        print("  [Warning] <think> and </think> count mismatch, but allowing it")
    elif think_start_count == 0 and think_end_count > 0:
        print("  [Warning] Only </think> found without <think>, but allowing it")

    # Verify tag order
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    if len(matches) != 2:
        print("  [Error] Should have exactly 2 answer tags")
        return -1.0

    think_end_pos = processed_str.find('</think>') if think_end_count == 1 else -1
    first_answer_pos = matches[0].start()
    if think_end_count == 1 and not (think_end_pos <= first_answer_pos):
        print("  [Error] First <answer> must appear after </think>")
        return -1.0
    print("  Tag sequence validation passed")

    # Validate answer content format
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    
    if len(matches) != 2:
        print("  [Error] Should have exactly 2 answer tags")
        return -1.0
        
    try:
        # Extract both scores as float to allow decimal values
        score_1 = float(matches[0].group(1).strip())
        score_2 = float(matches[1].group(1).strip())
        
        # Validate scores range as floats
        if not (0.0 <= score_1 <= 10.0) or not (0.0 <= score_2 <= 10.0):
            print("  [Error] Scores must be between 0.0 and 10.0")
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
        
        expected_relation = float(expected_scores[0]) >= float(expected_scores[1])
        
        answer_relation = float(answer_text[0]) >= float(answer_text[1])
        
        
        if expected_relation == answer_relation:
            print("  Content validation: FULL MATCH")
            score = 2.0
            
            # Only compute absolute error and delta magnitude in non-reward_wo_score mode
            if reward_ablation != 'reward_wo_score':
                
                abs_diff = abs(float(expected_scores[0]) - float(answer_text[0])) + \
                           abs(float(expected_scores[1]) - float(answer_text[1]))
                
                
                if abs_diff == 0:
                    score += 1.0
                    print("  Perfect match: +1.0")
                elif abs_diff <= 2:
                    score += 0.6
                    print(f"  Close match (diff={abs_diff}): +0.6")
                    
                
                pred_diff = abs(float(answer_text[1]) - float(answer_text[0]))
                expected_diff = abs(float(expected_scores[1]) - float(expected_scores[0]))
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
        # Print ordering of Expected vs Predicted
        expected_relation = expected_scores[0] >= expected_scores[1]
        predicted_relation = answer_dict[0] >= answer_dict[1]
        
        print(f"\n[Content Validation]")
        print(f"  Expected: {expected_scores[0]} {'>=' if expected_relation else '<'} {expected_scores[1]}")
        print(f"  Predicted: {answer_dict[0]} {'>=' if predicted_relation else '<'} {answer_dict[1]}")
        

        answer_score = parse_model_answer(answer_dict, expected_scores, reward_ablation) * answer_reward

        if reward_ablation == 'reward_w_length':
            length_bonus = 1.8 if response_length > 120 else 0.0
            length_penalty = -1.0 if response_length >= max_response_length else 0.0
            answer_score +=  length_bonus + length_penalty
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
