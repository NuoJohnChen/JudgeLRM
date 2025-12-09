import json
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_and_match_data(pandalm_path, testset_path):
    
    with open(pandalm_path, 'r') as f:
        pandalm_data = json.load(f)
    with open(testset_path, 'r') as f:
        testset_data = json.load(f)
    
    
    testset_dict = {item['idx']: item for item in testset_data}
    
    
    y_true = []
    y_pred = []
    for item in pandalm_data:
        idx = item['idx']
        if idx in testset_dict:
            y_true.append(testset_dict[idx]['label'])
            y_pred.append(item["result"])
    
    return y_true, y_pred

def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

if __name__ == "__main__":
    
    DEFAULT_PANDALM_PATH = "/shared/hdd/nuochen/PandaLM/data/results_judge_JudgeLRM-8B_tem07seed2025c111.json"#"/shared/hdd/nuochen/PandaLM/data/results_judge_JudgeLRM-8B_tem07seed2024c111.json"#"/shared/hdd/nuochen/PandaLM/data/results_jud_infgelrmwoabs350new.json"#"/shared/hdd/nuochen/PandaLM/data/results_lrm8b560.json"#"/shared/hdd/nuochen/PandaLM/data/results_judgelrm4b_step520new_stop.json"#"/shared/hdd/nuochen/PandaLM/data/results_jugdelrm7b.json"#"/shared/hdd/nuochen/PandaLM/data/results/results_qwen257binstructsft_Social_Professional_Networking_needreasoning.json" #"/shared/hdd/nuochen/PandaLM/data/JudgeLRM-14B-reward-wo-score-step400.json"#/shared/hdd/nuochen/PandaLM/data/results_qwen25_7b_zhiyuan_function_rm_323-step-600.json"
    DEFAULT_TESTSET_PATH = "/shared/hdd/nuochen/PandaLM/data/testset-v1.json"#"/shared/hdd/nuochen/PandaLM/data/category_neededreasoning/pandalmtest_Life_Utility_needreasoning.json"##"/shared/hdd/nuochen/PandaLM/data/testset-v1.json"

    
    parser = argparse.ArgumentParser(description='Compute model evaluation metrics')
    parser.add_argument('--pandalm', type=str, default=DEFAULT_PANDALM_PATH,
                       help=f'PandaLM result file path (default: {DEFAULT_PANDALM_PATH})')
    parser.add_argument('--testset', type=str, default=DEFAULT_TESTSET_PATH,
                       help=f'Test set file path (default: {DEFAULT_TESTSET_PATH})')
    args = parser.parse_args()

    
    y_true, y_pred = load_and_match_data(args.pandalm, args.testset)
    
    
    metrics = calculate_metrics(y_true, y_pred)
    
    
    print("Evaluation Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"\nFile paths used:")
    print(f"PandaLM result: {args.pandalm}")
    print(f"Test set: {args.testset}")