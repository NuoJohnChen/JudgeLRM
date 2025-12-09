import json
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
    pandalm_path = "/shared/hdd/nuochen/PandaLM/data/results_oursgrammarly.json"#"/shared/hdd/nuochen/PandaLM/data/qwen257binstruct.json"#"/shared/hdd/nuochen/PandaLM/data/results_judgelrm3B_step2000.json"#"/shared/hdd/nuochen/PandaLM/data/results_r1.json"#"/shared/hdd/nuochen/PandaLM/data/pandalm-7b-testset-v1.json"
    testset_path = "/shared/hdd/nuochen/PandaLM/data/devide_by_domain/pandalm_typecase_non-reasoning_grammarly.json"
    
    
    y_true, y_pred = load_and_match_data(pandalm_path, testset_path)
    
    
    metrics = calculate_metrics(y_true, y_pred)
    
    
    print("Evaluation Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")