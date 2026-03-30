import json
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_and_match_data(pandalm_path, testset_path):
    # 加载数据
    with open(pandalm_path, 'r') as f:
        pandalm_data = json.load(f)
    with open(testset_path, 'r') as f:
        testset_data = json.load(f)
    
    # 创建索引映射
    testset_dict = {item['idx']: item for item in testset_data}
    
    # 匹配数据
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
    # 默认路径
    DEFAULT_PANDALM_PATH = "/user/PandaLM/data/results_qwen38b_sft_think.json"#"/user/PandaLM/data/results_jugdelrm7b.json"#"/user/PandaLM/data/results/results_qwen257binstructsft_Social_Professional_Networking_needreasoning.json" #"/user/PandaLM/data/JudgeLRM-14B-reward-wo-score-step400.json"#/user/PandaLM/data/results_qwen25_7b_zhiyuan_function_rm_323-step-600.json"
    DEFAULT_TESTSET_PATH = "/user/PandaLM/data/testset-v1.json"#"/user/PandaLM/data/category_neededreasoning/pandalmtest_Life_Utility_needreasoning.json"##"/user/PandaLM/data/testset-v1.json"

    # 设置命令行参数
    parser = argparse.ArgumentParser(description='计算模型评估指标')
    parser.add_argument('--pandalm', type=str, default=DEFAULT_PANDALM_PATH,
                       help=f'PandaLM结果文件路径 (默认: {DEFAULT_PANDALM_PATH})')
    parser.add_argument('--testset', type=str, default=DEFAULT_TESTSET_PATH,
                       help=f'测试集文件路径 (默认: {DEFAULT_TESTSET_PATH})')
    args = parser.parse_args()

    # 加载并匹配数据
    y_true, y_pred = load_and_match_data(args.pandalm, args.testset)
    
    # 计算指标
    metrics = calculate_metrics(y_true, y_pred)
    
    # 输出结果
    print("Evaluation Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"\n使用的文件路径:")
    print(f"PandaLM 结果: {args.pandalm}")
    print(f"测试集: {args.testset}")