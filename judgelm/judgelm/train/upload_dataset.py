import json
from datasets import Dataset
from pathlib import Path
import sys

# Add root path to use absolute import
file = Path(__file__).resolve()
root = file.parents[2]
sys.path.append(str(root))

from judgelm.utils import jlload

def process_and_upload_dataset(input_file, repo_id):
    """Process dataset and push to Hugging Face Hub"""
    print("Loading data...")
    data = jlload(input_file)
    
    # Process data and ensure metadata is not empty
    processed_data = []
    for item in data:
        # Ensure chosen and rejected are lists
        if isinstance(item['chosen'], dict):
            item['chosen'] = [item['chosen']]
        if isinstance(item['rejected'], dict):
            item['rejected'] = [item['rejected']]
            
        # Add a dummy metadata field
        if 'metadata' not in item:
            item['metadata'] = {'dummy': 'field'}
            
        processed_data.append(item)
    
    # Create Dataset object
    dataset = Dataset.from_list(processed_data)
    
    # Push to Hub
    print(f"Pushing dataset to {repo_id}...")
    dataset.push_to_hub(repo_id)
    print("Done!")

if __name__ == "__main__":
    input_file = "/shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_train_100k_dpo.jsonl"
    repo_id = "nuojohnchen/judgelm-train-100k-dpo"
    process_and_upload_dataset(input_file, repo_id) 