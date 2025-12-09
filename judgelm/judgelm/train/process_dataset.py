from datasets import load_dataset
from get_dpo_data import remove_empty_metadata

def process_dataset():
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("json", data_files="/shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_train_100k_think.jsonl", split="train")
    
    # Remove empty metadata fields
    print("Removing empty metadata...")
    dataset = remove_empty_metadata(dataset)
    
    # Save processed dataset
    print("Saving processed dataset...")
    dataset.to_json("/shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_train_100k_think_processed.jsonl")
    
    print("Dataset processing completed!")

if __name__ == "__main__":
    process_dataset() 