import json

def majority_vote(annotations):
    # Count occurrences of each annotation
    count = {}
    for a in annotations:
        count[a] = count.get(a, 0) + 1
    # Find the annotation with the highest count
    max_count = max(count.values())
    # Get all annotations tied for max count
    candidates = [k for k, v in count.items() if v == max_count]
    # If multiple candidates, return the smallest
    return min(candidates)

def count_annotations_and_save(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    count = {0: 0, 1: 0, 2: 0}
    
    for item in data:
        # Get three annotations
        annotations = [item['annotator1'], item['annotator2'], item['annotator3']]
        # Compute majority vote
        result = majority_vote(annotations)
        # Add label field
        item['label'] = result
        # Tally results
        count[result] += 1
    
    # Save back to original file
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    return count

if __name__ == "__main__":
    file_path = "/shared/hdd/nuochen/PandaLM/data/devide_by_domain/pandalm_typecase_reasoning.json"
    result = count_annotations_and_save(file_path)
    print(f"Annotation stats:")
    print(f"Count of 0: {result[0]}")
    print(f"Count of 1: {result[1]}")
    print(f"Count of 2: {result[2]}")
    print(f"File updated; label field added and saved to {file_path}")