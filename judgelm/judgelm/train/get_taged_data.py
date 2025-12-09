import json

def convert_text_format(text):
    # Split score and evaluation content
    parts = text.split('\n', 1)
    if len(parts) != 2:
        return text  # If format is unexpected, return original text
    
    scores = parts[0].strip()
    content = parts[1].strip()
    
    # Build new format
    new_text = f"<think>{content}</think>The pair-wise score of two responses are {scores}"
    return new_text

def process_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            try:
                data = json.loads(line.strip())
                # Convert text field
                if 'text' in data:
                    data['text'] = convert_text_format(data['text'])
                # Convert text_w_reference field if present
                if 'text_w_reference' in data:
                    data['text_w_reference'] = convert_text_format(data['text_w_reference'])
                # Write to new file
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
            except json.JSONDecodeError:
                print(f"Error processing line: {line}")
                continue

if __name__ == "__main__":
    input_file = "/shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_train_100k.jsonl"
    output_file = "/shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_train_100k_think.jsonl"
    process_file(input_file, output_file)
    print("Conversion completed!")