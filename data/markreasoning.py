import json
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Initialize OpenAI client
client = OpenAI(
    base_url="https://api.openai.com/v1",
    api_key=os.environ.get("OPENAI_API_KEY", ""),
)


model = "gpt-4"

def process_item(item):
    # Build prompt
    prompt = f"""For the data provided below, "response1" and "response2" represent two responses generated for the given "instruction" and "input". Consider the task of judging the performance of "response1" and "response2" in response to the "instruction" and "input". 

On a scale of 1 to 10, rate the level of reasoning ability needed to perform this judgment. 

Please provide your response in EXACTLY the following format:
----------------------------------------
Score: [your score, an integer between 1 and 10]
Explanation: [your explanation]
----------------------------------------

Instruction: {item['instruction']}
Input: {item['input']}
Response1: {item['response1']}
Response2: {item['response2']}"""

    # Call API
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    # Parse response
    result = response.choices[0].message.content.strip()
    try:
        # Handle possible format variations
        score_part = result.split('Explanation:')[0].split('Explanation')[0].strip()
        score = ''.join(filter(str.isdigit, score_part))
        
        explanation_part = result.split('Explanation:')[-1].split('Explanation')[-1].strip()
        
        item['needed_reasoning_rate1-10'] = int(score) if score else None
        item['rate_explanation'] = explanation_part if explanation_part else None
    except Exception as e:
        print(f"Parse error: {e}, raw response: {result}")
        item['needed_reasoning_rate1-10'] = None
        item['rate_explanation'] = None
    return item

# Read raw data
input_path = "/shared/hdd/nuochen/PandaLM/data/testset-v1.json"
output_path = "/shared/hdd/nuochen/PandaLM/data/testset-v1_reasoningrategpt51.json"

with open(input_path, 'r') as f:
    data = json.load(f)

# Process in parallel with thread pool
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(process_item, item) for item in data]
    results = []
    for future in tqdm(as_completed(futures), total=len(data)):
        results.append(future.result())

# Save results
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Processing complete, results saved to {output_path}")