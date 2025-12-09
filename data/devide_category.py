import json
import os
from collections import defaultdict

# Part 1: categorize by motivation_app
input_path = "/shared/hdd/nuochen/PandaLM/data/testset-v1_reasoningrate.json"
category_dir = "/shared/hdd/nuochen/PandaLM/data/category"
os.makedirs(category_dir, exist_ok=True)

# Define app category mapping
app_categories = {
    # Language & Writing
    "Grammarly": "Language_Writing",
    "merriam-webster.com": "Language_Writing",
    "ludwig.guru": "Language_Writing",
    "Google Docs": "Language_Writing",
    # Education & Learning
    "Coursera": "Education_Learning",
    "Socratic by Google": "Education_Learning",
    "Doulingo": "Education_Learning",
    "Wolfram alpha": "Education_Learning",
    "Wikipedia": "Education_Learning",
    "National Geographic": "Education_Learning",
    "(Wolfram alpha)?": "Education_Learning",
    # Search & Information Retrieval
    "Google Search": "Search_Information_Retrieval",
    "Quora": "Search_Information_Retrieval",
    "Reddit": "Search_Information_Retrieval",
    "CNN News": "Search_Information_Retrieval",
    "Indeed": "Search_Information_Retrieval",
    "Meetup": "Search_Information_Retrieval",
    # Entertainment & Media
    "Netflix": "Entertainment_Media",
    "IMDB": "Entertainment_Media",
    "Spotify": "Entertainment_Media",
    "YouTube": "Entertainment_Media",
    "ESPN": "Entertainment_Media",
    "Instagram": "Entertainment_Media",
    "Facebook": "Entertainment_Media",
    "Twitter": "Entertainment_Media",
    "Telegram": "Entertainment_Media",
    # Life & Utility
    "Weather": "Life_Utility",
    "Tasty": "Life_Utility",
    "Sudoku": "Life_Utility",
    "Goodreads": "Life_Utility",
    "yelp": "Life_Utility",
    "Yelp": "Life_Utility",
    "tripadvisor.com": "Life_Utility",
    "Redfin": "Life_Utility",
    "Play Store": "Life_Utility",
    "Amazon": "Life_Utility",
    "Wysa": "Life_Utility",
    "sth related to real estate?": "Life_Utility",
    # Office & Productivity
    "MS Excel": "Office_Productivity",
    "MS Powerpoint": "Office_Productivity",
    "Google Sheet": "Office_Productivity",
    "Jira": "Office_Productivity",
    "Google Meet": "Office_Productivity",
    "Gmail": "Office_Productivity",
    # Technology & Development
    "https://cohere.ai/": "Technology_Development",
    "https://abcnotation.com/": "Technology_Development",
    "instructables": "Technology_Development",
    # Social & Professional Networking
    "LinkedIn": "Social_Professional_Networking",
    "Messenger": "Social_Professional_Networking",
    "Blogger": "Social_Professional_Networking"
}

# import json
# from collections import defaultdict

# 
# input_path = "/shared/hdd/nuochen/PandaLM/data/testset-v1_reasoningrate.json"
# with open(input_path, 'r') as f:
#     data = json.load(f)

# # Collect all motivation_app values
# all_apps = set()
# unrecognized_apps = set()

# for item in data:
#     app = item.get('motivation_app', '')
#     if app:  # ignore empty values
#         all_apps.add(app)
#         if app not in app_categories:
#             unrecognized_apps.add(app)

# 
# print("All motivation_app values:", len(all_apps))
# print("Uncategorized motivation_app:", len(unrecognized_apps))
# print("\nUncategorized app names:")
# for app in sorted(unrecognized_apps):
#     print(f"- {app}")

# Read data and categorize by app
with open(input_path, 'r') as f:
    data = json.load(f)

categorized_data = defaultdict(list)
for item in data:
    app = item.get('motivation_app', '')
    category = app_categories.get(app, 'Unknown')
    categorized_data[category].append(item)

# Save categorized results
for category, items in categorized_data.items():
    output_path = os.path.join(category_dir, f"pandalmtest_{category}.json")
    with open(output_path, 'w') as f:
        json.dump(items, f, indent=2)

# After saving categorized results, log Unknown details
if 'Unknown' in categorized_data:
    print("\nDebug info - Unknown category details:")
    print(f"Total entries: {len(categorized_data['Unknown'])}")
    
    # Collect all motivation_app values within Unknown category
    unknown_apps = defaultdict(int)
    for item in categorized_data['Unknown']:
        app = item.get('motivation_app', '')
        unknown_apps[app] += 1
    
    print("\nDistinct motivation_app values and counts:")
    for app, count in sorted(unknown_apps.items(), key=lambda x: x[1], reverse=True):
        print(f"{repr(app)}: {count} times")  # use repr to show raw format
    
    # Check for None or empty strings
    none_count = sum(1 for item in categorized_data['Unknown'] if not item.get('motivation_app', ''))
    print(f"\nEntries with empty/None: {none_count}")

# Part 2: categorize by reasoning requirement
reasoning_dir = "/shared/hdd/nuochen/PandaLM/data/category_neededreasoning"
os.makedirs(reasoning_dir, exist_ok=True)

category_counts = {}

# Process each category file
for filename in os.listdir(category_dir):
    if filename.startswith('pandalmtest_') and filename.endswith('.json'):
        category = filename[12:-5]  # extract category name
        input_path = os.path.join(category_dir, filename)
        
        with open(input_path, 'r') as f:
            category_data = json.load(f)
        
        need_reasoning = []
        not_need_reasoning = []
        
        for item in category_data:
            rate = item.get('needed_reasoning_rate1-10', 0)
            if rate >= 5:
                need_reasoning.append(item)
            else:
                not_need_reasoning.append(item)
        
        # Save reasoning-required subset
        reasoning_output_path = os.path.join(reasoning_dir, f"pandalmtest_{category}_needreasoning.json")
        with open(reasoning_output_path, 'w') as f:
            json.dump(need_reasoning, f, indent=2)
        
        not_reasoning_output_path = os.path.join(reasoning_dir, f"pandalmtest_{category}_notneedreasoning.json")
        with open(not_reasoning_output_path, 'w') as f:
            json.dump(not_need_reasoning, f, indent=2)
        
        category_counts[category] = {
            'need_reasoning': len(need_reasoning),
            'not_need_reasoning': len(not_need_reasoning)
        }

        # Save non-reasoning subset

# Print summary stats
print("Reasoning requirement counts by category:")
for category, counts in category_counts.items():
    print(f"{category}:")
    print(f"  Needs reasoning (≥5): {counts['need_reasoning']} entries")
    print(f"  Does not need reasoning (<5): {counts['not_need_reasoning']} entries")
    print()