from transformers import pipeline
import json
from transformers import AutoTokenizer


# Example JSON input
# json_input = {"review_id": "abtw7no6nDD2dxjG9yAkBo", "question_id": 99999, "answer1_id": "", "answer2_id": "", "reviewer_id": 1, "text": "1 10\nAssistant 1's response is completely irrelevant to the question asked. The user asked for the categorization of a specific artwork, but the assistant provided a programming exercise which is not related to the question at all. Therefore, Assistant 1 gets a score of 1. On the other hand, Assistant 2's response is accurate and directly answers the user's question. The artwork by Mark Rothko is indeed categorized as Abstract. Therefore, Assistant 2 gets a score of 10.", "score": [1.0, 10.0], "question_body": "Categorize the following artwork: Abstraction of Reality, by Mark Rothko.\n", "answer1_body": "![Image](assets/abstraction2.png)\n---  \n  \n## [Exercise](https://repl.it/@trashman/artistic-conflation-tutorial?lite=true)\nThis programming exercise is a follow up to the previous module's exercise, Exercise: [Abstracting the Abstract](https://repl.it/@trashman/artistic-conflation-tutorial). In this exercise, we're going to do the same thing with the previous exercise's output, but this time we're going to do it using the `pandas` library.\n  \nWe need to go back and download the data from the previous module's exercise, do some cleaning work on it, and then use it to create a new Pandas dataframe. We're going to use this new dataframe to create our second `Pandas` art piece, which will be a `scatterplot` of the data, with the x-axis being the `Artist` and the y-axis being the `Year`.\n# How to use this template\n\n## Requirements\n- A `GitHub` account\n- An idea\n- A laptop\n-", "answer2_body": "Abstract", "answer1_model_id": "mpt-7b", "answer2_model_id": "flan-t5-xxl", "answer1_metadata": {"decoding_method": "top_p_sampling"}, "answer2_metadata": {"decoding_method": "top_p_sampling"}, "reference": {"decoding_method": "None", "model": "itwgpt4/40500", "text": "The artwork \"Abstraction of Reality\" by Mark Rothko would be categorized as Abstract Expressionism."}, "score_w_reference": [1.0, 5.0], "text_w_reference": "1 5\n\nAssistant 1's response is completely irrelevant to the question asked. The user asked for the categorization of an artwork, but the assistant started discussing a programming exercise, which is not related to the question at all. Therefore, it gets a score of 1.\n\nAssistant 2's response is somewhat relevant as it identifies the artwork as \"Abstract\". However, it lacks detail and accuracy. The correct categorization should be \"Abstract Expressionism\", not just \"Abstract\". Therefore, it gets a score of 5. It provided a relevant but incomplete answer.", "system": "You are a helpful assistant tasked with evaluating the performance of two AI assistants in response to a user\u2019s question. Your role is to assess the helpfulness, relevance, accuracy, and level of detail in each assistant\u2019s response.\n\nFor each assistant, provide an overall score between 1 and 10, where a higher score indicates better performance.\n\nYour output must follow this format:\n\nFirst, a single line with two scores separated by a space: the first score for Assistant 1 and the second for Assistant 2.\nNext, a comprehensive explanation of your evaluation, ensuring no bias and that the order of responses does not influence your judgment.\nBase your evaluation on how well each assistant addressed the user\u2019s question, the correctness of the information, and the clarity and thoroughness of the response.\n<|im_end|>\n<|im_start|>user\n[Question]\n{question}\n\n[Assistant 1\u2019s Answer]\n{answer_1}\n\n[Assistant 2\u2019s Answer]\n{answer_2}\n<|im_end|>\n<|im_start|>assistant", "instruction": "[Question]\nCategorize the following artwork: Abstraction of Reality, by Mark Rothko.\n\n[Assistant 1\u2019s Answer]\n![Image](assets/abstraction2.png)\n---  \n  \n## [Exercise](https://repl.it/@trashman/artistic-conflation-tutorial?lite=true)\nThis programming exercise is a follow up to the previous module's exercise, Exercise: [Abstracting the Abstract](https://repl.it/@trashman/artistic-conflation-tutorial). In this exercise, we're going to do the same thing with the previous exercise's output, but this time we're going to do it using the `pandas` library.\n  \nWe need to go back and download the data from the previous module's exercise, do some cleaning work on it, and then use it to create a new Pandas dataframe. We're going to use this new dataframe to create our second `Pandas` art piece, which will be a `scatterplot` of the data, with the x-axis being the `Artist` and the y-axis being the `Year`.\n# How to use this template\n\n## Requirements\n- A `GitHub` account\n- An idea\n- A laptop\n-\n\n[Assistant 2\u2019s Answer]\nAbstract"}
# json_input=   {
#     "idx": 149,
#     "motivation_app": "Wolfram alpha",
#     "cmp_key": "bloom-7b_opt-7b",
#     "instruction": "Based on the facts that have been provided, prove the following statement.",
#     "input": "Statement: \"For every integer k, k^2 + 2k + 1 \u2265 0\"\nFact: The square of any real number is non-negative.",
#     "response1": "Fact: For every integer k, k^2 + 2k + 1 \u2265 0.",
#     "response2": "The square of any real number is non-negative.",
#     "annotator1": 1,
#     "annotator2": 1,
#     "annotator3": 1,
#     "label": 1,
#     "needed_reasoning_rate1-10": 6,
#     "rate_explanation": "The task requires judging whether the responses correctly use the given fact to prove the statement. Response1 merely restates the statement without proof, while Response2 correctly cites the given fact but does not complete the proof by showing how it applies to the statement. The reasoning needed involves verifying logical consistency and completeness, which is intermediate in complexity.\n----------------------------------------"
#   }
json_input=  {
    "idx": 549,
    "motivation_app": "Google Docs",
    "cmp_key": "cerebras-gpt-6.7B_pythia-6.9b",
    "instruction": "Include important study notes and key points that someone should know about the given subject.",
    "input": "history of the USA",
    "response1": "The history of the United States is one of the most influential and influential countries in the world. It has had a profound impact on the world and continues to shape the world today. The United States has had a long and storied history, from its founding in 1775 to the present day. It has had a tumultuous and tumultuous history, from the Revolutionary War to the Civil War and beyond. The United States has had a tumultuous and tumultuous history, from the Revolutionary War to the Civil War and beyond. The United States has had a long and storied history, from its founding in 1775 to the present day. It has had a profound impact on the world and continues to shape the world today.",
    "response2": "1. The United States of America was founded in 1776.\n2. The Declaration of Independence was signed in 1776.\n3. The Constitution of the United States of America was signed in 1787.\n4. The Civil War began in 1861.\n5. The Emancipation Proclamation was issued in 1863.\n6. The 13th Amendment was ratified in 1865.\n7. The 14th Amendment was ratified in 1868.\n8. The 15th Amendment was ratified in 1870.\n9. The 16th Amendment was ratified in 1913.\n10. The 17th Amendment was ratified in 1913.\n11. The 18th Amendment was ratified in 1919.\n12. The 19th Amendment was ratified in 1920.\n13. The 20th Amendment was ratified in 1933.\n14. The 21st Amendment was ratified in 1933.\n15. The 22nd Amendment was ratified in",
    "annotator1": 2,
    "annotator2": 2,
    "annotator3": 2,
    "label": 2,
    "needed_reasoning_rate1-10": 7,
    "rate_explanation": "The task requires evaluating the quality of responses based on their adherence to the instruction to include important study notes and key points about the history of the USA. Response1 is repetitive and lacks specific details, while Response2 provides a clear, concise list of key historical events. The reasoning needed to judge these responses involves assessing clarity, specificity, and relevance to the instruction, which is moderately complex.\n----------------------------------------"
}
# json_input =   {
#     "idx": 514,
#     "motivation_app": "Redfin",
#     "cmp_key": "bloom-7b_llama-7b",
#     "instruction": "In a completely dark room there are 100 coins, 10 of which are heads up and the remaining 90 are tails up. You can't tell heads or tails by sight or touch, but you can flip the coins any way you want. How can you divide the coins into two piles so that the same number of coins are face up in both piles?",
#     "input": "",
#     "response1": "Randomly select 10 coins for the first pile, leaving 90 for the second pile. Flip all the coins in the first pile. At this point, assume that the first pile originally had k heads coins and was flipped to 10-k heads; the second pile originally had 10-k heads coins and was not flipped. The final number of heads in both piles is 10-k, and the goal is reached.",
#     "response2": "Divide the coins equally into two piles (50 coins each). Since the initial number of heads is 10, with probability, there are about 5 heads coins in each pile, so the two piles are naturally equal in number of heads.",
#     "annotator1": 2,
#     "annotator2": 2,
#     "annotator3": 2,
#     "label": 2,
#     "needed_reasoning_rate1-10": 8,
#     "rate_explanation": "The task involves evaluating the relevance and quality of the responses in relation to the instruction, which requires some level of reasoning to determine whether the responses effectively list top real estate marketing words that add value and engage buyers. Response1 provides a straightforward but somewhat generic list of real estate terms, while Response2 offers more descriptive and potentially engaging words, albeit with significant repetition. Assessing the effectiveness, uniqueness, and engagement potential of each response requires moderate to high reasoning ability, particularly in discerning the subtle differences in their utility for marketing purposes.\n----------------------------------------"
#   }
# json_input =   {
#     "idx": 514,
#     "motivation_app": "Redfin",
#     "cmp_key": "bloom-7b_llama-7b",
#     "instruction": "List some of the top real estate marketing words to add value to the listing and engage more potential buyers.",
#     "input": "",
#     "response1": "1. Location\n2. Amenities\n3. Pricing\n4. Features\n5. Photos\n6. Video\n7. Virtual tours\n8. Open houses\n9. Showings\n10. Offers",
#     "response2": "Some of the top real estate marketing words to add value to the listing and engage more potential buyers include: \n\n- Luxurious \n- Spacious \n- Modern \n- Convenient \n- Stylish \n- Comfortable \n- Affordable \n- Charming \n- Quiet \n- Secluded \n- Serene \n- Inviting \n- Comfortable \n- Convenient \n- Spacious \n- Modern \n- Stylish \n- Affordable \n- Charming \n- Quiet \n- Secluded \n- Serene \n- Inviting \n- Comfortable \n- Convenient \n- Spacious \n- Modern \n- Stylish \n- Affordable \n- Charming \n- Quiet \n- Secluded \n- Serene \n- Inviting \n- Comfortable \n- Convenient \n- Spacious \n- Modern \n- Stylish \n- Affordable \n- Charming \n- Quiet \n- Secluded \n- Serene \n- Inviting \n- Comfortable \n- Convenient \n- Spacious \n-",
#     "annotator1": 2,
#     "annotator2": 2,
#     "annotator3": 2,
#     "label": 2,
#     "needed_reasoning_rate1-10": 8,
#     "rate_explanation": "The task involves evaluating the relevance and quality of the responses in relation to the instruction, which requires some level of reasoning to determine whether the responses effectively list top real estate marketing words that add value and engage buyers. Response1 provides a straightforward but somewhat generic list of real estate terms, while Response2 offers more descriptive and potentially engaging words, albeit with significant repetition. Assessing the effectiveness, uniqueness, and engagement potential of each response requires moderate to high reasoning ability, particularly in discerning the subtle differences in their utility for marketing purposes.\n----------------------------------------"
#   }

# Extract necessary fields from the JSON
# question = json_input.get("question_body", "").strip()
# answer_1 = json_input.get("answer1_body", "").strip()
# answer_2 = json_input.get("answer2_body", "").strip()
question = json_input.get("instruction", "").strip()+"\n"+json_input.get("input", "").strip()
answer_1 = json_input.get("response1", "").strip()
answer_2 = json_input.get("response2", "").strip()

# Construct the prompt with chain-of-thought instruction
prompt = """<|im_start|>system\nYou are a helpful assistant. The assistant first performs a detailed, step-by-step reasoning process in its mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> detailed reasoning process here, explaining each step of your evaluation for both assistants </think><answer> answer here </answer>. Now the user asks you to judge the performance of two AI assistants in response to the question. Score assistants 1-10 (higher=better). Criteria includes helpfulness, relevance, accuracy, and level of detail. Avoid order, length, style or other bias. After thinking, when you finally reach a conclusion, clearly  provide your evaluation scores within <answer> </answer> tags, i.e. for example,<answer>3</answer><answer>5</answer>\n<|im_end|>\n<|im_start|>user\n[Question]\n{question}\n\n[Assistant 1’s Answer]\n{answer_1}\n\n[Assistant 2’s Answer]\n{answer_2}\n<|im_end|>\n<|im_start|>assistant\n<think>"""
# Fill prompt placeholders with actual values
formatted_prompt = prompt.format(question=question, answer_1=answer_1, answer_2=answer_2)

# model_path = "/home/nuochen/.cache/huggingface/hub/models--zhiyuanhucs--qwen25_7b_zhiyuan_function_rm_323-step-600/snapshots/fbf74d02832769bf78bc44d6c68cc0e9750c5a12/"#"zhiyuanhucs/qwen25_7b_zhiyuan_function_rm_323-step-1800"
model_path = "/shared/hdd/nuochen/models/Qwen3-8B-Instruct-think-evaluator"#"/shared/hdd/nuochen/models/GRPO_logic_KK_321/abrewardwoconf/actor/global_step_400/"#"/shared/hdd/nuochen/JudgeLM/output/Qwen2.5-7B-Instruct-evaluator-rm"
# Explicitly choose CUDA device
device = "cuda:6"  # use first GPU

# Explicitly specify device when loading model
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
generator = pipeline(
    "text-generation", 
    model=model_path, 
    tokenizer=tokenizer, 
    device=device,  # explicitly use GPU
    torch_dtype="auto"  # auto-select best dtype
)

# Check GPU usage
print(f"Model on GPU: {next(generator.model.parameters()).is_cuda}")

# Generate with formatted prompt
result = generator(formatted_prompt, max_new_tokens=2048)
print(result[0]['generated_text'])  # print generated text only
