import json
from collections import defaultdict

def transform_data(tmp_file_path, nq_file_path, output_file_path):
    # Load the tmp-a1.json data
    with open(tmp_file_path, 'r') as f:
        tmp_data = json.load(f)
    
    # Load the nq-a1.json data
    with open(nq_file_path, 'r') as f:
        nq_data = json.load(f)
    
    with open('../datasets/serial_questions/nq-serials-questions-with-id.json', 'r') as f:
        serial_questions = json.load(f)

    # Create a dictionary to map IDs to their corresponding data from nq-a1.json
    nq_data_map = {v['id']: v for k, v in nq_data.items()}
    
    # First group all adv_text_after_attack by their ID
    adv_texts_map = defaultdict(list)
    for item in tmp_data:
        adv_texts_map[item['id']].append({
            "context": item["adv_text_after_attack"]
        })
    
    # Transform the data
    transformed_data = []
    
    # We'll process each unique ID from the adv_texts_map
    for item_id, adv_texts in adv_texts_map.items():
        # Get the corresponding data from nq-a1.json
        nq_item = nq_data_map.get(item_id, {})
        for question_dict in serial_questions[item_id]["serial_questions"]:
            # Create the transformed item
            transformed_item = {
                "id": question_dict["id"],
                "question": question_dict["question"],
                "incorrect_answer": nq_item.get("incorrect answer", None),
                "answer": [nq_item.get("correct answer", "")],
                "adv_texts": adv_texts,  # All adv_texts for this ID

            }
            
            transformed_data.append(transformed_item)
    
    # Save the transformed data
    with open(output_file_path, 'w') as f:
        json.dump(transformed_data, f, indent=4)
# Example usage:
transform_data(
    'serials/serials_ans_6.json',
    'diff-corpus/new-diff-corpus/nq-inco_ans_6.json',
    'gaslite-serials-ans6.json'
)
