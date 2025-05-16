import json
from collections import defaultdict

def transform_data(tmp_file_path, nq_file_path, output_file_path):
    # Load the tmp-a1.json data
    with open(tmp_file_path, 'r') as f:
        tmp_data = json.load(f)
    
    # Load the nq-a1.json data
    with open(nq_file_path, 'r') as f:
        nq_data = json.load(f)
    
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
        
        # Create the transformed item
        transformed_item = {
            "id": item_id,
            "question": nq_item.get("question", ""),
            "query_type": nq_item.get("query_type", ""),
            "incorrect_answer": nq_item.get("incorrect answer", None),
            "answer": [nq_item.get("correct answer", "")],
            "golden_docs": nq_item.get("golden_docs", []),
            "adv_texts": adv_texts,  # All adv_texts for this ID

        }
        
        transformed_data.append(transformed_item)
    
    # Save the transformed data
    with open(output_file_path, 'w') as f:
        json.dump(transformed_data, f, indent=4)
# Example usage:
transform_data(
    'ms/ms-dcorpus-ans_6.json',
    'ms/ms-inco_ans_6.json',
    'ms/gaslite-dcorpus-ans6.json'
)