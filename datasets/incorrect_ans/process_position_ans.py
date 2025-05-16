import json
import os
from collections import defaultdict, OrderedDict

def split_json_by_answer_position(input_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    sorted_items = sorted(data.items(), key=lambda x: int(x[0].replace('test', '')))
    
    max_incorrect_answers = max(len(item['all_adv_docs']) for _, item in sorted_items)
    
    output_data = defaultdict(OrderedDict)
    
    for item_id, item_data in sorted_items:
        question = item_data['question']
        correct_answer = item_data['correct answer']
        all_adv_docs = item_data['all_adv_docs']
        
        for i, adv_doc in enumerate(all_adv_docs, start=1):
            incorrect_answer = adv_doc['incorrect_answer']
            adv_texts = adv_doc['adv_texts']
            
            new_entry = {
                "id": item_id,
                "question": question,
                "correct answer": correct_answer,
                "incorrect answer": incorrect_answer,
                "adv_texts": adv_texts
            }
            
            output_data[i][item_id] = new_entry
    
    for i, data_dict in output_data.items():
        output_path = os.path.join(output_dir, f"nq-inco_ans_{i}.json")
        with open(output_path, 'w') as out_f:
            json.dump(data_dict, out_f, indent=4)
        print(f"Created file: {output_path}")

input_file = "nq-all-adv-docs.json"
output_dir = "split_by_answer_position"
split_json_by_answer_position(input_file, output_dir)