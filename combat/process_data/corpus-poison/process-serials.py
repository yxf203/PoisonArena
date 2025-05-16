import json
import csv

with open('/data2/chenliuji/combat/main/process_data/AdvDec/nq/nq-inco_ans_5.json', 'r') as f:
    provide_answers = json.load(f)

with open('serials/serials-dcorpus-ans5.json', 'r') as f:
    data_5000_10 = json.load(f)

with open('/data2/chenliuji/combat/datasets/serial_questions/nq-serials-questions-with-id.json', 'r') as f:
    serial_questions = json.load(f)

new_to_original_mapping = {}

with open('/data2/chenliuji/combat/datasets/serial_questions/test_to_s_mapping.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        original_test_id = row['original_test_id']
        new_s_id = row['new_s_id']
        new_to_original_mapping[new_s_id] = original_test_id
output_data = []
i = 0
for item in data_5000_10:
    id = item['id']
    id = new_to_original_mapping[id]
    pa_item = provide_answers[id]
    incorrect_answer = pa_item['incorrect answer']
    answer = pa_item['correct answer']
    
    for index, question in enumerate(serial_questions[id]['serial_questions']):
        new_dict = {
            # "id": f"s{i}",
            "id": question['id'],
            "question": question['question'],
            "incorrect_answer": incorrect_answer,
            "answer": answer,
            "adv_texts": [{"context": f"{incorrect_answer} {adv_text['dummy_text']}"} for adv_text in item['adv_texts']]
        }
        i += 1
        output_data.append(new_dict)
    # break

with open('/data2/chenliuji/combat/main/serials_data/original/corpus/ca-serials-ans5.json', 'w') as f:
    json.dump(output_data, f, indent=4)
print("done!")