import json

with open('nq-ans1.json', 'r') as f:
    provide_answers = json.load(f)

with open('nq-5000.json', 'r') as f:
    data_5000_10 = json.load(f)

output_data = []
i = 0
for item in data_5000_10:
    id = item['id']
    pa_item = provide_answers[id]
    incorrect_answer = pa_item['incorrect answer']
    answer = pa_item['correct answer']
    
    for index, question in enumerate(item['questions']):
        new_dict = {
            # "id": f"s{i}",
            "id": id,
            "question": question,
            "adv_texts": [{"context": f"{adv_text['dummy_text']}"} for adv_text in item['adv_texts']],
            "incorrect_answer": incorrect_answer,
            "answer": answer
        }
        i += 1
        output_data.append(new_dict)
    # break

with open('nq-5000-1.json', 'w') as f:
    json.dump(output_data, f, indent=4)
print("done!")