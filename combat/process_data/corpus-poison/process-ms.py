import json

with open('ms/ms-inco_ans_6.json', 'r') as f:
    provide_answers = json.load(f)

with open('ms/ms-dcorpus-ans6.json', 'r') as f:
    data_5000_10 = json.load(f)
data_5000_10.sort(key=lambda x: int(x['id']))
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
            "adv_texts": [{"context": f"{incorrect_answer} {adv_text['dummy_text']}"} for adv_text in item['adv_texts']],
            "incorrect_answer": incorrect_answer,
            "answer": answer,
            "query_type": pa_item['query_type'],
            "golden_docs": pa_item['golden_docs'],
        }
        i += 1
        output_data.append(new_dict)
    # break

with open('ms/corpus/corpus-dcorpus-answer-ans6.json', 'w') as f:
    json.dump(output_data, f, indent=4)
print("done!")