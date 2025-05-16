import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# 加载 nq-serials-questions-with-id.json 文件
with open('/data2/chenliuji/combat/datasets/serial_questions/nq-serials-questions-with-id.json', 'r') as f:
    serial_questions = json.load(f)

# 加载 PW-serials-ans6.json 文件
with open('PW-serials-ans6.json', 'r') as f:
    pb_serials = json.load(f)

formatted_data = []

for item in pb_serials:
    id = item["id"]
    for question_dict in serial_questions[id]["serial_questions"]:
        formatted_item = {
            "id": question_dict["id"],
            "question": question_dict["question"],
            "incorrect_answer": item["incorrect_answer"],
            "answer": item["answer"],
            "adv_texts": [{"context": context["context"]} for context in item["adv_texts"]]
        }
        formatted_data.append(formatted_item)

# Save the formatted data to a new JSON file
with open('PW-serials-formatted-ans6.json', 'w') as f:
    json.dump(formatted_data, f, indent=4, ensure_ascii=False)