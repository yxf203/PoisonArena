import json


with open('nq/nq-inco_ans_1.json', 'r') as f:
    new_questions = json.load(f)


with open('serials/nq-inco_ans_1-4ad_misinfo_contriever_reverse_dot.json', 'r') as f:
    misinfo_data = json.load(f)


with open('../../datasets/serial_questions/nq-serials-questions-with-id.json', 'r') as f:
    serial_questions = json.load(f)

formatted_data = []

for m in misinfo_data:
    id = m["id"]
    test_data = new_questions[id]
    for question_dict in serial_questions[id]["serial_questions"]:
        # for question in test_data["questions"]:
        formatted_entry = {
            # "id": f"s{len(formatted_data)}",
            "id": question_dict["id"],
            "question": question_dict["question"],
            "incorrect_answer": test_data["incorrect answer"],
            "answer": test_data["correct answer"],
            # "question":test_data["questions"][0],
            # "question":test_data["question"],
            "adv_texts": [{"context": f"{adv_text}"} for adv_text in m["adv_texts"]],

        }
        formatted_data.append(formatted_entry)


with open('../data/original/serials_data/AdvDec/AdvDec5-serials-dot-ans1.json', 'w') as f:
    json.dump(formatted_data, f, indent=4)

print("done!")