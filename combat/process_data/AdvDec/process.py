import json


with open('nq/nq-inco_ans_6.json', 'r') as f:
    new_questions = json.load(f)


with open('nq/nq-inco_ans_6-4ad_vicuna_misinfo_contriever_reverse_dot.json', 'r') as f:
    misinfo_data = json.load(f)

formatted_data = []

for m in misinfo_data:
    id = m["id"]
    test_data = new_questions[id]
    # for question in test_data["questions"]:
    formatted_entry = {
        # "id": f"s{len(formatted_data)}",
        "id": id,
        # "question": question,
        # "question":test_data["questions"][0],
        "question":test_data["question"],
        "adv_texts": [{"context": f"{adv_text}"} for adv_text in m["adv_texts"]],
        "incorrect_answer": test_data["incorrect answer"],
        "answer": test_data["correct answer"]
    }
    formatted_data.append(formatted_entry)

# Save the formatted data to a new JSON file
with open('nq/AdvDec/AdvDec5-dcorpus-dot-ans6.json', 'w') as f:
    json.dump(formatted_data, f, indent=4)

print("done!")