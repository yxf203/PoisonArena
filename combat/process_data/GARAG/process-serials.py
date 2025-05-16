import json
def find_same_id_item(items, target_id):
    # get the similar texts to the question with target_id
    same_items = []
    for item in items:
        if item.get('id') == target_id:
            same_items.append(item)
    return same_items

with open('../../datasets/serial_questions/nq-serials-questions-with-id.json', 'r') as f:
    serial_questions = json.load(f)

names = ["serials/new-ans-afp-1-serials", "serials/new-ans-afp-5-serials"]
G_texts = []
for name in names:
    with open(f"./{name}.json", "r") as f:
        t_texts = json.load(f)
        G_texts += t_texts
adv_list = []
ids = set()
for item in G_texts:
    id = item["q_id"]
    ids.add(id)
    question = item["question"]
    replaced_text = item["ctx"]
    answers = item["answers"]
    # print("replaced:\n",replaced_text)
    adv_text = item["att"][-1][0][0]
    # print("replace:\n",adv_text)
    result = {
        'context': adv_text,
    }
    adv_list.append({
        "id": id,
        "question": question,
        "adv_texts": result,
        "incorrect_answer": None,
        "answer": answers
    })
final_adv_list = []
for id in ids:
    items = find_same_id_item(adv_list, id)
    adv_texts = [item["adv_texts"] for item in items]
    # question = items[0]["question"]
    answers = items[0]["answer"]
    for question_dict in serial_questions[id]["serial_questions"]:
        question = question_dict["question"]
        final_adv_list.append({
            "id": question_dict["id"],
            "question": question,
            "incorrect_answer": None,
            "answer": answers,
            "adv_texts": adv_texts,
        })
file_path = "GARAG5-ans"
with open(f"{file_path}.json", 'w') as f:
    json.dump(final_adv_list, f, indent=4)