import json
def find_same_id_item(items, target_id):
    # get the similar texts to the question with target_id
    same_items = []
    for item in items:
        if item.get('id') == target_id:
            same_items.append(item)
    return same_items

# names = ["phi4/phi4-ans-5", "phi4/phi4-740-ans-1"]
# names = ["vicuna/vicuna-ans-2", "vicuna/vicuna-ans-3","vicuna/vicuna-ans-4-1", "vicuna/vicuna-ans-5", "vicuna/vicuna-add-ans-1-1", "vicuna/vicuna-add-ans-1"]
# names = ["llama3.2_3b/llama-740-ans-1"]
# names = ["ms/ms-llama-ans-5"]
names = ["order/order10-llama-ans-5"]
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
    question = items[0]["question"]
    answers = items[0]["answer"]
    final_adv_list.append({
        "id": id,
        "question": question,
        "adv_texts": adv_texts,
        "incorrect_answer": None,
        "answer": answers
    })
# file_path = "llama3.2_3b/GARAG5-ans-740"
# file_path = "ms/GARAG5-ans"
file_path = "order/GARAG5-ans"
with open(f"{file_path}.json", 'w') as f:
    json.dump(final_adv_list, f, indent=4)