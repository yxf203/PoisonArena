import json
with open("ms/ms-inco_ans_6.json", "r") as f:
    data = json.load(f)

transformed_data = []
count = 0
for key, value in data.items():
    id = value["id"]
    transformed_data.append({
        "id": id,
        "index_cluster": count,
        # "queries": value["questions"][:1],
        "queries": [value["question"]],
        # "queries": data_serial[id]["serial_questions"],
        "misinfo": value["adv_texts"]
    })
    count += 1


with open("ms/ms-inco_ans_6-4ad.json", "w") as f:
    json.dump(transformed_data, f, indent=4, ensure_ascii=False)

print("done!")