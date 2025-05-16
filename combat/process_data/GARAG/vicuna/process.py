import json

# Define the path to the JSON file
file_path = "/data2/chenliuji/combat/main/process_data/GARAG/vicuna/GARAG5-ans.json"

# Read and process the JSON file
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

count = 0
# Iterate through each element and check the condition
for item in data:
    adv_texts = item.get("adv_texts", [])
    if len(adv_texts) == 5:
        # print(item.get("id"))
        count += 1

print("The number of adv_texts with length 5 is:", count)