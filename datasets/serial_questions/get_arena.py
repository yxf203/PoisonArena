import json

# Load the JSON file
input_file = 'nq-all-adv-docs-with-serial-q.json'
output_file = 'process_get_serials.json'

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Initialize the output dictionary and counter
output_data = {}
count = 0

# Process each entry in the input data
for key, value in data.items():
    serial_questions = value.get("serial_questions", [])
    for question in serial_questions:
        new_key = f"s{count}"
        output_data[new_key] = {
            "id": new_key,
            "question": question,
            "correct answer": value["correct answer"],
            "all_adv_docs": value["all_adv_docs"]
        }
        count += 1
# Filter out entries where the numeric part of the id % 10 < 5
output_data = {k: v for k, v in output_data.items() if int(k[1:]) % 10 >= 5}
# Save the processed data to a new JSON file
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)

print(f"Processed data saved to {output_file}")