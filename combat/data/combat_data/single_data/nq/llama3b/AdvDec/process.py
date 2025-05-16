import json

# Define file paths
file1_path = 'AdvDec5-dcorpus-dot-ans1.json'
file2_path = 'AdvDec5-dcorpus-dot-ans5.json'

# Load data from files
def load_ids(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {item['id'] for item in data}  # Assuming each element has an 'id' field

# Load IDs from both files
ids1 = load_ids(file1_path)
ids2 = load_ids(file2_path)

# Compare ID sets
only_in_file1 = ids1 - ids2
only_in_file2 = ids2 - ids1

# Print results
print("IDs only in file1:", only_in_file1)
print("IDs only in file2:", only_in_file2)