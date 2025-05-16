import json
import os
import glob


output_folder = "../results/for-all_llama3_8b"

nq_test_file = "../data/process/ans_data/nq-inco_ans_6.json"

test_file = "../data/process/origin-4GARAG-dot.json" 


with open(nq_test_file, "r") as f:
    nq_test_data = json.load(f)


with open(test_file, "r") as f:
    test_data = json.load(f)


results = []

start = 0
end = 99

for i in range(start, end + 1):  
    succ_file = os.path.join(output_folder, f"attack_{i}_succ.json")
    fail_file = os.path.join(output_folder, f"attack_{i}_fail.json")

    if os.path.exists(succ_file):
        with open(succ_file, "r") as f:
            succ_data = json.load(f)
        succ_data = succ_data["0-success"]
        id = succ_data["id"]
        success_cotrol = succ_data["success_cotrol"]
        # status = "succ"
    elif os.path.exists(fail_file):
        with open(fail_file, "r") as f:
            fail_data = json.load(f)
        fail_data = fail_data["0-fail"]
        id = fail_data["id"]
        success_cotrol = fail_data["fail_cotrol"]
        # status = "fail"
    else:
        print(f"nonono {i}")
        continue 


    if id in nq_test_data:
        nq_entry = nq_test_data[id]

        test_entry = next((item for item in test_data if item["id"] == id), None)

        result = {
            "id": id,
            "question": nq_entry["question"],
            "adv_texts": [
                {
                    "context": ctx["context"] + success_cotrol
                } for ctx in test_entry["ctxs"]
            ],
            "incorrect_answer": nq_entry["incorrect answer"],
            "answer": [nq_entry["correct answer"]],
        }

        results.append(result)

output_file = f"new-diff-corpus/forall_{start}_{end}.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=4)

print(f"done!")