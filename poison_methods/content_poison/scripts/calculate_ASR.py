import csv
import json
import os
from transformers import AutoTokenizer
import sys
sys.path.append(".")
from scripts.utils import format_print, write_csv 

# results = [["results/tutorials_mistral", "results/tutorials_llama2", "results/tutorials_llama2_13b", "results/tutorials_vicuna", "results/tutorials_vicuna_13b"], ["results/reviews_mistral", "results/reviews_llama2", "results/reviews_llama2_13b", "results/reviews_vicuna", "results/reviews_vicuna_13b"]]
results = [["results/output0-100_llama3_8b"], ["results/output0-100_llama3_8b"]]

# Order = {"llama2":[0, "Llama2-7b"], "vicuna":[1, "Vicuna-7b"], "mistral":[2, "Mistral-7b"], "llama2_13b": [3, "Llama2-13b"], "vicuna_13b":[4, "Vicuna-13b"]}
Order = {"llama3_8b":[0, "Llama3_8b"]}
if __name__ == "__main__":
    table = [[ "LLMs", "Word-Level Attack", "", "", "", "", "Whole-Content Attack", "", "","", "", "Avg ASR"]]
    # table = [[ "LLMs", "Word-Level Attack", "", "", "", "", "Avg ASR"]]
    table.append(["", "ASR", "Iteration", "Trigger", "Request", "Response", "ASR", "Iteration", "Trigger", "Request", "Response", "Avg ASR"])
    # table.append(["", "ASR", "Iteration", "Trigger", "Request", "Response", "Avg ASR"])
    for key, item in Order.items():
        table.append([item[1]])
    for results_i, attack_mode in zip(results, ["Word-Level", "Whole-Content"]):
    # for results_i, attack_mode in zip(results, ["Word-Level"]):
        for target_folder in results_i:

            file_list = os.listdir(target_folder)
            splits = target_folder.split("_")
            if len(splits) == 3:
                model_name = splits[-2] +"_"+ splits[-1]
            else:
                model_name = splits[-1]
            
            print("process result of ", model_name, "on mode", attack_mode)
            
            try:
                # import configs.mistral as mistral
                # init__.llama2
                model_config = __import__("configs."+model_name,  fromlist=["configs"])
                model_config = model_config.get_config()
            except Exception as e:
                print("fail to load config of", model_name)
                print(e)
                exit(1)
            line = Order[model_name][0] + 2
            
            tokenizer = tokenizer = AutoTokenizer.from_pretrained(
                model_config.tokenizer_paths[0],
                trust_remote_code=True,
                **model_config.tokenizer_kwargs[0]) 
            seq, req, res, Iter, index = [], [], [], [], []
            succ_index = []
            succ_num = 0
            all_num = 0
            # i = -1
            for res_file in file_list:
                file_name, file_extension = os.path.splitext(res_file)
                file_name_split = file_name.split("_")
                if file_extension == ".json" and "attack_" in file_name:
                    all_num += 1
                    file_path = os.path.join(target_folder, res_file)
                    with open(file_path, "r", ) as f:
                        content = json.load(f)
                    index_i = int(file_name_split[1])
                    if file_name_split[-1] == "succ":
                        succ_num += 1
                        succ_index.append(index_i)
                        succ_toks = content["success_toks"]
                        control_length = len(succ_toks["success_control_toks"])
                        success_control_toks = succ_toks["success_control_toks"]
                        input_length = len(succ_toks["success_input_idx"]) - control_length
                        success_input_idx = succ_toks["success_input_idx"]

                        succ_output = content["0-success"]["success_generate"]
                        succ_output = str(succ_output[0][0])
                        # print('succ output:',succ_output)
                        succ_output_tok = tokenizer(succ_output, add_special_tokens=False).input_ids
                        output_length = len(succ_output_tok)
                        iteration = content["0-success"]["success_iteration"]
                        # print(succ_output)
                        # print(control_length)
                        # print(input_length)
                    elif file_name_split[-1]  == "fail":
                        input = content["config"]["input"]
                        target = content["config"]["target"]
                        control = content["controls"][-1]

                        input_length = len(tokenizer(input, add_special_tokens=False).input_ids)
                        # print('input tok:', input_length)
                        # print('input tok:', tokenizer(input, add_special_tokens=False).input_ids)
                        control_length = len(tokenizer(control, add_special_tokens=False).input_ids)
                        output_length = len(tokenizer(target, add_special_tokens=False).input_ids)
                        iteration = model_config.n_steps

                    seq.append(control_length)
                    req.append(input_length)
                    res.append(output_length)
                    Iter.append(iteration)
                    index.append(index_i)
            print("\t success index:", succ_index)
            print("\t attack sequence length:", seq)
            print("\t input request length:", req)
            print("\t output response length:", res)
            avg_seq = sum(seq) / len(seq)
            avg_req = sum(req) / len(req)
            avg_res = sum(res) / len(res)
            avg_Iter = sum(Iter) / len(Iter)
            ASR = succ_num / all_num
            # print(table)
            # print(line)
            table[line]+=[ASR, avg_Iter, avg_seq, avg_req, avg_res]
    for i in range(2, len(table)):
        table[i].append((table[i][1] + table[i][6]) / 2)
    avg = []
    for i in range(1, 11):
        avg.append(sum([table[j][i] for j in range(2, len(table))]) / 5)
    table.append(["Average", *avg, (avg[0]+avg[5])/2])
    print("\n")
    print("Table 2: The effectiveness of content poisoning on various content and LLMs. “Trigger”, “Request”, and “Response” refer to the token lengths of the generated trigger sequence, augmented request, and output response, respectively.")
    print("+------------+-------+-----------+---------+---------+----------+--------+-----------+---------+---------+----------+---------+")
    print("|    LLM     |                  Word-Level Attack               |                  Whole-Content Attack             | Avg ASR |")
    format_print(table[1:])
    write_csv(table, "all_tables/table2.csv")

