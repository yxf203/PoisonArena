import os
import json
import sys
sys.path.append(".")
from scripts.utils import format_print, write_csv


results = [["results/tutorials_mistral", "results/tutorials_llama2", "results/tutorials_llama2_13b", "results/tutorials_vicuna", "results/tutorials_vicuna_13b"], ["results/reviews_mistral", "results/reviews_llama2", "results/reviews_llama2_13b", "results/reviews_vicuna", "results/reviews_vicuna_13b"]]
Order = {"llama2":[0, "Llama2-7b"], "vicuna":[1, "Vicuna-7b"], "mistral":[2, "Mistral-7b"], "llama2_13b": [3, "Llama2-13b"], "vicuna_13b":[4, "Vicuna-13b"]}


if __name__ == "__main__":
    table3 = [["Modes", "Llama2_7b", "Vicuna_7b", "Mistral_7b", "Llama2_13b", "Vicuna_13b", "Average"]]
    table3.append(["Word-Level", 0, 0, 0, 0, 0 ])
    table3.append(["Whole-Content", 0, 0, 0, 0, 0])
    table4 = [["Quantization", "Llama2_7b", "Vicuna_7b", "Mistral_7b", "Llama2_13b", "Vicuna_13b"]]
    print("Evaluating ASR of different prompt.")
    for index, mode_dirs in enumerate(results):
        for target_folder in mode_dirs:
            file_list = os.listdir(target_folder)
            splits = target_folder.split("_")
            if len(splits) == 3:
                model_name = splits[-2] +"_"+ splits[-1]
            else:
                model_name = splits[-1]
            print("process result of ", model_name, "on mode", "Word-Level" if index == 0 else "Whole-Content")
            try:
                # import configs.mistral as mistral
                # init__.llama2
                model_config = __import__("configs."+model_name,  fromlist=["configs"])
                model_config = model_config.get_config()
            except Exception as e:
                print("fail to load config of", model_name)
                print(e)
                exit(1)
            with open(os.path.join(target_folder, "prompt_transfer.json"), "r", ) as f:
                res = json.load(f)
            tot = 0
            succ = []
            for content in res:
                assert "is_success" in content, ("is_success is not labeled for index " + str(content["index"]))
                assert content["is_success"] in [0, 1], "is_success is not 0 or 1"
                if content["is_success"] == 1:
                    succ.append(content['index'])
                tot += 1
            print("success indexes are", succ)
            ASR = len(succ) / tot
            table3[index+1][Order[model_name][0] + 1] = ASR
        table3[index+1].append(sum(table3[index+1][1:])/5)
    # print(table3)
    print("Table 3: The ASR of trigger sequences when attacking different augmented requests.")
    format_print(table3)
    write_csv(table3, "all_tables/table3.csv")
    