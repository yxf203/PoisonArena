
import os
import json
import sys
sys.path.append(".")
from scripts.utils import format_print, write_csv


results = [["results/tutorials_mistral", "results/tutorials_llama2", "results/tutorials_llama2_13b", "results/tutorials_vicuna", "results/tutorials_vicuna_13b"], ["results/reviews_mistral", "results/reviews_llama2", "results/reviews_llama2_13b", "results/reviews_vicuna", "results/reviews_vicuna_13b"]]
Order = {"llama2":[0, "Llama2-7b"], "vicuna":[1, "Vicuna-7b"], "mistral":[2, "Mistral-7b"], "llama2_13b": [3, "Llama2-13b"], "vicuna_13b":[4, "Vicuna-13b"]}

if __name__ == "__main__":

    table5 = [["Models", "Llama2_7b", "Misral_7b", "Average"]]
    table5.append(["ASR",0,0])
    total = [0,0]
    print("Evaluating ASR of finetuned models.")
    for index, mode_dirs in enumerate(results):
        for target_folder in mode_dirs:
            splits = target_folder.split("_")
            if len(splits) == 3:
                model_name = splits[-2] +"_"+ splits[-1]
            else:
                model_name = splits[-1]
            if model_name not in ["llama2", "mistral"]:
                continue
            file_list = os.listdir(target_folder)
            print("process result of ", model_name, "on mode", "Word-Level" if index == 0 else "Whole-Content")
            try:
                model_config = __import__("configs."+model_name,  fromlist=["configs"])
                model_config = model_config.get_config()
            except Exception as e:
                print("fail to load config of", model_name)
                print(e)
                exit(1)
            with open(os.path.join(target_folder, "finetune_transfer.json"), "r", ) as f:
                res = json.load(f)
            succ = []
            tot=0
            for content in res:
                assert "is_success" in content, ("is_success is not labeled for index " + str(content["index"]))
                assert content["is_success"] in [0, 1], "is_success is not 0 or 1"
                if content["is_success"] == 1:
                    succ.append(content['index'])
                tot += 1
            if model_name == "llama2":
                order = 1
            else:
                order = 2
            # print("success indexes are", succ)
            table5[1][order]+=len(succ)
            total[order-1] += tot
    table5[1][1] /= total[0]
    table5[1][2] /= total[1]
    table5[1].append(sum(table5[1][1:])/2)
    print("\n")
    print("Table 5: The ASR of finetuned models.")
    format_print(table5)
    write_csv(table5, "all_tables/table5.csv")
    # print(total)
    
                    
