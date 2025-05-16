import os
import json
import sys
sys.path.append(".")
from scripts.utils import format_print, write_csv


results = [["results/tutorials_mistral", "results/tutorials_llama2", "results/tutorials_llama2_13b", "results/tutorials_vicuna", "results/tutorials_vicuna_13b"], ["results/reviews_mistral", "results/reviews_llama2", "results/reviews_llama2_13b", "results/reviews_vicuna", "results/reviews_vicuna_13b"]]
Order = {"llama2":[0, "Llama2-7b"], "vicuna":[1, "Vicuna-7b"], "mistral":[2, "Mistral-7b"], "llama2_13b": [3, "Llama2-13b"], "vicuna_13b":[4, "Vicuna-13b"]}


if __name__ == "__main__":

    table9 = [["Border", "Llama2_7b", "Vicuna_7b", "Mistral_7b", "Llama2_13b", "Vicuna_13b"]]
    table9.append(["----", 0, 0, 0, 0, 0])
    table9.append(["====", 0, 0, 0, 0, 0])
    total = [0 for i in range(5)]
    for index, mode_dirs in enumerate(results):
        for target_folder in mode_dirs:
            file_list = os.listdir(target_folder)
            splits = target_folder.split("_")
            if len(splits) == 3:
                model_name = splits[-2] +"_"+ splits[-1]
            else:
                model_name = splits[-1]
            print("Evaluating ASR of Border Defense.")
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
            with open(os.path.join(target_folder, "border_defense.json"), "r", ) as f:
                res_all = json.load(f)
            tot = 0
            succ_1 = []
            succ_2 = []
            order = Order[model_name][0]

            for content in res_all:
                if "success" in content:
                    succ_1.append(content['index'])
                if "success2" in content:
                    succ_2.append(content['index'])
                tot += 1
            print("defense with border ----, success indexes are", succ_1)
            print("defense with border ====, success indexes are", succ_2)

            table9[1][order+1]+=len(succ_1)
            table9[2][order+1]+=len(succ_2)
            total[order] += tot
            
            # ASR = len(succ) / tot
    print(total)
    for i in range(5):
        table9[1][i+1] /= total[i]
        table9[2][i+1] /= total[i]
    print("\n")
    print("Table 9: Attack results on the structured prompt template.  “Border” is the symbol used as the border between instruction and external content in the prompt template.")
    format_print(table9)
    write_csv(table9, "all_tables/table9.csv")

    
                    
