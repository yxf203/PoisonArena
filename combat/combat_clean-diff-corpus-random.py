import argparse
import os
import sys
import json
from tqdm import tqdm
import random
import numpy as np
from utils.models import create_model
from utils.utils import load_beir_datasets
from utils.utils import setup_seeds, clean_str
import copy
from utils.eval import find_same_id_item, evaluate
from utils.tools import query_gpt
from utils.prompt import wrap_prompt
import itertools
import pandas as pd
import torch
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

def parse_args():
    parser = argparse.ArgumentParser(description='test')

    # Retriever and BEIR datasets
    parser.add_argument("--eval_model_code", type=str, default="contriever")
    parser.add_argument('--eval_dataset', type=str, default="nq", help='BEIR dataset to evaluate')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument("--orig_beir_results", type=str, default=None, help='Eval results of eval_model on the original beir eval_dataset')
    parser.add_argument("--query_results_dir", type=str, default='main')

    # LLM settings
    parser.add_argument('--model_config_path', default=None, type=str)
    parser.add_argument('--model_name', type=str, default='palm2')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--gpu_id', type=int, default=0)

    # attack
    parser.add_argument('--adv_per_query', type=int, default=5, help='The number of adv texts for each target query.')
    parser.add_argument('--score_function', type=str, default='dot', choices=['dot', 'cos_sim'])
    parser.add_argument('--seed', type=int, default=12, help='Random seed')
    parser.add_argument("--name", type=str, default='debug', help="Name of log and result.")

    # combat_settings
    parser.add_argument('--is_random', type=str, default='False', help='Whether to use random answers in the combat, if False, use all the combinations of the answers.')
    parser.add_argument('--iteration', type=int, default=10, help='the number of combat iterations, if is_random is True')
    parser.add_argument('--combat_n', type=int, default=2, help='the number of attackers in the combat')

    args = parser.parse_args()
    print(args)
    return args


def init_db(args):
    # load BEIR top_k results  
    if args.orig_beir_results is None: 
        print(f"Please evaluate on BEIR first -- {args.eval_model_code} on {args.eval_dataset}")
        # Try to get beir eval results from ./beir_results
        print("Now try to get beir eval results from results/beir_results/...")
        if args.split == 'test':
            args.orig_beir_results = f"../beir_results/{args.eval_dataset}-{args.eval_model_code}-new.json"
        elif args.split == 'dev':
            args.orig_beir_results = f"../beir_results/{args.eval_dataset}-{args.eval_model_code}-dev.json"
        if args.score_function == 'cos_sim':
            args.orig_beir_results = f"../beir_results/{args.eval_dataset}-{args.eval_model_code}-cos.json"
        assert os.path.exists(args.orig_beir_results), f"Failed to get beir_results from {args.orig_beir_results}!"
        print(f"Automatically get beir_resutls from {args.orig_beir_results}.")
    with open(args.orig_beir_results, 'r') as f:
        results = json.load(f)
    # assert len(qrels) <= len(results)
    print('Total samples:', len(results))
    
    return results

def get_all_ans(answers, attackers, id, flag = 0):
    """
    falg: 0, list without source; 1, list with source
    """
    for attacker in attackers:
        with open(attacker["path"], "r") as f:
            adv_texts = json.load(f)
        item = find_same_id_item(adv_texts, id)
        if item["incorrect_answer"] != None:
            if flag == 0:
                answers.append(item["incorrect_answer"])
            elif flag == 1:
                answers.append({
                    "attack": attacker["name"],
                    "answer": item["incorrect_answer"]
                })
    return answers



# used for cp suffixes!
def generate_valid_suffixes_efficient(n, attackers, cp_index):
    all_values = list(range(1, n + 1))
    cp_suffix = random.choice([1, 2, 3])
    
    remaining_values = [v for v in all_values if v != cp_suffix]
    other_indices = [i for i in range(len(attackers)) if i != cp_index]
    other_suffixes = random.sample(remaining_values, len(attackers) - 1)

    suffixes = [None] * len(attackers)
    suffixes[cp_index] = cp_suffix
    for i, idx in enumerate(other_indices):
        suffixes[idx] = other_suffixes[i]
    
    return suffixes

        


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu_id)
    device = 'cuda'
    setup_seeds(args.seed)

    # load the corpus
    if args.eval_dataset == 'msmarco':
        corpus, queries, qrels = load_beir_datasets('msmarco', 'train')
    else:
        corpus, queries, qrels = load_beir_datasets(args.eval_dataset, args.split)


    if args.model_config_path == None:
        args.model_config_path = f'model_configs/{args.model_name}_config.json'
    
    # Load generative model such as llama
    llm = create_model("../"+args.model_config_path)
    results = init_db(args)

    all_attackers = [
        {
            "nick_name": "CP",
            "name": "Cpoison5-ans",
            "base_path":f"data/combat_data/single_data/nq/{args.model_name}/cpoison/cpoison5-inco-ans.json",
            "adv_per_query": 5
        },
        {
            "nick_name": "GASLITE",
            "name": "GASLITE5-ans",
            "base_path": f"data/combat_data/single_data/nq/gaslite/gaslite-dcorpus-ans.json",
            "adv_per_query": 5
        },
        {
            "nick_name": "PB",
            "name": "PoisonedRAG5-black-suffix-ans",
            "base_path":f"data/combat_data/single_data/nq/P/P5-black-suffix-dcorpus-ans.json",
            "adv_per_query": 5
        },
        {
            "nick_name": "PW",
            "name": "PoisonedRAG5-white-suffix-ans",
            "base_path":f"data/combat_data/single_data/nq/P/P5-white-suffix-dcorpus-ans.json",
            "adv_per_query": 5
        },
        {
            "nick_name": "A",
            "name": "AdvDec5-dcorpus-ans",
            "base_path":f"data/combat_data/single_data/nq/{args.model_name}/AdvDec/AdvDec5-dcorpus-dot-ans.json",
            "adv_per_query": 5
        },
        {
            "nick_name": "CA",
            "name": "corpus5-answer-ans",
            "base_path":f"data/combat_data/single_data/nq/corpus/corpus-dcorpus-answer-ans.json",
            "adv_per_query": 5
        },
        {
            "nick_name": "GAR",
            "name": "GARAG5-ans",
            "base_path":f"data/combat_data/single_data/nq/{args.model_name}/G/GARAG5-ans.json",
            "path":f"data/combat_data/single_data/nq/{args.model_name}/G/GARAG5-ans.json",
            "adv_per_query": 5
        },
    ]
    all_results = {}
    combat_n = args.combat_n
    print("combat_n:", combat_n)

    all_combinations = list(itertools.combinations(all_attackers, combat_n))

    start = 0
    # all_combinations = filtered_combinations
    end = len(all_combinations)

    print("start end:", start, end)
    all_combinations = all_combinations[start:end]
    for attackers in all_combinations:
        if attackers[0]["nick_name"] != "GAR":
            temp_path = attackers[0]["base_path"].replace("ans.json", f"ans1.json")
        else:
            temp_path = attackers[0]["base_path"]
        with open(temp_path, "r") as f:
            adv_texts = json.load(f)
            # 2 places should be changed
            adv_texts = adv_texts[:100]
        original_db = []
        missing_ids = 0
        # for item in adv_texts:
        #     id = item["id"]
        for item in adv_texts:
            id = item["id"]
            if id not in results:
                print(f"Warning: ID {id} not found in results, skipping...")
                missing_ids += 1
                continue
            question = item["question"]
            topk_idx = list(results[id].keys())[:args.top_k*2]
            topk_results = [{'score': results[id][idx], 'context': corpus[idx]['text'], 'source': 'original'} for idx in topk_idx] 
            temp = {
                "id": id,
                "question": question,
                "ground truth answer": item["answer"],
                "ctxs": topk_results,
            }
            original_db.append(temp)

        print("original db length: ", len(original_db))
        is_random = bool(args.is_random)
        print("is_random:", is_random)
        iteration = args.iteration
        n = 6
        # Check if any attacker has nick_name "CP" and get its index
        cp_index = next((i for i, attacker in enumerate(attackers) if attacker["nick_name"] == "CP"), None)
        if is_random:
            permutations = range(iteration)
        else:
            # permutations = [list(p) for p in itertools.permutations(range(1, n+1))]
            permutations = list(itertools.permutations(range(1, n+1), combat_n))

            # Filter permutations where the element at cp_index is greater than 3
            if cp_index is not None:
                permutations = [p for p in permutations if p[cp_index] <= 3]
                print("len of permutations:", len(permutations)) 
        total_file_name = ""
        start_time = time.time()

        scores_record = {}
        tie_total = []
        for attacker in attackers:
            scores_record[attacker["name"]] = []
            total_file_name += attacker["nick_name"] + "-"
        if is_random:
            total_file_name += 'random-' + str(iteration)
        for p in permutations:
            print("permutation:", p)
            if is_random:
                suffixes = random.sample(range(1, n + 1), len(attackers))
                # get permutations where the element at cp_index is less than 3
                if cp_index is not None:
                    suffixes = generate_valid_suffixes_efficient(n, attackers, cp_index)
                    print("CP!suffixs:", suffixes)
                print("suffixs:", suffixes)
            else:
                suffixes = p
            for attacker, suffix in zip(attackers, suffixes):
                if attacker["nick_name"] == "GAR":
                    continue
                attacker["path"] = attacker["base_path"].replace("ans.json", f"ans{suffix}.json")
                # attacker["name"] = attacker["name"].replace("ans", f"ans{suffix}")


            
            attacked_db = copy.deepcopy(original_db)
            filename = ""
            for attacker in attackers:
                print(f"{attacker['name']} is attacking……")
                with open(attacker["path"], "r") as f:
                    adv_texts = json.load(f)
                    adv_texts = adv_texts[:100]
                ans_list = []

                for item in adv_texts:
                    id = item["id"]
                    # topk_idx = list(results[id].keys())[:args.top_k]
                    # topk_results = [{'id': idx, "has_answer":True, 'score': results[id][idx], 'context': corpus[idx]['text']} for idx in topk_idx]    
                    info = find_same_id_item(attacked_db, id)
                    if info is None:
                        print(f"Warning: ID {id} not found in attacked_db, skipping...")
                        continue 
                    topk_results = info["ctxs"]
                    for index, i in enumerate(item["adv_texts"]):
                        i["source"] = attacker["name"]
                        topk_results.append(i)
                    # info["ctxs"] = topk_results
                    # temp = {
                    #     "id": item["id"],
                    #     "question": item["question"],
                    #     "ctxs": topk_results,
                    # }
                    answers = item["answer"]
                    if item["incorrect_answer"] == None:
                        answers = get_all_ans(answers, attackers, id)
                    print("answers:", answers)
                    ans_list.append({
                        "id": item["id"],
                        "adv_texts": item["adv_texts"],
                        "incorrect answer": item["incorrect_answer"],
                        "answer": answers
                    })

                    # attacked_db.append(temp)
                attacker["ans_list"] = ans_list
                filename += attacker["nick_name"] + "-"
            filename += f"{args.top_k}"
            filename += str(suffixes)
            if is_random:
                filename += "random-" + str(iteration)
            # here I previously process the answer of llm, in case that the random output affects the fairness
            # ensure that in one battle, the answer to the same question is same.
            for d in attacked_db:
                id = d["id"]
                topk_results = d["ctxs"]
                question = d["question"]
                topk_results = sorted(topk_results, key=lambda x: float(x['score']), reverse=True)
                topk_contents = [topk_results[j]["context"] for j in range(args.top_k)]
                # query_prompt = wrap_prompt(question, topk_contents, prompt_id=4)
                query_prompt = wrap_prompt(question, topk_contents)
                d["input_prompt"] = query_prompt
                d["ctxs"] = sorted(d["ctxs"], key=lambda x: float(x['score']), reverse=True)
                # add llm response to corresponding query
                response = llm.query(query_prompt)
                d["response"] = response
                # get all target answers in order to justify the tie
                d["answers"] = get_all_ans([], attackers, id, 1)


            records = []
            for attacker in attackers:
                print(f"{attacker['name']} is evaluating……")
                asr_mean, ret_precision_mean, ret_recall_mean, ret_f1_mean = evaluate(args, llm, attacked_db, attacker["ans_list"], adv_per_query=attacker["adv_per_query"], filename=filename, name=attacker['name'])
                record = {
                    "ASR": asr_mean,
                    "Precision": ret_precision_mean,
                    "Recall": ret_recall_mean,
                    "F1": ret_f1_mean,
                    "name": attacker["name"]
                }
                records.append(record)
                scores_record[attacker["name"]].append(record)
            
            tie_count = 0
            tie_detail = []
            # fake_tie = []
            details = []
            # justify the tie and record review attack detail
            for d in attacked_db:
                target_answers = d["answers"]
                response = d["response"]

                detail = {
                    "id": d["id"],
                    "question": d["question"],
                    "top-k": d["ctxs"][:args.top_k],
                    "input_prompt": d["input_prompt"],
                    "attacked_LLM_output": response,
                    "all_answers": target_answers,
                    "ground truth answer": d["ground truth answer"]
                }
                if "WITHOUT GARAG RES" in d:
                    detail["WITHOUT GARAG RES"] = d["WITHOUT GARAG RES"]
                details.append(detail)
                if all(clean_str(a["answer"]) in clean_str(response) for a in target_answers):
                    tie_count += 1
                    tie_detail.append(detail)
            tie_total.append(tie_count / len(attacked_db))
            records.append({
                "tie_rate": tie_count / len(attacked_db),
                "from": filename
            })

            output_dir = f"combat_details/{args.model_name}/topk{args.top_k}"
            if is_random:
                output_dir += f"-random-i{iteration}"
            # os.makedirs(output_dir, exist_ok=True)

            # file_path = os.path.join(output_dir, filename)
            # with open(f"{file_path}_{args.score_function}.json", 'w') as f:
            #     json.dump(records, f, indent=4)

            
            tie_dir = f"./{output_dir}/tie"
            os.makedirs(tie_dir, exist_ok=True)
            with open(f"{tie_dir}/{filename}.json", 'w') as f:
                json.dump(tie_detail, f, indent=4)

            # Create directory and save overview
            overview_dir = f"./{output_dir}/overview"
            os.makedirs(overview_dir, exist_ok=True)
            with open(f"{overview_dir}/{filename}.json", 'w') as f:
                json.dump(details, f, indent=4)
            print(f"Detail also have saved!")
        
        
        # process the average of all results
        averages = {}
        
        for attacker_name, records in scores_record.items():

            asr_values = [record["ASR"] for record in records]

            precision_values = [record["Precision"] for record in records]
            recall_values = [record["Recall"] for record in records]
            f1_values = [record["F1"] for record in records]
            

            avg_asr = sum(asr_values) / len(asr_values) if asr_values else 0
            avg_precision = sum(precision_values) / len(precision_values) if precision_values else 0
            avg_recall = sum(recall_values) / len(recall_values) if recall_values else 0
            avg_f1 = sum(f1_values) / len(f1_values) if f1_values else 0

            print("avg_asr:", avg_asr)
            averages[attacker_name] = {
                "ASR_mean": avg_asr,
                "Precision_mean": avg_precision,
                "Recall_mean": avg_recall,
                "F1_mean": avg_f1
            }
        all_output_dir = f"all_results/{args.model_name}/topk{args.top_k}"
        if is_random:
            all_output_dir += f"-random-i{iteration}"
        os.makedirs(all_output_dir, exist_ok=True)
        all_results[total_file_name] = averages
        # with open(f"{all_output_dir}/combat_{combat_n}_{start}_{end}_add_g_cp.json", 'w') as f:
        with open(f"{all_output_dir}/combat_{combat_n}_{start}_{end}.json", 'w') as f:
            json.dump(all_results, f, indent=4)

        end_time = time.time()
        print(f"costing time: {end_time - start_time}")
if __name__ == '__main__':
    main()