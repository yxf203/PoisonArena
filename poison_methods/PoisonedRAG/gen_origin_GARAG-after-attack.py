import argparse
import os

import json
from tqdm import tqdm
import random
import numpy as np
from src.models import create_model
from src.utils import load_beir_datasets, load_models
from src.utils import save_results, load_json, setup_seeds, clean_str, f1_score
from src.prompts import wrap_prompt
import torch
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "7"


def parse_args():
    parser = argparse.ArgumentParser(description='test')

    # Retriever and BEIR datasets
    parser.add_argument("--eval_model_code", type=str, default="contriever")
    parser.add_argument('--eval_dataset', type=str, default="nq", help='BEIR dataset to evaluate')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument("--orig_beir_results", type=str, default=None, help='Eval results of eval_model on the original beir eval_dataset')

    # LLM settings
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--model_name', type=str, default='llama8b')
    # attack
    parser.add_argument('--score_function', type=str, default='dot', choices=['dot', 'cos_sim'])


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
            # args.orig_beir_results = f"results/beir_results/{args.eval_dataset}-{args.eval_model_code}-new.json"
            args.orig_beir_results = f"results/beir_results/{args.eval_dataset}-{args.eval_model_code}-new.json"
            # args.orig_beir_results = f"results/beir_results/{args.eval_dataset}-{args.eval_model_code}-cos.json"
            # args.orig_beir_results = f"results/beir_results/serials_contriever_100_new.json"

        elif args.split == 'dev':
            args.orig_beir_results = f"results/beir_results/{args.eval_dataset}-{args.eval_model_code}-dev.json"
        if args.score_function == 'cos_sim':
            args.orig_beir_results = f"results/beir_results/{args.eval_dataset}-{args.eval_model_code}-cos.json"
        assert os.path.exists(args.orig_beir_results), f"Failed to get beir_results from {args.orig_beir_results}!"
        print(f"Automatically get beir_resutls from {args.orig_beir_results}.")
    with open(args.orig_beir_results, 'r') as f:
        results = json.load(f)
    # assert len(qrels) <= len(results)
    print('Total samples:', len(results))
    return results

def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu_id)
    args.model_config_path = f'model_configs/{args.model_name}_config.json'
    # llm = create_model(args.model_config_path)

    results = init_db(args)

    # -----------------------------------------------------------------------------

    # need incorrect_answer(own answer) PRAG

    # load target queries and answers
    # here is the PoisonedRAG's own data
    if args.eval_dataset == 'msmarco':
        corpus, queries, qrels = load_beir_datasets('msmarco', 'train')
        # incorrect_answers = load_json(f'results/adv_targeted_results/{args.eval_dataset}-{s}-{v}.json')
        # random.shuffle(incorrect_answers)    
    else:
        corpus, queries, qrels = load_beir_datasets(args.eval_dataset, args.split)
        # incorrect_answers = load_json(f'results/adv_targeted_results/{args.eval_dataset}-{s}-{v}.json')

    # incorrect_answers = list(incorrect_answers.values())

    # with open("dataset/origin/nq-origin.json", "r") as f:
    # with open("dataset/serials_tmp/origin_blackbox.json", "r") as f:
    # with open("dataset/new_diff/diff-corpus/nq-ans1-dcorpus-sorted.json", "r") as f:
    
    with open("dataset/new_diff/incorrect_ans_docs/split_by_answer_position/nq-inco_ans_1.json", "r") as f:
    # with open("/data2/chenliuji/combat/datasets/incorrect_ans/ms-selected-queries-adv-ans-docs.json", "r") as f:
        adv_texts = json.load(f)
        # adv_texts = adv_texts[500:1000]
    
    with open("/data2/chenliuji/combat/main/data/combat_data/nq/llama8b/AdvDec/new-diff-corpus/AdvDec5-dcorpus-dot-ans2.json", "r") as f:
        attack_texts = json.load(f)
    
    trans2GARAG = []
    # for item in adv_texts:
    for (key, item) in adv_texts.items():
        
        id = item["id"]
        question = item["question"]
        attack_item = next((x for x in attack_texts if x["id"] == id), None)
        if not attack_item:
            print(f"No matching attack item found for id: {id}")
            continue
        print(id)
        # if id == 'test0':
        #     continue
        topk_idx = list(results[id].keys())[:args.top_k]
        # query_prompt = wrap_prompt(question, topk_contents, prompt_id=4)
        # response = llm.query(query_prompt)
        # topk_results = [{'id': idx, "has_answer":True, 'score': results[id][idx], 'context': corpus[idx]["title"] + " " + corpus[idx]["text"] if len(corpus[idx]["title"]) > 0 else corpus[idx]["text"]} for idx in topk_idx]    
        topk_results = [{'id': idx,"has_answer":True, 'score': results[id][idx], 'source': 'original', 'context': corpus[idx]["title"] + " " + corpus[idx]["text"] if len(corpus[idx]["title"]) > 0 else corpus[idx]["text"]} for idx in topk_idx]    
        for index, adv in enumerate(attack_item["adv_texts"]):
            
            adv['id'] = f'adv{index}'
            adv["has_answer"] = True
            adv["source"] = "A"
            topk_results.append(adv)
        # here is the change of the top k results after the attack of PoisonedRAG
        # for index, i in enumerate(item["adv_texts"]):
        #     i['id'] = 'adv' + str(index)
        #     i["has_answer"] = True
        #     topk_results.append(i)
        # print("len!!",len(topk_results))
        top_results = sorted(topk_results, key=lambda x: x['score'], reverse=True)[:5]
        answers = [item["correct answer"]]
        if attack_item["incorrect_answer"] != None:
            answers.append(attack_item["incorrect_answer"])
        trans = {
            "id": item["id"],
            "question": question,
            # "answers":[item["answer"]],
            "answers": answers,
            "ctxs": top_results
        }
        trans2GARAG.append(trans)
        # temp_db.append(temp)
        # print(trans2GARAG)
    # with open("dataset/origin/nq-GARAG-500-1000-1.json", "w") as f:
    # with open("dataset/new_diff/diff-corpus/origin-4GARAG-dot-ms.json", "w") as f:
    with open("after_attack/after-A-ans2.json", "w") as f:
        json.dump(trans2GARAG, f, indent=4)
    
if __name__ == '__main__':
    main()