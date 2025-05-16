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
os.environ["CUDA_VISIBLE_DEVICES"] = "6"


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

    print('Total samples:', len(results))
    return results

def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu_id)
    args.model_config_path = f'../model_configs/{args.model_name}_config.json'
    # llm = create_model(args.model_config_path)

    results = init_db(args)


    if args.eval_dataset == 'msmarco':
        corpus, queries, qrels = load_beir_datasets('msmarco', 'train')

    else:
        corpus, queries, qrels = load_beir_datasets(args.eval_dataset, args.split)

    # the file is going to get the id and question
    with open("dataset/new_diff/incorrect_ans_docs/split_by_answer_position/nq-inco_ans_1.json", "r") as f:
        adv_texts = json.load(f)
    trans2GARAG = []

    for (key, item) in adv_texts.items():
        
        id = item["id"]
        question = item["question"]
        print(id)

        topk_idx = list(results[id].keys())[:args.top_k]  
        topk_results = [{'id': idx,"has_answer":True, 'score': results[id][idx], 'context': corpus[idx]["title"] + " " + corpus[idx]["text"] if len(corpus[idx]["title"]) > 0 else corpus[idx]["text"]} for idx in topk_idx]    
        top_results = sorted(topk_results, key=lambda x: x['score'], reverse=True)[:5]
        trans = {
            "id": item["id"],
            "question": question,
            # "answers":[item["answer"]],
            "answers":[item["correct answer"]],
            "ctxs": top_results
        }
        trans2GARAG.append(trans)

    with open("dataset/new_diff/diff-corpus/origin-4GARAG-dot.json", "w") as f:
        json.dump(trans2GARAG, f, indent=4)
    
if __name__ == '__main__':
    main()