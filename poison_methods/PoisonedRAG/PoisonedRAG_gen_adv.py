import argparse
import os
import sys
import json
from tqdm import tqdm
import random
import numpy as np
from src.utils import load_beir_datasets, load_models
from src.utils import save_results, load_json, setup_seeds, clean_str, f1_score
from src.attack import Attacker
from src.prompts import wrap_prompt
import torch
import time

print("CUDA:",torch.cuda.is_available()) 
def parse_args():
    parser = argparse.ArgumentParser(description='test')

    # Retriever and BEIR datasets
    parser.add_argument("--eval_model_code", type=str, default="contriever")
    parser.add_argument('--eval_dataset', type=str, default="nq", help='BEIR dataset to evaluate')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument("--orig_beir_results", type=str, default=None, help='Eval results of eval_model on the original beir eval_dataset')
    parser.add_argument("--query_results_dir", type=str, default='attack_results', help='Directory to save the query results')
    parser.add_argument('--attack_data_dir', type=str, default='dataset/new_diff/incorrect_ans_docs/ms/split_by_answer_position', help='Directory to load the query with adv texts')
    parser.add_argument('--attack_data_name', type=str, default='ms-inco_ans_6', help='Name of the query with adv texts')
    # LLM settings
    parser.add_argument('--model_config_path', default=None, type=str)
    parser.add_argument('--model_name', type=str, default='llama8b')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--gpu_id', type=int, default=0)

    # attack
    parser.add_argument('--attack_method', type=str, default='hotflip')
    parser.add_argument('--adv_per_query', type=int, default=5, help='The number of adv texts for each target query.')
    parser.add_argument('--score_function', type=str, default='dot', choices=['dot', 'cos_sim'])
    parser.add_argument('--repeat_times', type=int, default=100, help='repeat several times to compute average')
    parser.add_argument('--M', type=int, default=1, help='one of our parameters, the number of target queries')
    parser.add_argument('--seed', type=int, default=12, help='Random seed')

    # OUTPUT
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save the results')
    args = parser.parse_args()
    print(args)
    return args


def main():
    args = parse_args()

    torch.cuda.set_device(args.gpu_id)
    device = 'cuda'
    in_data_name = args.attack_data_name
    setup_seeds(args.seed)
    if args.model_config_path == None:
        args.model_config_path = f'../model_configs/{args.model_name}_config.json'


    incorrect_answers = load_json(f'{args.attack_data_dir}/{in_data_name}.json')
    print(f"loading from {in_data_name}")
    incorrect_answers = list(incorrect_answers.values())

    # load BEIR top_k results  
    if args.orig_beir_results is None: 
        print(f"Please evaluate on BEIR first -- {args.eval_model_code} on {args.eval_dataset}")
        # Try to get beir eval results from ./beir_results
        print("Now try to get beir eval results from results/beir_results/...")
        if args.split == 'test':
            args.orig_beir_results = f"results/beir_results/{args.eval_dataset}-{args.eval_model_code}-new.json"
            # args.orig_beir_results = f"results/beir_results/{args.eval_dataset}-{args.eval_model_code}-cos.json"
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


    if args.attack_method not in [None, 'None']:
        # Load retrieval models
        model, c_model, tokenizer, get_emb = load_models(args.eval_model_code)
        model.eval()
        model.to(device)
        c_model.eval()
        c_model.to(device) 
        attacker = Attacker(args,
                            model=model,
                            eval_name=f"{in_data_name}",
                            c_model=c_model,
                            tokenizer=tokenizer,
                            get_emb=get_emb,
                            adv_texts_dir=f'{args.query_results_dir}',
                            ) 
    

    iter_results = []
    iter_times = []
    for iter in range(args.repeat_times):
        start_time = time.time()
        print(f'######################## Iter: {iter+1}/{args.repeat_times} #######################')

        target_queries_idx = range(iter * args.M, iter * args.M + args.M)

        target_queries = [incorrect_answers[idx]['question'] for idx in target_queries_idx]
        
        if args.attack_method not in [None, 'None']:
            for i in target_queries_idx:
                top1_idx = list(results[incorrect_answers[i]['id']].keys())[0]
                top1_score = results[incorrect_answers[i]['id']][top1_idx]
                target_queries[i - iter * args.M] = {'query': target_queries[i - iter * args.M], 'top1_score': top1_score, 'id': incorrect_answers[i]['id']}
                
            adv_text_groups = attacker.get_attack(target_queries)

            # print(adv_text_groups)
            adv_text_list = sum(adv_text_groups, []) # convert 2D array to 1D array

            adv_input = tokenizer(adv_text_list, padding=True, truncation=True, return_tensors="pt")
            adv_input = {key: value.cuda() for key, value in adv_input.items()}
            with torch.no_grad():
                adv_embs = get_emb(c_model, adv_input)        
                      
        # ret_sublist=[]
        
        for i in target_queries_idx:
            top_k_adv = []

            iter_idx = i - iter * args.M # iter index
            print(f'############# Target Question: {iter_idx+1}/{args.M} #############')
            question = incorrect_answers[i]['question']
            print(f'Question: {question}\n') 
            
            incco_ans = incorrect_answers[i]['incorrect answer']            

            # compute the similarity
            if args.attack_method not in [None, 'None']: 
                query_input = tokenizer(question, padding=True, truncation=True, return_tensors="pt")
                query_input = {key: value.cuda() for key, value in query_input.items()}
                with torch.no_grad():
                    query_emb = get_emb(model, query_input) 
                for j in range(len(adv_text_list)):

                    adv_emb = adv_embs[j, :].unsqueeze(0) 
                    # similarity     
                    if args.score_function == 'dot':
                        adv_sim = torch.mm(adv_emb, query_emb.T).cpu().item()
                    elif args.score_function == 'cos_sim':
                        adv_sim = torch.cosine_similarity(adv_emb, query_emb).cpu().item()
                                            
                    top_k_adv.append({'score': adv_sim, 'context': adv_text_list[j]})

        iter_results.append(
            {
                "id":incorrect_answers[i]['id'],
                "question": question,
                "adv_texts": top_k_adv,
                # "input_prompt": query_prompt,
                "incorrect_answer": incco_ans,
                "answer": incorrect_answers[i]['correct answer'],
                "golden_docs": incorrect_answers[i]["golden_docs"],
                "query_type": incorrect_answers[i]["query_type"]
            }
        )
        with open(f"{args.output_dir}/{in_data_name}-{args.attack_method}-suffix-{args.score_function}.json", "w") as f:
            json.dump(iter_results, f, indent=4)
        end_time = time.time()
        iter_times.append(end_time - start_time)
        print(f"Iter {iter+1} time: {end_time - start_time:.2f}s")
    avg_time = sum(iter_times) / len(iter_times)
    attack_time_data = {
        "average_time": avg_time,
        "iter_times": iter_times
    }
    output_path = f"{args.output_dir}/{in_data_name}_{args.attack_method}_{args.score_function}_time.json"
    with open(output_path, "w") as f:
        json.dump(attack_time_data, f, indent=4)
    print(f"Attack times and average time written to {output_path}")
    return iter_results

if __name__ == "__main__":
    main()