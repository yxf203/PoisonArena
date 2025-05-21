import argparse
import os
import sys
import json
from tqdm import tqdm
import random
import numpy as np
from src.models import create_model
from src.utils import load_beir_datasets, load_models
from src.utils import save_results, load_json, setup_seeds, clean_str, f1_score
from poison_methods.PoisonedRAG.src.serials_attack import SerialsAttacker
from src.prompts import wrap_prompt
import torch
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.makedirs("combat_results", exist_ok=True)

print("CUDA:",torch.cuda.is_available()) 
def parse_args():
    parser = argparse.ArgumentParser(description='test')

    # Retriever and BEIR datasets
    parser.add_argument("--eval_model_code", type=str, default="contriever")
    parser.add_argument('--eval_dataset', type=str, default="nq", help='BEIR dataset to evaluate')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument("--orig_beir_results", type=str, default=None, help='Eval results of eval_model on the original beir eval_dataset')
    parser.add_argument("--query_results_dir", type=str, default='Padv')

    # LLM settings
    parser.add_argument('--model_config_path', default=None, type=str)
    parser.add_argument('--model_name', type=str, default='llama8b')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--use_truth', type=str, default='False')
    parser.add_argument('--gpu_id', type=int, default=0)

    # attack
    parser.add_argument('--attack_method', type=str, default='hotflip')
    parser.add_argument('--adv_per_query', type=int, default=5, help='The number of adv texts for each target query.')
    parser.add_argument('--score_function', type=str, default='dot', choices=['dot', 'cos_sim'])
    parser.add_argument('--repeat_times', type=int, default=100, help='repeat several times to compute average')
    parser.add_argument('--M', type=int, default=1, help='one of our parameters, the number of target queries')
    parser.add_argument('--seed', type=int, default=12, help='Random seed')
    parser.add_argument("--name", type=str, default='debug', help="Name of log and result.")

    args = parser.parse_args()
    print(args)
    return args


def main():
    args = parse_args()

    s = 1
    v = 0
    # args = parse_args()
    print(args.eval_dataset, " ", s, " ", v)
    torch.cuda.set_device(args.gpu_id)
    device = 'cuda'
    # in_data_name = f"{args.eval_dataset}-test-4.0"
    in_data_name = "nq-inco_ans_6"
    setup_seeds(args.seed)
    if args.model_config_path == None:
        args.model_config_path = f'model_configs/{args.model_name}_config.json'

    # load target queries and answers
    if args.eval_dataset == 'msmarco':
        corpus, queries, qrels = load_beir_datasets('msmarco', 'train')
        incorrect_answers = load_json(f'results/adv_targeted_results/{args.eval_dataset}-{s}-{v}.json')
        random.shuffle(incorrect_answers)    
    else:
        corpus, queries, qrels = load_beir_datasets(args.eval_dataset, args.split)
        # incorrect_answers = load_json(f'results/adv_targeted_results/{args.eval_dataset}-{s}-{v}.json')
        # incorrect_answers = load_json(f'results/p_adv/{in_data_name}.json')
        # incorrect_answers = load_json(f'dataset/gpt4/{in_data_name}.json')
        incorrect_answers = load_json(f'dataset/new_diff/incorrect_ans_docs/split_by_answer_position/{in_data_name}.json')
        print(f"loading from {in_data_name}")

    incorrect_answers = list(incorrect_answers.values())

    # load BEIR top_k results  
    if args.orig_beir_results is None: 
        print(f"Please evaluate on BEIR first -- {args.eval_model_code} on {args.eval_dataset}")
        # Try to get beir eval results from ./beir_results
        print("Now try to get beir eval results from results/beir_results/...")
        if args.split == 'test':
            args.orig_beir_results = f"results/beir_results/{args.eval_dataset}-{args.eval_model_code}-new.json"
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

    if args.use_truth == 'True':
        args.attack_method = None

    if args.attack_method not in [None, 'None']:
        # Load retrieval models
        model, c_model, tokenizer, get_emb = load_models(args.eval_model_code)
        model.eval()
        model.to(device)
        c_model.eval()
        c_model.to(device) 
        attacker = SerialsAttacker(args,
                            model=model,
                            eval_name=f"{in_data_name}",
                            c_model=c_model,
                            tokenizer=tokenizer,
                            get_emb=get_emb) 
    

    iter_results = []
    for iter in range(args.repeat_times):
        print(f'######################## Iter: {iter+1}/{args.repeat_times} #######################')

        target_queries_idx = range(iter * args.M, iter * args.M + args.M)

        # target_queries = [incorrect_answers[idx]['question'] for idx in target_queries_idx]
        # change to the question list
        with open("../../datasets/serials_questions/nq-all-adv-docs-with-serial-q.json", "r") as f:
            for_questions = json.load(f)
        # target_queries = [incorrect_answers[idx]['questions'][:5] for idx in target_queries_idx]
        ids = [incorrect_answers[idx]['id'] for idx in target_queries_idx]
        target_queries = [for_questions[idx]['serial_questions'][:5] for idx in ids]
        
        print("target_queries:", target_queries)
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
            query_id = incorrect_answers[i]['id']
            questions = for_questions[query_id]['serial_questions'][:5]
            # questions = incorrect_answers[i]['questions']
            print(f'Questions: {questions}\n') 
            question = questions[0]
            print(f'Question: {question}\n') 
            
            gt_ids = list(qrels[incorrect_answers[i]['id']].keys())
            ground_truth = [corpus[id]["text"] for id in gt_ids]
            incco_ans = incorrect_answers[i]['incorrect answer']            

            if args.use_truth == 'True':
                query_prompt = wrap_prompt(question, ground_truth, 4)
                # # response = llm.query(query_prompt)
                # print(f"Output: {response}\n\n")
                # iter_results.append(
                #     {
                #         "question": question,
                #         "input_prompt": query_prompt,
                #         "output": response,
                #     }
                # )  

            else: # topk

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
                        # here 5 should be a params
                        # if (j + 1) % (len(adv_text_list) / args.M) == 0: 
        iter_results.append(
            {
                "id":incorrect_answers[i]['id'],
                "questions": questions,
                "adv_texts": top_k_adv,
                # "input_prompt": query_prompt,
                "incorrect_answer": incco_ans,
                "answer": incorrect_answers[i]['correct answer']
            }
        )
        # with open(f"dataset/gpt4_trans/{in_data_name}-{args.attack_method}.json", "w") as f:

        with open(f"dataset/new_diff/incorrect_ans_docs/attacked_serials/{in_data_name}-{args.attack_method}.json", "w") as f:
            json.dump(iter_results, f, indent=4)
    return iter_results

if __name__ == "__main__":
    main()