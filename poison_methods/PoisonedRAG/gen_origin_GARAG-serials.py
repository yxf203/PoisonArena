import argparse
import os

import json
from tqdm import tqdm
import random
import numpy as np
from src.models import create_model
from beir.datasets.data_loader import GenericDataLoader
from src.utils import load_beir_datasets, load_models
from src.utils import save_results, load_json, setup_seeds, clean_str, f1_score
from src.prompts import wrap_prompt
import torch
import time
import csv
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
            # args.orig_beir_results = f"results/beir_results/serials_contriever_100_new.json"
            args.orig_beir_results = f"results/beir_results/serials.json"

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
        dataset = args.eval_dataset
        out_dir = os.path.join(os.getcwd(), "datasets")
        data_path = os.path.join(out_dir, dataset)
        # corpus, queries, qrels = load_beir_datasets(args.eval_dataset, args.split)
        corpus, queries, qrels = GenericDataLoader(data_path, query_file='serials.jsonl').load(split='serials')

    eval_model_code = "contriever"
    model, c_model, tokenizer, get_emb = load_models(eval_model_code)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    c_model.eval()
    c_model.to(device) 

    # Load adversarial texts
    adv_texts_path = "/data/chenliuji/poison/datasets/serial_questions/nq-all-adv-docs-with-serial-q.json"
    with open(adv_texts_path, 'r') as f:
        adv_texts = json.load(f)
    # Load the mapping from the CSV file
    mapping = {}
    mapping_path = "/data/chenliuji/poison/datasets/serial_questions/test_to_s_mapping.csv"
    with open(mapping_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            original_id = row['original_test_id']
            new_id = row['new_s_id']
            if original_id not in mapping:
                mapping[original_id] = []
            mapping[original_id].append(new_id)

    # Sort the values for each key and keep the top 5 based on the numeric part of the ID
    for key in mapping:
        mapping[key] = sorted(mapping[key], key=lambda x: int(x[1:]))[:5]
    
    trans2GARAG = []
    temp_reflect = {}
    adv_texts = list(adv_texts.values())
    for item in adv_texts:
        id = item["id"]
        temp_reflect[id] = {}
        temp_reflect[id]["questions"] = []
        temp_reflect[id]["texts"] = []

        print(id)
        # if id == 'test0':
        #     continue
        for s_id in mapping[id]:
            temp_reflect[id]["questions"].append(queries[s_id])
            topk_idx = list(results[s_id].keys())[:args.top_k]
            # topk_results = [{'id': idx, "has_answer":True, 'score': results[id][idx], 'context': corpus[idx]["title"] + " " + corpus[idx]["text"] if len(corpus[idx]["title"]) > 0 else corpus[idx]["text"]} for idx in topk_idx]    
            topk_results = [{'id': idx, 'context': corpus[idx]["title"] + " " + corpus[idx]["text"] if len(corpus[idx]["title"]) > 0 else corpus[idx]["text"]} for idx in topk_idx]    
            temp_reflect[id]["texts"] += topk_results
        
        # Compute the average embedding for questions in temp_reflect[id]["questions"]
        question_embeddings = []
        for question in temp_reflect[id]["questions"]:
            tokenized_question = tokenizer(question, padding=True, truncation=True, return_tensors="pt")
            tokenized_question = {key: value.to(device) for key, value in tokenized_question.items()}
            question_emb = get_emb(model, tokenized_question)
            question_embeddings.append(question_emb)
        # Remove duplicates from temp_reflect[id]["texts"] based on 'id'
        unique_texts = {}
        for text in temp_reflect[id]["texts"]:
            if text['id'] not in unique_texts:
                unique_texts[text['id']] = text
        temp_reflect[id]["texts"] = list(unique_texts.values())

        # Calculate the average embedding
        query_emb = torch.mean(torch.stack(question_embeddings), dim=0)
        topk_texts = []
        for texts in temp_reflect[id]["texts"]:
            doc = tokenizer(texts["context"], padding=True, truncation=True, return_tensors="pt")
            
            # Move the document tensor to the same device as the model
            doc = {key: value.to(device) for key, value in doc.items()}

            adv_emb = get_emb(model, doc)
            
            if args.score_function == 'dot':
                # Dot product similarity
                adv_sim = torch.mm(adv_emb, query_emb.T).cpu().item()
            elif args.score_function == 'cos_sim':
                # Cosine similarity
                adv_sim = torch.cosine_similarity(adv_emb, query_emb).cpu().item()
            topk_texts.append({"id": texts["id"], "context": texts["context"], "score": adv_sim})
        # Sort the texts based on the computed similarity score
        topk_texts = sorted(topk_texts, key=lambda x: x['score'], reverse=True)[:args.top_k]
        print("topk_texts", topk_texts)
        # here is the change of the top k results after the attack of PoisonedRAG
        for index, i in enumerate(topk_texts):
            # i['id'] = 'adv' + str(index)
            i["has_answer"] = True
            # topk_results.append(i)
        # print("len!!",len(topk_results))
        temp_reflect[id]["texts"] = topk_texts

        trans = {
            "id": item["id"],
            "question": temp_reflect[id]["questions"][0],
            "questions": temp_reflect[id]["questions"],
            "answers":[item["correct answer"]],
            "ctxs": temp_reflect[id]["texts"]
        }
        trans2GARAG.append(trans)
        # temp_db.append(temp)
        # print(trans2GARAG)
    # with open("dataset/origin/nq-GARAG-500-1000-1.json", "w") as f:
    with open("dataset/serials/serials4GARAG.json", "w") as f:
        json.dump(trans2GARAG, f, indent=4)
    
if __name__ == '__main__':
    main()