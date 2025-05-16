import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import json
import random
from tqdm import tqdm
from collections import defaultdict
from datasets import Dataset
import pandas as pd
import time
from transformers import (
    AutoTokenizer,
    default_data_collator,
    set_seed,
)
import torch.nn.functional as F

from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

import argparse
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from collections import Counter

from utils import load_models

def find_same_id_item(items, target_id):
    # get the similar texts to the question with target_id
    for item in items:
        if item.get('id') == target_id:
            return item
    return None

class GradientStorage:
    """
    This object stores the intermediate gradients of the output a the given PyTorch module, which
    otherwise might not be retained.
    """
    def __init__(self, module):
        self._stored_gradient = None
        module.register_full_backward_hook(self.hook)

    def hook(self, module, grad_in, grad_out):
        self._stored_gradient = grad_out[0]

    def get(self):
        return self._stored_gradient

def get_embeddings(model):
    """Returns the wordpiece embedding module."""
    # base_model = getattr(model, config.model_type)
    # embeddings = base_model.embeddings.word_embeddings

    # This can be different for different models; the following is tested for Contriever
    if isinstance(model, DPRContextEncoder):
        embeddings = model.ctx_encoder.bert_model.embeddings.word_embeddings
    elif isinstance(model, SentenceTransformer):
        embeddings = model[0].auto_model.embeddings.word_embeddings
    else:
        embeddings = model.embeddings.word_embeddings
    return embeddings


def hotflip_attack(averaged_grad,
                   embedding_matrix,
                   increase_loss=False,
                   num_candidates=1,
                   filter=None,
                   score_function='dot'):
    """
    Performs a hotflip attack to find the top candidate replacements for a given gradient.
    Args:
        averaged_grad (torch.Tensor): Gradient tensor of shape [hidden_dim].
        embedding_matrix (torch.Tensor): Word embedding matrix of shape [vocab_size, hidden_dim].
        increase_loss (bool, optional): Whether to increase the loss (default is to decrease the loss). Defaults to False.
        num_candidates (int, optional): Number of candidate words to return. Defaults to 1.
        filter (torch.Tensor, optional): A mask to filter out specific words (e.g., [MASK], stopwords). Defaults to None.
        score_function (str, optional): Scoring function to use, either 'dot' (dot product) or 'cos_sim' (cosine similarity). Defaults to 'dot'.
    Returns:
        torch.Tensor: The IDs of the top-k candidate words.
    """
    with torch.no_grad():
        if score_function == 'dot':
            similarity = torch.matmul(embedding_matrix, averaged_grad)  # [vocab_size]
        elif score_function == 'cos_sim':
            grad_norm = torch.nn.functional.normalize(averaged_grad.unsqueeze(0), p=2, dim=1)  # [1, hidden_dim]
            emb_norm = torch.nn.functional.normalize(embedding_matrix, p=2, dim=1)  # [vocab_size, hidden_dim]
            similarity = torch.matmul(emb_norm, grad_norm.T).squeeze()  # [vocab_size]
        else:
            raise ValueError(f"Unsupported score_function: {score_function}")

        if filter is not None:
            similarity -= filter

        if not increase_loss:
            similarity *= -1
        _, top_k_ids = similarity.topk(num_candidates)
    return top_k_ids

def evaluate_acc(args, model, c_model, get_emb, dataloader, adv_passage_ids, adv_passage_attention, adv_passage_token_type, data_collator, device='cuda'):
    """Returns the 2-way classification accuracy (used during training)"""
    model.eval()
    c_model.eval()
    acc = 0
    tot = 0
    for idx, (data) in tqdm(enumerate(dataloader)):
        data = data_collator(data) # [bsz, 3, max_len]

        # Get query embeddings
        q_sent = {k: data[k][:, 0, :].to(device) for k in data.keys()}
        q_emb = get_emb(model, q_sent)  # [b x d]

        gold_pass = {k: data[k][:, 1, :].to(device) for k in data.keys()}
        gold_emb = get_emb(c_model, gold_pass) # [b x d]
        if args.score_function == 'dot':
            sim_to_gold = torch.bmm(q_emb.unsqueeze(dim=1), gold_emb.unsqueeze(dim=2)).squeeze()
        elif args.score_function == 'cos_sim':
            sim_to_gold = torch.cosine_similarity(q_emb, gold_emb, dim=-1)
        # sim_to_gold = torch.bmm(q_emb.unsqueeze(dim=1), gold_emb.unsqueeze(dim=2)).squeeze()

        p_sent = {'input_ids': adv_passage_ids, 
                  'attention_mask': adv_passage_attention, 
                  'token_type_ids': adv_passage_token_type}
        p_emb = get_emb(c_model, p_sent)  # [k x d]

        # sim = torch.mm(q_emb, p_emb.T).squeeze()  # [b x k]
        if args.score_function == 'dot':
            sim = torch.mm(q_emb, p_emb.T).squeeze()  # [b x k]
        elif args.score_function == 'cos_sim':
            sim = torch.cosine_similarity(q_emb.unsqueeze(1), p_emb.unsqueeze(0), dim=-1).squeeze()

        acc += (sim_to_gold > sim).sum().cpu().item()
        tot += q_emb.shape[0]
    
    print(f'Acc = {acc / tot * 100} ({acc} / {tot})')
    return acc / tot

def kmeans_split(data_dict, model, get_emb, tokenizer, k, split):
    """Get all query embeddings and perform kmeans"""
    
    # get query embs
    q_embs = []
    for q in tqdm(data_dict["sent0"]):
        query_input = tokenizer(q, padding=True, truncation=True, return_tensors="pt")
        query_input = {key: value.cuda() for key, value in query_input.items()}
        with torch.no_grad():
            query_emb = get_emb(model, query_input)
        q_embs.append(query_emb[0].cpu().numpy())
    q_embs = np.array(q_embs)
    print("q_embs", q_embs.shape)

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=k, random_state=0).fit(q_embs)
    print(Counter(kmeans.labels_))

    ret_dict = {"sent0": [], "sent1": []}
    for i in range(len(data_dict["sent0"])):
        if kmeans.labels_[i] == split:
            ret_dict["sent0"].append(data_dict["sent0"][i])
            ret_dict["sent1"].append(data_dict["sent1"][i])
    print("K = %d, split = %d, tot num = %d"%(k, split, len(ret_dict["sent0"])))

   

    return ret_dict


def main():
    parser = argparse.ArgumentParser(description='test')
    # parser.add_argument('--dataset', type=str, default="nq", help='BEIR dataset to evaluate')
    parser.add_argument('--dataset', type=str, default="msmarco", help='BEIR dataset to evaluate')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--model_code', type=str, default='contriever')
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--pad_to_max_length', default=True)
    parser.add_argument('--score_function', type=str, default='dot', choices=['dot', 'cos_sim'])
    parser.add_argument("--num_adv_passage_tokens", default=50, type=int)
    parser.add_argument("--num_cand", default=100, type=int)
    parser.add_argument("--per_gpu_eval_batch_size", default=64, type=int, help="Batch size per GPU/CPU for indexing.")
    parser.add_argument("--num_iter", default=5000, type=int)
    parser.add_argument("--num_grad_iter", default=1, type=int)

    parser.add_argument("--output_file", default="/output/nq-contriever", type=str)

    parser.add_argument("--k", default=1, type=int)
    parser.add_argument("--kmeans_split", default=0, type=int)
    parser.add_argument("--do_kmeans", default=False, action="store_true")

    parser.add_argument("--dont_init_gold", action="store_true", help="if ture, do not init with gold passages")
    args = parser.parse_args()

    print(args)

    device = 'cuda'

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    set_seed(0)
    
    # Load models
    model, c_model, tokenizer, get_emb = load_models(args.model_code)
        
    model.eval()
    model.to(device)
    c_model.eval()
    c_model.to(device)

    # Load datasets
    # url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(args.dataset)
    out_dir = os.path.join(os.getcwd(), "datasets")
    data_path = os.path.join(out_dir, args.dataset)

    with open("./dataset/diff-curpus/new-diff-corpus/ms/ms-inco_ans_6.json", "r") as f:
        adv_texts_all = json.load(f)

   
    corpus, queries, qrels = GenericDataLoader(data_path, query_file='ms_queries.jsonl').load(split='ms_dot')

    l = list(qrels.items())
    qrels = dict(l)
    
    grouped_queries = defaultdict(list)
    # group_size = 30
    group_size = 1
    current_group = 0

    for i, (query_id, rels) in enumerate(qrels.items()):
        if i % group_size == 0 and i != 0:
            current_group += 1
        grouped_queries[current_group].append(query_id)

    results = []
    iter_times = []
    count = 0
    for group_id, q_ids in grouped_queries.items():
        start_time = time.time()
        data_dict = {"sent0": [], "sent1": []}
        
        print(f"Processing group {group_id} (query_ids: {q_ids[0]} to {q_ids[-1]})")
        print(f"Number of queries in this group: {len(q_ids)}")
        
        for query_id in q_ids:
            q_ctx = queries[query_id]
            
            for corpus_id in qrels[query_id]:
                c_ctx = corpus[corpus_id].get("title", "") + ' ' + corpus[corpus_id].get("text", "")
                data_dict["sent0"].append(q_ctx)
                data_dict["sent1"].append(c_ctx)
        
        print(f"Number of pairs in this group: {len(data_dict['sent0'])}")

        result = {}
        result["questions"] = data_dict["sent0"]
        result["id"] = q_ids[0]
        result["adv_texts"] = []
    
        datasets = {"train": Dataset.from_dict(data_dict)}
        # print("see see datasets", datasets)
        def tokenization(examples):
            q_feat = tokenizer(examples["sent0"], max_length=args.max_seq_length, truncation=True, padding="max_length" if args.pad_to_max_length else False)
            c_feat = tokenizer(examples["sent1"], max_length=args.max_seq_length, truncation=True, padding="max_length" if args.pad_to_max_length else False)

            ret = {}
            for key in q_feat:
                ret[key] = [(q_feat[key][i], c_feat[key][i]) for i in range(len(examples["sent0"]))]

            return ret

        # use 30% examples as dev set during training
        print('Train data size = %d'%(len(datasets["train"])))
        # num_valid = min(1000, int(len(datasets["train"]) * 0.3))
        num_valid = 1
        datasets["subset_train"] = Dataset.from_dict(datasets["train"][:])
        datasets["subset_valid"] = Dataset.from_dict(datasets["train"][:])

        train_dataset = datasets["subset_train"].map(tokenization, batched=True, remove_columns=datasets["train"].column_names)
        dataset = datasets["subset_valid"].map(tokenization, batched=True, remove_columns=datasets["train"].column_names)
        print('Finished loading datasets')

        data_collator = default_data_collator
        dataloader = DataLoader(train_dataset, batch_size=args.per_gpu_eval_batch_size, shuffle=True, collate_fn=lambda x: x )
        valid_dataloader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=lambda x: x )

        # Set up variables for embedding gradients
        embeddings = get_embeddings(c_model)
        print('Model embedding', embeddings)
        
        # item = find_same_id_item(adv_texts_all, q_ids[0])
        item = adv_texts_all[q_ids[0]]
        # adv_texts = [i["context"] for i in item["adv_texts"]]
        adv_texts = item["adv_texts"]
        for adv_text in adv_texts:
            print("adv_text:", adv_text)
            embedding_gradient = GradientStorage(embeddings)

            # adv_passage_ids = tokenizer.encode(adv_text, add_special_tokens=False, max_length=args.num_adv_passage_tokens, truncation=True)
            adv_passage_ids = tokenizer(adv_text, max_length=args.num_adv_passage_tokens, truncation=True, padding=False)['input_ids']
            adv_len = len(adv_passage_ids)

            print('Init adv_passage', tokenizer.convert_ids_to_tokens(adv_passage_ids))
            adv_passage_ids = torch.tensor(adv_passage_ids, device=device).unsqueeze(0)

            adv_passage_attention = torch.ones_like(adv_passage_ids, device=device)
            adv_passage_token_type = torch.zeros_like(adv_passage_ids, device=device)

            best_adv_passage_ids = adv_passage_ids.clone()
            best_acc = evaluate_acc(args, model, c_model, get_emb, valid_dataloader, best_adv_passage_ids, adv_passage_attention, adv_passage_token_type, data_collator)
            print(best_acc)
            temp_result = {}
            temp_result["questions"] = data_dict["sent0"]
            temp_result["id"] = q_ids[0]
            temp_result["adv_texts"] = []
            t = {}
            for it_ in range(args.num_iter):
                print(f"Iteration: {it_}")
                
                print(f'Accumulating Gradient {args.num_grad_iter}')
                c_model.zero_grad()

                pbar = range(args.num_grad_iter)
                train_iter = iter(dataloader)
                grad = None

                for _ in pbar:
                    try:
                        data = next(train_iter)
                        data = data_collator(data) # [bsz, 3, max_len]
                    except:
                        print('Insufficient data!')
                        break
                
                    q_sent = {k: data[k][:, 0, :].to(device) for k in data.keys()}
                    q_emb = get_emb(model, q_sent).detach()

                    gold_pass = {k: data[k][:, 1, :].to(device) for k in data.keys()}
                    gold_emb = get_emb(c_model, gold_pass).detach()

                    # sim_to_gold = torch.bmm(q_emb.unsqueeze(dim=1), gold_emb.unsqueeze(dim=2)).squeeze()
                    if args.score_function == 'dot':
                        sim_to_gold = torch.bmm(q_emb.unsqueeze(dim=1), gold_emb.unsqueeze(dim=2)).squeeze()
                    elif args.score_function == 'cos_sim':
                        sim_to_gold = torch.cosine_similarity(q_emb, gold_emb, dim=-1)

                    sim_to_gold_mean = sim_to_gold.mean().cpu().item()
                    print('Avg sim to gold p =', sim_to_gold_mean)


                    p_sent = {'input_ids': adv_passage_ids, 
                            'attention_mask': adv_passage_attention, 
                            'token_type_ids': adv_passage_token_type}
                    p_emb = get_emb(c_model, p_sent)

                    # Compute loss
                    # sim = torch.mm(q_emb, p_emb.T)  # [b x k]
                    if args.score_function == 'dot':
                        sim = torch.mm(q_emb, p_emb.T)
                    elif args.score_function == 'cos_sim':
                        sim = torch.cosine_similarity(q_emb, p_emb)
                    print(it_, _, 'Avg sim to adv =', sim.mean().cpu().item(), 'sim to gold =', sim_to_gold_mean)

                    suc_att = ((sim - sim_to_gold.unsqueeze(-1)) >= 0).sum().cpu().item()

                    print('Attack on train: %d / %d'%(suc_att, sim_to_gold), 'best_acc', best_acc)
                    loss = sim.mean()
                    print('loss', loss.cpu().item())
                    loss.backward()

                    temp_grad = embedding_gradient.get()
                    if grad is None:
                        grad = temp_grad.sum(dim=0) / args.num_grad_iter
                    else:
                        grad += temp_grad.sum(dim=0) / args.num_grad_iter
                    
                print('Evaluating Candidates')
                pbar = range(args.num_grad_iter)
                train_iter = iter(dataloader)

                # token_to_flip = random.randrange(args.num_adv_passage_tokens)
                print("len len len:", adv_len)
                token_to_flip = random.randrange(adv_len)
                candidates = hotflip_attack(grad[token_to_flip],
                                            embeddings.weight,
                                            increase_loss=True,
                                            num_candidates=args.num_cand,
                                            filter=None,
                                            score_function=args.score_function)
                
                current_score = 0
                candidate_scores = torch.zeros(args.num_cand, device=device)
                current_acc_rate = 0
                candidate_acc_rates = torch.zeros(args.num_cand, device=device)

                for step in pbar:
                    try:
                        data = next(train_iter)
                        data = data_collator(data) # [bsz, 3, max_len]
                    except:
                        print('Insufficient data!')
                        break
                        
                    q_sent = {k: data[k][:, 0, :].to(device) for k in data.keys()}
                    q_emb = get_emb(model, q_sent).detach()

                    gold_pass = {k: data[k][:, 1, :].to(device) for k in data.keys()}
                    gold_emb = get_emb(c_model, gold_pass).detach()

                    if args.score_function == 'dot':
                        sim_to_gold = torch.bmm(q_emb.unsqueeze(dim=1), gold_emb.unsqueeze(dim=2)).squeeze()
                    elif args.score_function == 'cos_sim':
                        sim_to_gold = torch.cosine_similarity(q_emb, gold_emb, dim=-1)
                    # sim_to_gold = torch.bmm(q_emb.unsqueeze(dim=1), gold_emb.unsqueeze(dim=2)).squeeze()
                    sim_to_gold_mean = sim_to_gold.mean().cpu().item()
                    print('Avg sim to gold p =', sim_to_gold_mean)

                    p_sent = {'input_ids': adv_passage_ids, 
                            'attention_mask': adv_passage_attention, 
                            'token_type_ids': adv_passage_token_type}
                    p_emb = get_emb(c_model, p_sent)

                    # Compute loss
                    # sim = torch.mm(q_emb, p_emb.T)  # [b x k]
                    if args.score_function == 'dot':
                        sim = torch.mm(q_emb, p_emb.T)
                    elif args.score_function == 'cos_sim':
                        sim = torch.cosine_similarity(q_emb, p_emb)
                    print(it_, _, 'Avg sim to adv =', sim.mean().cpu().item(), 'sim to gold =', sim_to_gold_mean)
                    suc_att = ((sim - sim_to_gold.unsqueeze(-1)) >= 0).sum().cpu().item()
                    # print('Attack on train: %d / %d'%(suc_att, sim_to_gold.shape[0]), 'best_acc', best_acc)
                    print('Attack on train: %d / %d'%(suc_att, sim_to_gold), 'best_acc', best_acc)
                    loss = sim.mean()
                    temp_score = loss.sum().cpu().item()

                    current_score += temp_score
                    current_acc_rate += suc_att

                    for i, candidate in enumerate(candidates):
                        temp_adv_passage = adv_passage_ids.clone()
                        temp_adv_passage[:, token_to_flip] = candidate
                        p_sent = {'input_ids': temp_adv_passage, 
                            'attention_mask': adv_passage_attention, 
                            'token_type_ids': adv_passage_token_type}
                        p_emb = get_emb(c_model, p_sent)
                        with torch.no_grad():
                            # sim = torch.mm(q_emb, p_emb.T)
                            if args.score_function == 'dot':
                                sim = torch.mm(q_emb, p_emb.T)
                            elif args.score_function == 'cos_sim':
                                sim = torch.cosine_similarity(q_emb, p_emb)
                        
                            can_suc_att = ((sim - sim_to_gold.unsqueeze(-1)) >= 0).sum().cpu().item()
                            can_loss = sim.mean()
                            temp_score = can_loss.sum().cpu().item()

                            candidate_scores[i] += temp_score
                            candidate_acc_rates[i] += can_suc_att
                print(current_score, max(candidate_scores).cpu().item())
                print(current_acc_rate, max(candidate_acc_rates).cpu().item())

                # if find a better one, update
                if (candidate_scores > current_score).any() or (candidate_acc_rates > current_acc_rate).any():
                    logger.info('Better adv_passage detected.')
                    best_candidate_score = candidate_scores.max()
                    best_candidate_idx = candidate_scores.argmax()
                    adv_passage_ids[:, token_to_flip] = candidates[best_candidate_idx]
                    print('Current adv_passage', tokenizer.convert_ids_to_tokens(adv_passage_ids[0]))
                else:
                    print('No improvement detected!')
                    continue

                cur_acc = evaluate_acc(args, model, c_model, get_emb, valid_dataloader, adv_passage_ids, adv_passage_attention, adv_passage_token_type, data_collator)
                if cur_acc < best_acc or (cur_acc == best_acc and ((candidate_scores > current_score).any() or (candidate_acc_rates > current_acc_rate).any())):
                # if cur_acc < best_acc:
                    best_acc = cur_acc
                    best_adv_passage_ids = adv_passage_ids.clone()
                    logger.info('!!! Updated best adv_passage')
                    print(tokenizer.convert_ids_to_tokens(best_adv_passage_ids[0]))
                    print("output_file:", args.output_file)
                    if args.output_file is not None:
                        dummy = tokenizer.convert_ids_to_tokens(best_adv_passage_ids[0])
                        # dummy_text = " ".join(dummy)
                        # dummy_text = tokenizer.convert_tokens_to_string(dummy)
                        dummy_text = tokenizer.decode(best_adv_passage_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                        temp_result["adv_texts"].append(
                            {"it": it_, "best_acc": best_acc, "dummy": dummy, "dummy_text": dummy_text, "tot": num_valid}
                        )
                        t = {"it": it_, "best_acc": best_acc, "dummy_text": dummy_text, "tot": num_valid}
                        with open(f"{args.output_file}/mid.json", 'a') as f:
                            json.dump(temp_result, f)    
                print('best_acc', best_acc)
            result["adv_texts"].append(t)
            with open(f"{args.output_file}/mid-f.json", 'a') as f:
                json.dump(result, f)    
        results.append(result)
        with open(f"{args.output_file}/all.json", 'w') as f:
            json.dump(results, f)   
        end_time = time.time()
        iter_times.append(end_time - start_time)
        print(f"time: {end_time - start_time:.2f}s") 
        avg_time = sum(iter_times) / len(iter_times)
    attack_time_data = {
        "average_time": avg_time,
        "iter_times": iter_times
    }
    output_path = f"{args.output_file}/attack_time.json"
    with open(output_path, "w") as f:
        json.dump(attack_time_data, f, indent=4)
    print(f"Attack times and average time written to {output_path}")


if __name__ == "__main__":
    main()

