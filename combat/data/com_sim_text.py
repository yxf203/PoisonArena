import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import json
import torch
from utils.tools import compute_sim
from PoisonedRAG.src.utils import load_beir_datasets, load_models
eval_model_code = "contriever"
model, c_model, tokenizer, get_emb = load_models(eval_model_code)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.eval()
model.to(device)
c_model.eval()
c_model.to(device) 

# name = '100items/P5-nq-0-100_attack_serials.json'
# name = '100items/poisoned_results_TrojanRAG.json'
# name = 'AdvDec/AdvDec-100-single-a2.json'
# name = 'gaslite/gaslite-test.json'
# name = 'corpus/corpus-a2-format.json'
# name = 'cpoison/new-diff-corpus/cpoison5-inco-ans3.json'
# name = 'nq/phi4/AdvDec/AdvDec5-dcorpus-dot-ans6.json'
# name = 'nq/llama3b/AdvDec/AdvDec5-dcorpus-dot-ans5-tmp.json'
# name = 'nq/phi4/G/GARAG5-ans.json'
# name = 'nq/vicuna7b/G/GARAG5-ans.json'
# name = 'nq/vicuna7b/AdvDec/AdvDec5-dcorpus-dot-ans6.json'
# name = '/nq/vicuna7b/cpoison/cpoison5-inco-ans3.json'
# name = 'nq/phi4/cpoison/cpoison5-inco-ans2.json'
# name = 'nq/llama3.2_3b/cpoison/cpoison5-inco-ans3.json'
# name = 'nq/llama3b/G/GARAG5-ans-740.json'
# name = 'ms/AdvDec/AdvDec5-ans6.json'
# name = 'ms/corpus/corpus-dcorpus-answer-ans6.json'
# name = 'ms/gaslite/gaslite-dcorpus-ans6.json'
# name = 'msmarco/llama8b/cpoison/cpoison5-inco-ans3.json'
name = 'msmarco/llama8b/G/GARAG5-ans.json'
# name = 'cpoison5-20.json'
with open(f"data/original/{name}", "r") as f:
# with open(f"data/serials_origin/{name}", "r") as f:
    texts = json.load(f)
texts.sort(key=lambda x: int(x["id"].replace("test", "")))
for text in texts:
    question = text["question"]
    adv_texts = text["adv_texts"]
    for adv_text in adv_texts:
        context = adv_text["context"]
        score = compute_sim(model, tokenizer, get_emb, question, context, score_function="dot", device=device)
        adv_text["score"] = score
with open(f"data/combat_data/{name}", 'w') as f:
# with open(f"data/serials_combat/{name}", 'w') as f:
    json.dump(texts, f, indent=4)
    print("done!")
    