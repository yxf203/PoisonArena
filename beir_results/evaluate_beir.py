import pathlib, os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import logging
import json
import torch
import sys
import transformers

from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval.models import DPR


from contriever_src.contriever import Contriever
from contriever_src.beir_utils import DenseEncoderModel

import argparse
parser = argparse.ArgumentParser(description='test')

parser.add_argument('--model_code', type=str, default="contriever")
parser.add_argument('--score_function', type=str, default='dot', choices=['dot', 'cos_sim'])
parser.add_argument('--top_k', type=int, default=100)
parser.add_argument('--dataset', type=str, default="msmarco", help='BEIR dataset to evaluate')
parser.add_argument('--split', type=str, default='test')

parser.add_argument('--result_output', default="./ms-contriever-dot.json", type=str)
# parser.add_argument('--result_output', default="/data1/chenliuji/combat/PoisonedRAG/results/TrojanRAG_results/TrojanRAG_100.json", type=str)

parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument("--per_gpu_batch_size", default=64, type=int, help="Batch size per GPU/CPU for indexing.")
parser.add_argument('--max_length', type=int, default=128)

args = parser.parse_args()


model_code_to_qmodel_name = {
    "contriever": "facebook/contriever",
    "contriever-msmarco": "facebook/contriever-msmarco",
    "ance": "sentence-transformers/msmarco-roberta-base-ance-firstp"
}

model_code_to_cmodel_name = {
    "contriever": "facebook/contriever",
    "contriever-msmarco": "facebook/contriever-msmarco",
    "ance": "sentence-transformers/msmarco-roberta-base-ance-firstp"
}

def compress(results):
    for y in results:
        k_old = len(results[y])
        break
    sub_results = {}
    for query_id in results:
        sims = list(results[query_id].items())
        sims.sort(key=lambda x: x[1], reverse=True)
        sub_results[query_id] = {}
        for c_id, s in sims[:2000]:
            sub_results[query_id][c_id] = s
    for y in sub_results:
        k_new = len(sub_results[y])
        break
    logging.info(f"Compressed retrieval results from top-{k_old} to top-{k_new}.")
    return sub_results

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

logging.info(args)


# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#### Download and load dataset
dataset = args.dataset
# url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
parent_dir = os.path.dirname(os.getcwd())
out_dir = os.path.join(parent_dir, "datasets")
data_path = os.path.join(out_dir, dataset)
# if not os.path.exists(data_path):
#     data_path = util.download_and_unzip(url, out_dir)
logging.info(data_path)


corpus, queries, qrels = GenericDataLoader(data_path, query_file='ms_queries.jsonl').load(split='ms-qrels')



# grp: If you want to use other datasets, you could prepare your dataset as the format of beir, then load it here.

logging.info("Loading model...")
if 'contriever' in args.model_code:
    parent_dir = os.path.dirname(os.getcwd())
    r_path = os.path.join(parent_dir, "retriever", model_code_to_qmodel_name[args.model_code])
    encoder = Contriever.from_pretrained(r_path).cuda()
    tokenizer = transformers.BertTokenizerFast.from_pretrained(r_path)
    model = DRES(DenseEncoderModel(encoder, doc_encoder=encoder, tokenizer=tokenizer), batch_size=args.per_gpu_batch_size)
elif 'dpr' in args.model_code:
    model = DRES(DPR((model_code_to_qmodel_name[args.model_code], model_code_to_cmodel_name[args.model_code])), batch_size=args.per_gpu_batch_size, corpus_chunk_size=5000)
elif 'ance' in args.model_code:
    model = DRES(models.SentenceBERT(model_code_to_cmodel_name[args.model_code]), batch_size=args.per_gpu_batch_size)
else:
    raise NotImplementedError

logging.info(f"model: {model.model}")
print("score_function:", args.score_function)
retriever = EvaluateRetrieval(model, score_function=args.score_function, k_values=[args.top_k]) # "cos_sim"  or "dot" for dot-product
results = retriever.retrieve(corpus, queries)
                                            
logging.info("Printing results to %s"%(args.result_output))
sub_results = compress(results)

with open(args.result_output, 'w') as f:
    json.dump(sub_results, f)
