# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
import glob
import json
import os
import pickle
import time
import torch
import lightning.pytorch as pl

from transformers import BertModel, XLMRobertaModel, AlbertModel, T5EncoderModel, DPRContextEncoder, DPRQuestionEncoder
from transformers import AutoTokenizer

from src import util
from src import task

import logging
logger = logging.getLogger(__name__)

save_keys = [
    "is_og", "question", "doc_id", "att_id", "question"
]

hf_model_name ={
    "contriever": "facebook/contriever", 
    "dpr": ("dpr-ctx_encoder-multiset-base", "dpr-question_encoder-multiset-base")
}


def load_retriever(opt):
    return Retriever(opt)

def _load_retriever(opt):
    retriever_name = opt.retriever
    if "contriever" in retriever_name:
        t_model_dir = "../retriever"
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(t_model_dir, hf_model_name[retriever_name]))
        d_encoder = Contriever.from_pretrained(os.path.join(t_model_dir, hf_model_name[retriever_name])).to("cuda")
        q_encoder = d_encoder
    elif "dpr" in retriever_name:
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(t_model_dir, hf_model_name[retriever_name][0]))
        d_encoder = DPRC.from_pretrained(os.path.join(t_model_dir, hf_model_name[retriever_name][0])).to("cuda")
        q_encoder = DPRQ.from_pretrained(os.path.join(t_model_dir, hf_model_name[retriever_name][1])).to("cuda")
    else:
        raise NotImplementedError("Not supported retriever class")
    d_encoder.eval()
    q_encoder.eval()
    return tokenizer, d_encoder, q_encoder

class Encoder(BertModel):
    def __init__(self, config, pooling="average", **kwargs):
        super().__init__(config, add_pooling_layer=False)
        if not hasattr(config, "pooling"):
            self.config.pooling = pooling

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        normalize=False,
        labels=None,
        return_dict=None,
    ):

        model_output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden = model_output["last_hidden_state"]
        last_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.0)

        if self.config.pooling == "average":
            emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.config.pooling == "cls":
            emb = last_hidden[:, 0]

        if normalize:
            emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb

class DPRQ(DPRQuestionEncoder):
    def forward(self, **kwargs):
        output = super().forward(**kwargs)
        return output.pooler_output
    
class DPRC(DPRContextEncoder):
    def forward(self, **kwargs):
        output = super().forward(**kwargs)
        return output.pooler_output

class Contriever(BertModel):
    def __init__(self, config, pooling="average", **kwargs):
        super().__init__(config, add_pooling_layer=False)
        if not hasattr(config, "pooling"):
            self.config.pooling = pooling

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        normalize=False,
        labels=None,
        return_dict=None,
    ):

        model_output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden = model_output["last_hidden_state"]
        last_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.0)

        if self.config.pooling == "average":
            emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.config.pooling == "cls":
            emb = last_hidden[:, 0]

        if normalize:
            emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb

class Evaluator(object):
    def __init__(self, opt):
        self.opt = opt

        model_cls = _load_retriever(opt.retriever)
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(opt.model_dir, opt.retriever))
        self.q_encoder = model_cls.from_pretrained(os.path.join(opt.model_dir, opt.retriever)).cuda()
        self.p_encoder = copy.deepcopy(self.q_encoder).cuda()
        self.task_ids = None

        self.index = index.Indexer(opt.projection_size, opt.n_subquantizers, opt.n_bits)
    
    def encode_docs(self):
        # index all passages
        self.passages = task.load_passages(self.opt.passages)
        self.passage_id_map = {x["id"]: x for x in self.passages}

        if not os.path.exists(os.path.join(self.opt.passage_embeddings, "passages_00")):
            allids, allembeddings = index.embed_passages(self.opt, self.passages, self.p_encoder, self.tokenizer)
            with open(os.path.join(self.opt.passage_embeddings, "passages_00"), mode="wb") as f:
                pickle.dump((allids, allembeddings), f)

        input_paths = glob.glob(self.opt.passage_embeddings + "*")
        input_paths = sorted(input_paths)
        index_path = os.path.join(self.opt.index_dir, "index.faiss")

        if self.opt.save_or_load_index and os.path.exists(index_path):
            self.index.deserialize_from(self.opt.index_dir)
        else:
            logging.info(f"Indexing passages from files {input_paths}")
            self.index.index_encoded_data(input_paths, self.opt.indexing_batch_size)
            if self.opt.save_or_load_index:
                self.index.serialize(self.opt.index_dir)
            
    def search(self):
        data = self._load_queries(self.opt.data_path)
        output_path = os.path.join(self.opt.output_dir, "result.jsonl")

        queries = [ex["question"] for ex in data]

        questions_embedding = index.embed_queries(self.opt, queries, self.q_encoder, self.tokenizer, self.task_ids)

        # get top k results
        start_time_retrieval = time.time()
        top_ids_and_scores = self.index.search_knn(questions_embedding, self.opt.n_docs)
        logger.info(f"Search time: {time.time()-start_time_retrieval:.1f} s.")

        self._add_passages(data, self.passage_id_map, top_ids_and_scores)
        hasanswer = self._validate(data, self.opt.validation_workers)
        self._add_hasanswer(data, hasanswer)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as fout:
            for ex in data:
                json.dump(ex, fout, ensure_ascii=False)
                fout.write("\n")
        logger.info(f"Saved results to {output_path}")

    def _validate(self, data, workers_num):
        match_stats = util.calculate_matches(data, workers_num)
        top_k_hits = match_stats.top_k_hits

        logger.info("Validation results: top k documents hits %s", top_k_hits)
        top_k_hits = [v / len(data) for v in top_k_hits]
        message = ""
        for k in [5, 10, 20, 100]:
            if k <= len(top_k_hits):
                message += f"R@{k}: {top_k_hits[k-1]} "
        logger.info(message)
        return match_stats.questions_doc_hits

    def _load_queries(self, data_path):
        if data_path.endswith(".json"):
            with open(data_path, "r") as fin:
                data = json.load(fin)
        elif data_path.endswith(".jsonl"):
            data = []
            with open(data_path, "r") as fin:
                for k, example in enumerate(fin):
                    example = json.loads(example)
                    data.append(example)
        return data

    def _add_passages(self, data, passages, top_passages_and_scores):
        # add passages to original data
        merged_data = []
        assert len(data) == len(top_passages_and_scores)
        for i, d in enumerate(data):
            results_and_scores = top_passages_and_scores[i]
            docs = [passages[doc_id] for doc_id in results_and_scores[0]]
            scores = [str(score) for score in results_and_scores[1]]
            ctxs_num = len(docs)
            d["ctxs"] = [
                {
                    "id": results_and_scores[0][c],
                    "title": docs[c]["title"] if "title" in docs[c] else "",
                    "text": docs[c]["text"],
                    "score": scores[c],
                }
                for c in range(ctxs_num)
            ]

    def _add_hasanswer(self, data, hasanswer):
        # add hasanswer to data
        for i, ex in enumerate(data):
            for k, d in enumerate(ex["ctxs"]):
                d["hasanswer"] = hasanswer[i][k]

class Retriever(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.tokenizer, self.d_encoder, self.q_encoder = _load_retriever(opt)
    
    def forward(self, queries, contexts):
        queries.to(self.q_encoder.device)
        contexts.to(self.q_encoder.device)
        query_embeddings = self.q_encoder(**queries)
        context_embeddings = self.d_encoder(**contexts)
        scores = [q@c for q,c in zip(query_embeddings, context_embeddings)]
        return scores
    
    def get_tokenizer(self):
        return self.tokenizer

class Retrieve_Module(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.model = Retriever(opt)
        logger.info("Model Load Done")

    def forward(self, query, context):
        scores = self.model(query, context)
        return scores

    def predict_step(self, batch, batch_idx):
        context = batch["context_embeddings"]
        og_context = batch["og_context_embeddings"]
        query = batch["query_embeddings"]

        og_scores = self(query, og_context)
        scores = self(query, context)
        result = self._process_output(batch, og_scores, scores)
        return result
    
    def _process_output(self, batch, og_scores, scores):
        keys = list(batch.keys())
        result = []
        for i in range(len(scores)):
            instance = {}
            for key in keys:
                if not isinstance(batch[key][i],torch.Tensor) and key in save_keys:
                    instance[key] = batch[key][i]
            instance["scores"] = {
                "og": float(og_scores[i]),
                "att": float(scores[i])
            }
            result.append(instance)
        return result
    