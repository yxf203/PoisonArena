import csv
import io
import os
import json
import logging
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset,  SequentialSampler
from src.util import EM, f1, acc, _load_wiki, norm_acc
from textattack import datasets

logger = logging.getLogger(__name__)

def evaluate(data):
    result = [0,0,0]
    e2e = [0,0]
    count = 0
    for d in data:
        print(d["att"][-1][0][0])
        if d["att"][-1][0][1][0] < 1:
            result[0] += 1
        if d["att"][-1][0][1][1] < 1:
            result[1] += 1
        if d["att"][-1][0][1][0] < 1 and d["att"][-1][0][1][1] < 1:
            result[2] += 1
            e2e[0] += EM(d["answers"], d["att_pred"][0])
            # if EM(d["answers"], d["att_pred"][0]) == 0:
            #     from IPython import embed; embed()
            e2e[1] += norm_acc(d["answers"], d["att_pred"][0]) 
            count += 1

        else:
            e2e[0] += 1
            e2e[1] += 1


    result = [round(r/len(data)*100,1) for r in result] 
    e2e = [round(e / len(data) * 100, 1) for e in e2e]
    logger.info("ASR_R: {}, ASR_L: {}, ASR_T: {}, EM: {}, Acc: {}".format(result[0], result[1], result[2], e2e[0], e2e[1]))


def get_dataloader(opt, dataset, tokenizer, template=None):
    
    collator = dataset.get_collator()(
        opt, tokenizer, template
    )

    eval_sampler = SequentialSampler(dataset)

    dataloader = DataLoader(
        dataset,
        sampler=eval_sampler,
        drop_last = False,
        num_workers=opt.num_workers,
        collate_fn=collator,
        batch_size=opt.batch_size
    )
    return dataloader

class QACDataset(Dataset):
    def __init__(self, opt):
        super().__init__()
        self.is_vllm = opt.is_vllm
        self.data = self._load_corpus_and_data(opt)

    def _load_corpus_and_data(self, opt):
        with open(opt.data_dir, 'r') as f:
            data = json.load(f)
        return data
        # if "context" not in data[0]["ctxs"][0]:
        #     corpus = _load_wiki(opt.corpus_dir)
        #     result = []
        #     for d in data:
        #         d["ctxs"] = [ctxs for ctxs in d["ctxs"][:100] if ctxs["has_answer"]]
        #         if len(d["ctxs"]) > 0:
        #             for ctxs in d["ctxs"]:
        #                 ctxs["context"] = corpus[ctxs["id"]]["title"] + " " + corpus[ctxs["id"]]["text"]
        #             result.append(d)
        #     data_dir = opt.data_dir.split(".json")[0]
        #     with open(data_dir + "_100.json", 'w') as f: json.dump(result, f)
        # return result

    def get_data(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    
    def get_collator(self):
        raise NotImplementedError()

class AttackDataset(datasets.Dataset):
    def __init__(self, dataset, input_columns=["context"], label_map=None, label_names=None, output_scale_factor=None, shuffle=False):
        super().__init__(dataset, input_columns, label_map, label_names, output_scale_factor, shuffle)

    def __getitem__(self, i):
        """Return i-th sample."""
        return self._dataset[i]
    
class ReaderDataset(QACDataset):
    def get_collator(self):
        if self.is_vllm:
            return vLLMCollator
        else:
            return ReaderCollator

class RetrieverDataset(QACDataset):
    def get_collator(self):
        return RetrieverCollator

class vLLMCollator(object):

    def __init__(self, opt, tokenizer, template):
        self.tokenizer = tokenizer
        self.passage_maxlength = opt.passage_maxlength
        self.template = template
    
    def _truncate(self, inputs):
        return self.tokenizer.batch_decode(self.tokenizer(inputs, return_tensors="pt", max_length=self.passage_maxlength, padding=True, truncation=True).input_ids, skip_special_tokens=True)

    def __call__(self, batch):
        questions = [ex["question"] for ex in batch]
        contexts = self._truncate([ex["context"] for ex in batch])
        inputs = [self.template.format(q=q,d=c) for q,c in zip(questions, contexts)]

        keys = list(batch[0].keys())
        output = {key:[] for key in keys}
        for ex in batch:
            for key in keys:
                output[key].append(ex[key])
        output["inputs"] = inputs
        
        return output

class ReaderCollator(object):

    def __init__(self, opt, tokenizer, template):
        self.tokenizer = tokenizer
        self.passage_maxlength = opt.passage_maxlength
        self.template = template
    
    def _truncate(self, inputs):
        return self.tokenizer.batch_decode(self.tokenizer(inputs, return_tensors="pt", max_length=self.passage_maxlength, padding=True, truncation=True).input_ids, skip_special_tokens=True)

    def __call__(self, batch):
        questions = [ex["question"] for ex in batch]
        contexts = self._truncate([ex["context"] for ex in batch])
        inputs = [self.template.format(q=q,d=c) for q,c in zip(questions, contexts)]

        input_embeddings = self.tokenizer(
            inputs,
            max_length=512,
            truncation=True,
            padding=True, 
            return_tensors="pt",
        )

        keys = list(batch[0].keys())
        output = {key:[] for key in keys}
        for ex in batch:
            for key in keys:
                output[key].append(ex[key])
        output.update(input_embeddings)
        
        return output

class RetrieverCollator(object):

    def __init__(self, opt, tokenizer, template=None):
        self.tokenizer = tokenizer
        self.passage_maxlength = opt.passage_maxlength
        self.tokenizer_kwargs = {
            "max_length":opt.passage_maxlength,
            "truncation":True,
            "padding":True, 
            "return_tensors":"pt"
        }
    
    def _truncate(self, inputs):
        return self.tokenizer.batch_decode(self.tokenizer(inputs, return_tensors="pt", max_length=self.passage_maxlength, padding=True, truncation=True).input_ids, skip_special_tokens=True)

    def __call__(self, batch):
        questions = [ex["question"] for ex in batch]
        contexts = self._truncate([ex["context"] for ex in batch])
        og_contexts = self._truncate([ex["og_context"] for ex in batch])

        query_embeddings = self.tokenizer(
            questions,**self.tokenizer_kwargs
        )

        context_embeddings = self.tokenizer(
            contexts, **self.tokenizer_kwargs
        )

        og_context_embeddings = self.tokenizer(
            og_contexts, **self.tokenizer_kwargs
        )

        keys = list(batch[0].keys())
        output = {key:[] for key in keys}
        for ex in batch:
            for key in keys:
                output[key].append(ex[key])

        output['og_context_embeddings'] = og_context_embeddings
        output['context_embeddings']=context_embeddings
        output['query_embeddings']=query_embeddings

        return output