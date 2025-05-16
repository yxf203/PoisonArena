import torch.distributed as dist

from lightning.pytorch.callbacks import BasePredictionWriter
from typing import Any, Sequence, Optional, List

import numpy as np
import collections
import logging
import string
import torch
import json
import time
import csv
import sys
import os
import re

logger = logging.getLogger(__name__)

def _load_wiki(dir):
    reader = csv.reader(open(dir, encoding="utf-8"),
                        delimiter="\t", quoting=csv.QUOTE_MINIMAL)
    next(reader)
    wiki = {}
    for _, row in enumerate(reader):
        id, text, title = int(row[0]), row[1], row[2]
        wiki[id] = {"text": text, "title": title}
    return wiki

def get_sim(emb1, emb2):
    result = []
    for i,j in zip(emb1, emb2):
        result.append(np.sum(i.numpy() * j.numpy()))
    return torch.tensor(result)

def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def is_main(args):
    if args.gpus == 1:
        return True
    else:
        return get_rank() == 0

def init_logger(args, stdout_only=False):
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [stdout_handler]
    if not stdout_only:
        file_handler = logging.FileHandler(filename=os.path.join(args.output_dir, "run.log"))
        handlers.append(file_handler)
    logging.basicConfig(
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main(args) else logging.WARN,
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        handlers=handlers,
        force=True
    )
    logger.info("Log Set")
    return logger

def timestr():
    return time.strftime("%Y%m%d-%H%M%S")

class CustomWriter(BasePredictionWriter):

    def __init__(self, opt, name, wrtie_interval="epoch"):
        super().__init__(wrtie_interval)
        self.results = []
        self.out_dir = opt.output_dir
        self.output_name = name

    def write_on_epoch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        predictions: Sequence[Any],
        batch_indices: Optional[Sequence[Any]],
    ):
        result = []
        for dd in predictions:
            for d in dd:
                result.append(d)
        with open(os.path.join(self.out_dir, self.output_name), 'w') as f:
            json.dump(result, f)


class Distributed_Writer(BasePredictionWriter):

    PRED_FILENAME_EXT = "json"
    def __init__(self, opt, name, wrtie_interval="epoch"):
        super().__init__(wrtie_interval)
        self.out_dir = opt.output_dir
        self.gpus = opt.gpus
        self.name = name

    def get_pred_file_name(self, global_rank=None):
        file_name = "pred_{}.{}".format(global_rank, self.PRED_FILENAME_EXT)
        return os.path.join(self.out_dir, file_name)

    def on_predict_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        self.results = [[] for _ in range(trainer.world_size)]

    def write_on_epoch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        predictions: Sequence[Any],
        batch_indices: Optional[Sequence[Any]],
    ):
        result = []
        for dd in predictions:
            for d in dd:
                result.append(d)
        file_dir = self.get_pred_file_name(trainer.global_rank)
        with open(file_dir, 'w') as f:
            json.dump(result, f)

    def write_total_result(self):
        result = []
        for i in range(self.gpus):
            file_dir = "pred_{}.json".format(i)
            with open(os.path.join(self.out_dir, file_dir), 'r') as f:
                data = json.load(f)
            result += data
        with open(os.path.join(self.out_dir, self.name),'w') as f:
            json.dump(result,f)

def load_writer(opt, name):
    if opt.gpus > 1:
        writer = Distributed_Writer(opt, name)
    else:
        writer = CustomWriter(opt, name)
    return [writer]


def _normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def EM(golds, pred):
    pred = _normalize_answer(pred)
    golds = [_normalize_answer(g) for g in golds]
    cor = max([gold == pred for gold in golds])
    return cor

def _compute_f1(gold, pred):
    common = collections.Counter(gold) & collections.Counter(pred)
    num_same = sum(common.values())
    if len(gold) == 0 or len(pred) == 0:
        return int(gold == pred)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred)
    recall = 1.0 * num_same / len(gold)
    f1_score = (2 * precision * recall) / (precision + recall)
    return f1_score

def f1(golds, pred):
    pred = _normalize_answer(pred).split()
    golds = [_normalize_answer(gold).split() for gold in golds]
    f1_scores = max([_compute_f1(gold, pred) for gold in golds])
    return f1_scores

def acc(golds, pred):
    acc = max([gold in pred for gold in golds])
    return acc

def norm_acc(golds, pred):
    pred = _normalize_answer(pred)
    golds = [_normalize_answer(gold) for gold in golds]
    acc = max([gold in pred for gold in golds])
    return acc