from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers import LlamaForCausalLM, LlamaTokenizer, MistralForCausalLM
from vllm import LLM, SamplingParams
from openai import OpenAI
from .util import f1

import lightning.pytorch as pl

import os
import math
import torch
import logging

cls_mapping = {
    "Llama-8b": (AutoModelForCausalLM, AutoTokenizer, True, "Llama-3-8B-Instruct"),
    "Llama3.2-3b": (AutoModelForCausalLM, AutoTokenizer, True, "Llama-3.2-3B-Instruct"),
    "phi-4-mini": (AutoModelForCausalLM, AutoTokenizer, True, "Phi-4-mini-instruct"),
    "vicuna-7b": (LlamaForCausalLM, LlamaTokenizer, True, "vicuna-7b-v1.5"),
}

logger = logging.getLogger(__name__)

save_keys = [
    "question", "doc_id", "question", "answers"
]

def load_reader(opt):
    if opt.reader == "chatgpt":
        return Reader_GPT(opt)
    elif opt.is_vllm:
        return Reader_vLLM(opt)
    else:
        return Reader(opt)

def _load_model(opt):
    reader_name = opt.reader
    if reader_name in cls_mapping:
        return cls_mapping[reader_name]
    else:
        NotImplementedError()

class Reader(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        model_cls, tokenizer_cls, self.is_decoder, hf_name = _load_model(opt)
        self.model = model_cls.from_pretrained(os.path.join(opt.model_dir, hf_name)).to("cuda:0")
        self.tokenizer = tokenizer_cls.from_pretrained(os.path.join(opt.model_dir, hf_name))
        terminators = [
            self.tokenizer.eos_token_id, 
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")  
        ]
        self.generate_kwargs = dict(
            max_new_tokens=opt.max_new_tokens,
            return_dict_in_generate=True,
            output_hidden_states=True,
            output_scores=True,
            eos_token_id=terminators
        )
        if self.is_decoder:
            self.tokenizer.padding_side = "left"

        self.tokenizer.pad_token = "<|eot_id|>"

    
    def forward(self, input_ids, attention_mask):

        outputs = self.model.generate(input_ids=input_ids.to(self.model.device), attention_mask=attention_mask.to(self.model.device), **self.generate_kwargs)
        preds = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        print("preds:",preds)
        return preds
    

    def _cal_label_prob(self, probs, labels):
        result = []
        # print("here is probs", probs)
        # print("here is labels", labels)
        for prob, label in zip(probs, labels):
            mask = label > 0
            prob, label = prob[mask], label[mask]
            log_softmax = torch.nn.functional.log_softmax(prob, dim=-1)
            # from IPython import embed; embed(); exit(0)
            nll = -log_softmax.gather(1, label.unsqueeze(0).transpose(0, 1))
            avg_nll = torch.sum(nll, dim=0) * -1
            result.append(float(torch.exp(avg_nll / float(label.shape[0]))))
        return result
    
    def get_scores(self, input_ids, label_ids):

        outputs = self.model(input_ids=input_ids.to(self.model.device), labels=label_ids.to(self.model.device))
        # print(outputs)
        scores = self._cal_label_prob(outputs.logits, label_ids.to(self.model.device))
        print("finish computing scores!")
        return scores
    
    def get_tokenizer(self):
        return self.tokenizer

class Reader_vLLM(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        _, tokenizer_cls, _, hf_name = _load_model(opt)
        self.model = LLM(model=os.path.join(opt.model_dir, hf_name), gpu_memory_utilization=0.35, kv_cache_dtype="fp8_e5m2")
        self.tokenizer = tokenizer_cls.from_pretrained(os.path.join(opt.model_dir, hf_name))
        terminators = [
            self.tokenizer.eos_token_id,  
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>") 
        ]
        self.tokenizer.pad_token = "<|eot_id|>"
        self.tokenizer.add_special_tokens({'pad_token': '<|eot_id|>'})
        self.gen_sampling = SamplingParams(temperature=1, top_p=1, max_tokens=30,stop_token_ids=terminators)
        self.score_sampling = SamplingParams(temperature=1, top_p=1, prompt_logprobs=0, max_tokens=1,stop_token_ids=terminators)



    def _cal_label_prob(self, outputs, labels):
        labels = [input_id for input_id in self.tokenizer(labels).input_ids]
        probs = [output.prompt_logprobs for output in outputs]
        result = []
        
        for prob, label in zip(probs, labels):
            prs = []

            prob = [p for p in prob if p is not None]
            prob_tokens = [list(p.keys())[0] for p in prob]
            min_len = min(len(prob_tokens), len(label))
            

            for i in range(1, min_len + 1):
                k = prob_tokens[-i]
                l = label[-i]
                if k == l:
                    pr = prob[-i]  
                    prs.append(list(pr.values())[0].logprob)
                else:
                    break 
            
            if prs: 
                avg_nll = sum(prs)
                result.append(math.exp(avg_nll) / len(prs)) 
            else:
                print("errorrrrrr!")
                result.append(0.0) 
        
        return result

    def forward(self, inputs):
        # print("forward inputs:", inputs)
        preds= [output.outputs[0].text.strip() for output in self.model.generate(inputs, use_tqdm=False, sampling_params=self.gen_sampling)]
        # print("preds:", preds)
        return preds
    
    def get_scores(self, inputs, labels):
        # print("score inputs:", inputs)
        outputs = self.model.generate(inputs, use_tqdm=False, sampling_params=self.score_sampling)
        scores = self._cal_label_prob(outputs, labels)
        return scores
    
    def get_tokenizer(self):
        return self.tokenizer
    
class Reader_GPT(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        OPENAI_API_KEY = opt.openai_key
        self.client = OpenAI(
            api_key=OPENAI_API_KEY
        )
        self.system_prompt = "You are a QA assistant. Read the document and answer the question. Your answer should be concise and short phrase, not sentence."
    

    def _cal_label_prob(self, outputs, labels):
        raise NotImplementedError
    
    def forward(self, contexts, question):
        preds = []
        for context in contexts:
            completion = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": "Document: {}\nQuestion: {}".format(context, question)}
                ],
                logprobs=True
            )
            preds.append(completion.choices[0].message.content)
        return preds
    
    def get_scores(self, contexts, question, answers):
        from math import exp

        scores = []
        for context in contexts:
            completion = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                n=10,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": "Document: {}\nQuestion: {}".format(context, question)}
                ],
                logprobs=True
            )
            score = 0
            for choice in completion.choices:
                pred = choice.message.content
                if f1(answers, pred) > 0.5:
                    for token in choice.logprobs.content:
                        score += token.logprob
                    score = exp(score)
                    break
            scores.append(score)
        return scores
    
    def get_tokenizer(self):
        raise NotImplementedError

class Read_Module(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        if opt.is_vllm:
            self.model = Reader(opt)
        else:
            self.model = Reader_vLLM(opt)
        self.is_vllm = opt.is_vllm
        logger.info("Model Load Done")

    # def forward(self, input_ids, attention_mask):
    #     preds = self.model(input_ids, attention_mask)
    #     return preds

    def predict_step(self, batch, batch_idx):
        if self.is_vllm:
            preds = self.model(batch['inputs'])
        else:
            preds = self.model(batch['input_ids'], batch['attention_mask'])
        result = self._process_output(preds, batch)
        return result
    
    def _process_output(self, preds, batch):
        keys = list(batch.keys())
        result = []
        for i in range(len(preds)):
            instance = {}
            for key in keys:
                if not isinstance(batch[key][i],torch.Tensor) and key in save_keys:
                    instance[key] = batch[key][i]
            instance["pred"] = preds[i]
            result.append(instance)
        # result = [{
        #     "question": batch["question"][i],
        #     "context": batch["context"][i],
        #     "answers": batch["answers"][i],
        #     "pred": preds[i],
        # }  for i in range(len(preds))]
        return result
    