import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoTokenizer,AutoModelForCausalLM
import os
from .Model import Model
from pathlib import Path
import transformers

class Phi(Model):
    def __init__(self, config):
        super().__init__(config)
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        self.device = config["params"]["device"]
        self.max_output_tokens = config["params"]["max_output_tokens"]
        print("--------------------------------------------------------------")
        api_pos = int(config["api_key_info"]["api_key_use"])
        hf_token = config["api_key_info"]["api_keys"][api_pos]
        path = config["path"]
        print(path)
        model = AutoModelForCausalLM.from_pretrained(
            path,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(path)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

    # the used version
    def query(self, msg):
        print("temperature:", self.temperature)
        results = self.pipeline(
            msg,
            temperature=self.temperature,
            do_sample=True,
            max_new_tokens=self.max_output_tokens
        )
        # print(results)
        return results[0]["generated_text"][-1]["content"]
        # return results[0]['generated_text']
    