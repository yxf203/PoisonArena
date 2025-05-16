import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoTokenizer,AutoModelForCausalLM
import os
from .Model import Model
from pathlib import Path
import transformers

class Llama(Model):
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
    
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=path,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )


    # the used version
    def query(self, msg):
        print("temperature:", self.temperature)
        results = self.pipeline(
            msg,
            temperature=self.temperature,
            max_new_tokens=self.max_output_tokens
        )
        # print(results)
        return results[0]["generated_text"][-1]["content"]
    
