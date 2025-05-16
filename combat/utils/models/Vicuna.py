from fastchat.model import load_model, get_conversation_template
import torch
import transformers
from .Model import Model


class Vicuna(Model):
    def __init__(self, config):
        super().__init__(config)
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        self.device = config["params"]["device"]
        self.max_output_tokens = config["params"]["max_output_tokens"]
        print("--------------------------------------------------------------")
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
