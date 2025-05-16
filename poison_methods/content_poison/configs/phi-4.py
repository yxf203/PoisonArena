import os

import torch

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():
    config = default_config()
    
    config.model_name = "phi-4"

    config.result_prefix = 'results/'

    config.tokenizer_paths = ["../models/Phi-4-mini-instruct"]
    config.model_paths = ["../models/Phi-4-mini-instruct"] 

    config.conversation_templates = ['one-shot']

    config.model_kwargs = [{
        'device_map': 'auto'
    }]

    config.data_type = torch.float16
    config.devices = ['cuda']

    return config
