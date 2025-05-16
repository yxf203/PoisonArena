from ml_collections import config_dict
import torch
from transformers import BitsAndBytesConfig,GPTQConfig#, AWQConfig 
def create_bnb_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_enable_fp32_cpu_offload=True
    )

    return bnb_config

def create_GptqConfig():
    gptq_config =  GPTQConfig(
        bits=4,
        # groupsize=12,
        disable_exllama=True

    )

    return gptq_config

# def create_AWQConfig():
#     awq_config =  AWQConfig(
#         bits=16,
#         groupsize=-1,
#     )
#     return awq_config

def get_config():
    config = config_dict.ConfigDict()

    # Experiment type
    config.transfer = False

    # General parameters 
    config.target_weight=1.0
    config.control_weight=0.0
    config.progressive_goals=False
    config.progressive_models=False
    config.anneal=False
    config.incr_control=False
    config.stop_on_success=True
    config.verbose=True
    config.allow_non_ascii=True
    config.num_train_models=1

    # Results
    config.result_prefix = 'results/'


    # tokenizers
    config.tokenizer_paths=['../../data/vicuna/vicuna-7b-v1.3']
    config.tokenizer_kwargs=[{"use_fast": False}, {"use_fast": False}]
    
    config.model_paths=['../../data/vicuna/vicuna-7b-v1.3']
    config.model_kwargs=[{"low_cpu_mem_usage": True, "use_cache": False}, {"low_cpu_mem_usage": True, "use_cache": False}]
    config.conversation_templates=['vicuna']
    config.devices = ['cuda:4','cuda:5', 'cuda:6', 'cuda:7']

    # data
    config.train_data = ''
    config.test_data = ''
    config.n_train_data = 1
    config.n_test_data = 0
    config.data_offset = 0

    # attack-related parameters
    config.attack = 'gcg'
    # config.control_init = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"
    config.control_init = "! ! ! ! ! ! ! ! ! !"
    config.n_steps = 500
    config.test_steps = 50
    config.batch_size = 512
    config.lr = 0.01
    config.topk = 256
    config.temp = 1
    config.filter_cand = True


    config.gbda_deterministic = True
    config.dataType = torch.float16
    config.insert_middle = False
    config.weighted_update = 0.8
    config.max_rand_pos = 60
    config.dynamic_pos = True

    # here we add id for the next step
    config.id=""

    return config
