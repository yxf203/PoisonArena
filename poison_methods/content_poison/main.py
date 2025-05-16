'''A main script to run attack for LLMs.'''
import time
import importlib
import traceback
import numpy as np
# import torch.multiprocessing as mp
import torch
from absl import app
from ml_collections import config_flags
from attack import *
from attack.gcg.gcg_attack import GCGMultiPromptAttack, GCGAttackPrompt, GCGPromptManager
from attack import get_goals_and_targets, get_workers

import os

_CONFIG = config_flags.DEFINE_config_file('config')

# Function to import module at the runtime
def dynamic_import(module):
    return importlib.import_module(module)

def main(_):

    # mp.set_start_method('spawn')

    params = _CONFIG.value

    print(params)
    
    train_goals, train_targets, train_succ_flag, train_fail_flag, test_goals, test_targets = get_goals_and_targets(params)

    workers, test_workers = get_workers(params)
    

    managers = {
        "AP": GCGAttackPrompt,
        "PM": GCGPromptManager,
        "MPA": GCGMultiPromptAttack,
    }
    print("Managers:")
    print(managers)

    print(params.transfer)
    
    attack = IndividualPromptAttack(
        params,
        train_goals,
        train_targets,
        train_succ_flag,
        train_fail_flag,
        workers,
        control_init=params.control_init,
        logfile=f"{params.result_prefix}",
        managers=managers,
        test_goals=getattr(params, 'test_goals', []),
        test_targets=getattr(params, 'test_targets', []),
        test_workers=test_workers,
        mpa_deterministic=params.gbda_deterministic,
        mpa_lr=params.lr,
        mpa_batch_size=params.batch_size,
        mpa_n_steps=params.n_steps,
        insert_middle = params.insert_middle,
        weighted_update = params.weighted_update,
        dynamic_pos = params.dynamic_pos,
    )
    try:
        attack.run(
            n_steps=params.n_steps,
            batch_size=params.batch_size, 
            data_offset=params.data_offset,
            topk=params.topk,
            temp=params.temp,
            target_weight=params.target_weight,
            control_weight=params.control_weight,
            test_steps=getattr(params, 'test_steps', 1),
            anneal=params.anneal,
            incr_control=params.incr_control,
            stop_on_success=params.stop_on_success,
            verbose=params.verbose,
            filter_cand=params.filter_cand,
            allow_non_ascii=params.allow_non_ascii,
            # dynamic_pos = params.dynamic_pos
        )
        for worker in workers + test_workers:
            worker.stop()
        print('attack done')
        
    except Exception as e:
        for worker in workers + test_workers:
            worker.stop()
        traceback.print_exc()
        exit(1)
    

if __name__ == '__main__':
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:4280"
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()    
    app.run(main)