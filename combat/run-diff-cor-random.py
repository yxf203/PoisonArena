import os
import time
def run(test_params):

    log_file, log_name = get_log_name(test_params)

    print(log_file, log_name)
    cmd = f"nohup python3 -u main/combat_clean-diff-corpus-random.py \
        --eval_model_code {test_params['eval_model_code']}\
        --eval_dataset {test_params['eval_dataset']}\
        --split {test_params['split']}\
        --query_results_dir {test_params['query_results_dir']}\
        --model_name {test_params['model_name']}\
        --top_k {test_params['top_k']}\
        --use_truth {test_params['use_truth']}\
        --gpu_id {test_params['gpu_id']}\
        --attack_method {test_params['attack_method']}\
        --adv_per_query {test_params['adv_per_query']}\
        --score_function {test_params['score_function']}\
        --repeat_times {test_params['repeat_times']}\
        --M {test_params['M']}\
        --seed {test_params['seed']}\
        --name {log_name}\
        > {log_file} &"
        
    os.system(cmd)


def get_log_name(test_params):
    # Generate a log file name
    os.makedirs(f"logs/{test_params['query_results_dir']}_logs", exist_ok=True)

    if test_params['use_truth']:
        log_name = f"{test_params['eval_dataset']}-{test_params['eval_model_code']}-{test_params['model_name']}-Truth--M{test_params['M']}x{test_params['repeat_times']}"
    else:
        log_name = f"{test_params['eval_dataset']}-{test_params['eval_model_code']}-{test_params['model_name']}-Top{test_params['top_k']}--M{test_params['M']}x{test_params['repeat_times']}"
    
    if test_params['attack_method'] != None:
        log_name += f"-adv-{test_params['attack_method']}-{test_params['score_function']}-{test_params['adv_per_query']}-{test_params['top_k']}"

    if test_params['note'] != None:
        log_name = test_params['note']
    moment = time.strftime("%Y%m%d_%H%M")
    return f"logs/{test_params['query_results_dir']}_logs/{moment}_{log_name}.txt", log_name


model_name = "phi4"  # "vicuna7b" "llama3b" "llama8b" "phi4" "gpt3.5" "gpt4o"
test_params = {
    # beir_info
    'eval_model_code': "contriever",
    'eval_dataset': "nq",
    'split': "test",
    'query_results_dir': f'combat-random-{model_name}',

    # LLM setting

    'model_name': model_name,

    'top_k': 5,
    # 'top_k': 10,
    'gpu_id': 0,

    # attack
    'adv_per_query': 5,
    'score_function': 'dot',
    'seed': 12,
    
    # combat setting
    'is_random': False,
    'iteration': 10,
    'combat_n': 2

}

# for dataset in ['nq', 'hotpotqa', 'msmarco']:
for dataset in ['nq']:
    test_params['eval_dataset'] = dataset
    run(test_params)