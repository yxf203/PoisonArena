#! /bin/sh

# Attacking a whole diverse query set (MSMARCO eval set), by crafting an adversarial passage with GASLITE, using portion of 'held-in' queries.

# DATASET=msmarco-test
DATASET=nq-test
# MODEL=sentence-transformers/all-MiniLM-L6-v2
MODEL=../retriever/facebook/contriever
# SIM_FUNC=cos_sim
SIM_FUNC=dot

# Target choice:
N_CLUSTERS=100  # budget size, also the # of crafted adv. passages
RANDOM_SEED=0  # determines the sampled query
# Additional config:
BATCH_SIZE=2048

python hydra_entrypoint.py --config-name default \
    model.sim_func_name=${SIM_FUNC} "model.model_hf_name=${MODEL}" dataset=${DATASET} \
    core_objective=covering batch_size=${BATCH_SIZE}\
    exp_tag=exp2_knows-nothing random_seed="${RANDOM_SEED}" constraints=as-suffix-tox 


