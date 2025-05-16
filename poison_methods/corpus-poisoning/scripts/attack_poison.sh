MODEL=$1
DATASET=$2
# OUTPUT_PATH=results/advp
OUTPUT_PATH=output/
k=$3
DIR_NAME=${DATASET}-${MODEL}-dcorpus-dot-inco-ans-6-i100

mkdir -p $OUTPUT_PATH/$DIR_NAME


python src/attack_poison_specific.py \
   --dataset ${DATASET} --split train \
   --model_code ${MODEL} \
   --num_cand 100 --per_gpu_eval_batch_size 16 --num_iter 100 --num_grad_iter 1 \
   --output_file ${OUTPUT_PATH}/${DIR_NAME} \

done