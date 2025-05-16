
MODEL=$1
DATASET=$2
# OUTPUT_PATH=results/advp
OUTPUT_PATH=output/
k=$3
DIR_NAME=${DATASET}-${MODEL}-dcorpus-serials-ans6

mkdir -p $OUTPUT_PATH/$DIR_NAME

# for s in $(eval echo "{0..$((k-1))}"); do
for s in $(seq 0 $((k-1))); do

python src/attack_poison_specific_serials.py \
   --dataset ${DATASET} --split train \
   --model_code ${MODEL} \
   --num_cand 100 --per_gpu_eval_batch_size 16 --num_iter 100 --num_grad_iter 1 \
   --output_file ${OUTPUT_PATH}/${DIR_NAME} \
   --do_kmeans --k $k --kmeans_split $s
   # --output_file ${OUTPUT_PATH}/${DATASET}-${MODEL}-k${k}-s${s}.json \

done