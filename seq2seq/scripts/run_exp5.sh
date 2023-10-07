level=2
sample_ratio=0.3
# model="t5-large"
# model="bert-large-cased"
model="t5-base"
seed_buf=(0)
dataset_buf=("rte") #  "qnli"

for seed in ${seed_buf[@]}
    do
    for dataset in ${dataset_buf[@]}
    do
    bash scripts/lora_approx_linear.sh 0 $dataset $model $level $sample_ratio $seed 32
    # bash scripts/lora_approx_linear_bert.sh 0 $dataset $model $level $sample_ratio $seed
    # bash scripts/approx_linear_bert.sh 0 $dataset $model $level $sample_ratio $seed       
    # bash scripts/approx_linear.sh 0 $dataset $model $level $sample_ratio $seed
    done
done