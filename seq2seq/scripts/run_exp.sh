level=2
sample_ratio=0.1
model="t5-large"
# model="bert-large-cased"
# model="facebook/opt-350m"
seed_buf=(30)
dataset_buf=("mrpc") # "rte" "cola" "mrpc" "stsb") # "sst2" "qnli") # "qqp" "mnli")

for seed in ${seed_buf[@]}
    do
    for dataset in ${dataset_buf[@]}
    do
    # bash scripts/lora_approx_linear.sh 0 $dataset $model $level $sample_ratio $seed 32
    # bash scripts/lora_approx_linear_bert.sh 0 $dataset $model $level $sample_ratio $seed
    # bash scripts/approx_linear_bert.sh 0 $dataset $model $level $sample_ratio $seed
    bash scripts/approx_linear.sh 0 $dataset $model $level $sample_ratio $seed
    # bash scripts/lora_approx_linear_opt.sh 0 $dataset $model $level $sample_ratio $seed
    done
done


