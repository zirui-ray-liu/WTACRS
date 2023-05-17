level=2
sample_ratio=0.3
model="t5-base"
# model="bert-base-cased"
# model="t5-large"
# model="bert-large-cased"
seed_buf=(0 1 2)
dataset_buf=("rte" "mrpc" "stsb" "cola" "sst2" "qnli" "qqp" "mnli")

for seed in ${seed_buf[@]}
do
    for dataset in ${dataset_buf[@]}
    do
        for batchsize in ${batchsize_buf[@]}
        do
            # bash scripts/approx_linear_bert.sh 0 $dataset $model $level $sample_ratio $seed
            # bash scripts/lora_approx_linear_bert.sh 0 $dataset $model $level $sample_ratio $seed
            bash scripts/approx_linear.sh 0 $dataset $model $level $sample_ratio $seed
            # bash scripts/lora_approx_linear.sh 0 $dataset $model $level $sample_ratio $seed
        done
    done
done




