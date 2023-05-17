model="t5-base"
# model="t5-large"
# model="bert-base-cased"
# model="bert-large-cased"
dataset_buf=("rte" "mrpc" "stsb" "cola" "sst2" "qnli" "qqp" "mnli")
seed_buf=(0 1 2)

for seed in ${seed_buf[@]}
do
    for dataset in ${dataset_buf[@]}
    do
        # bash scripts/baseline_bert.sh 0 $dataset $model $seed
        # bash scripts/lora_baseline_bert.sh 0 $dataset $model $seed
        bash scripts/baseline.sh 0 $dataset $model $seed
        # bash scripts/lora_baseline.sh 0 $dataset $model $seed
    done
done
