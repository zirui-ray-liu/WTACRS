# model="t5-large"
model="bert-large-cased"
seed_buf=(0 1)
dataset_buf=("cola") # "rte" "mrpc" "stsb" "cola") # "sst2" "qnli") # "qqp" "mnli")

for seed in ${seed_buf[@]}
do
    for dataset in ${dataset_buf[@]}
    do
    # bash scripts/baseline_bert.sh 0 $dataset $model $seed
    bash scripts/lora_baseline_bert.sh 0 $dataset $model $seed
    done
done