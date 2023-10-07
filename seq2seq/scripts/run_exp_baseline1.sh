# model="t5-large"
model="t5-base"
# model="bert-large-cased"
# model="facebook/opt-350m"

seed_buf=(0)
dataset_buf=("mrpc") # ("rte" "cola" "mrpc" "stsb") # "sst2" "qnli") # "qqp" "mnli")

for seed in ${seed_buf[@]}
do
    for dataset in ${dataset_buf[@]}
    do
    bash scripts/baseline.sh 0 $dataset $model $seed
    # bash scripts/baseline_bert.sh 0 $dataset $model $seed
    # bash scripts/lora_baseline_bert.sh 0 $dataset $model $seed
    # bash scripts/ladder_side_tuning_baseline.sh 0 $dataset $model $seed
    # bash scripts/lora_baseline_opt.sh 0 $dataset $model $seed
    # bash scripts/baseline_opt.sh 0 $dataset $model $seed
    done
done