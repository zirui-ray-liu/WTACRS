# run_exp_baseline_4gpu.sh

model="t5-small"
dataset_buf=("rte") # "mrpc" "stsb" "cola" "sst2" "qnli" "qqp" "mnli")
seed_buf=(0)

for dataset in ${dataset_buf[@]}
do
    for seed in ${seed_buf[@]}
    do
    bash scripts/lora_baseline.sh "0,1,2,3" $dataset $model $seed # 32
    # bash scripts/baseline.sh "0,1,2,3" $dataset $model $seed # 32
    done
done