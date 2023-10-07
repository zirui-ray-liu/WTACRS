# run_exp_baseline_4gpu.sh

model="t5-3b"
dataset_buf=("rte") # ("rte" "mrpc" "stsb" "cola" "sst2" "qnli") # 
seed_buf=(9)

for dataset in ${dataset_buf[@]}
do
    for seed in ${seed_buf[@]}
    do
    # bash scripts/baseline_4gpu.sh "0,1,2,3" $dataset $model $seed # OOM
    bash scripts/lora_baseline_4gpu.sh "0,1,2,3" $dataset $model $seed 64
    # bash scripts/ladder_side_tuning_t5-3b_baseline.sh "0,1,2,3" $dataset $model $seed 
    done
done

