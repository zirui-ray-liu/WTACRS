# run_exp_4gpu.sh

model="t5-3b"
dataset_buf=("rte") # ("rte" "mrpc" "stsb" "cola" "sst2" "qnli" "qqp" "mnli") # 
level=2
sample_ratio=0.1
seed_buf=(9)

for seed in ${seed_buf[@]}
    do
    for dataset in ${dataset_buf[@]}
    do
    bash scripts/approx_linear_4gpu.sh "0,1,2,3" $dataset $model $level $sample_ratio $seed
    bash scripts/lora_approx_linear_4gpu.sh "0,1,2,3" $dataset $model $level $sample_ratio $seed 64
    done
done

