


level=2
sample_ratio=0.1
model="t5-base"
seed_buf=(0 1)
dataset_buf=("rte" "mrpc" "stsb" "cola" "sst2") # "qnli")

# for seed in ${seed_buf[@]}
# do
# bash scripts/approx_linear.sh 6 cola $model $level $sample_ratio $seed
# done

# for seed in ${seed_buf[@]}
# do
# bash scripts/lora_approx_linear.sh 3 rte $model $level $sample_ratio $seed 80
# done

# for dataset in ${dataset_buf[@]}
# do
#     for seed in ${seed_buf[@]}
#     do
#     bash scripts/approx_linear.sh 4 $dataset $model $level $sample_ratio $seed
#     done
# done


for dataset in ${dataset_buf[@]}
do
    for seed in ${seed_buf[@]}
    do
    bash scripts/approx_linear_ablation.sh 3 $dataset $model $level $sample_ratio $seed randomize
    done
done

