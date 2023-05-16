


level=2
sample_ratio=0.1
model="t5-base"
seed_buf=(0 1)
dataset_buf=("rte" "mrpc" "stsb" "cola" "sst2") # ("mnli")

# for seed in ${seed_buf[@]}
# do
# bash scripts/approx_linear.sh 0 cola $model $level $sample_ratio $seed
# done

# for seed in ${seed_buf[@]}
# do
# bash scripts/lora_approx_linear.sh 6 sst2 $model $level $sample_ratio $seed
# done

# for dataset in ${dataset_buf[@]}
# do
#     for seed in ${seed_buf[@]}
#     do
#     bash scripts/approx_linear.sh 5 $dataset $model $level $sample_ratio $seed
#     done
# done


for dataset in ${dataset_buf[@]}
do
    for seed in ${seed_buf[@]}
    do
    bash scripts/approx_linear_ablation.sh 4 $dataset $model $level $sample_ratio $seed uniform
    done
done