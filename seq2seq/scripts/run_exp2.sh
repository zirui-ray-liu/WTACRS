
level=2
sample_ratio=0.3
# model="t5-small"
model="roberta-base
seed_buf=(0 1)
dataset_buf=("sst2" "mnli")

# for seed in ${seed_buf[@]}
# do
# bash scripts/approx_linear.sh 6 cola $model $level $sample_ratio $seed
# done

# for seed in ${seed_buf[@]}
# do
# bash scripts/lora_approx_linear.sh 2 mnli $model $level $sample_ratio $seed
# done

# for dataset in ${dataset_buf[@]}
# do
#     for seed in ${seed_buf[@]}
#     do
#     # bash scripts/approx_linear_ablation.sh 1 $dataset $model $level $sample_ratio $seed uniform
#     bash scripts/approx_linear.sh 1 $dataset $model $level $sample_ratio $seed
#     done
# done

for dataset in ${dataset_buf[@]}
do
    for seed in ${seed_buf[@]}
    do
    bash scripts/approx_linear_roberta.sh 1 $dataset $model $level $sample_ratio $seed
    # bash scripts/lora_approx_linear.sh 1 $dataset $model $level $sample_ratio $seed
    done
done












