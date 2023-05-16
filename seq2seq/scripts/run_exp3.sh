

level=2
sample_ratio=0.1
model="t5-base"
seed_buf=(0 1 2)
dataset_buf=("qnli" "qqp")

# for seed in ${seed_buf[@]}
# do
# bash scripts/approx_linear.sh 6 cola $model $level $sample_ratio $seed
# done
                    
# for dataset in ${dataset_buf[@]}
# do
#     for seed in ${seed_buf[@]}
#     do
#     bash scripts/lora_approx_linear.sh 2 $dataset $model $level $sample_ratio $seed
#     done
# done

# for dataset in ${dataset_buf[@]}
# do
#     for seed in ${seed_buf[@]}
#     do
#     bash scripts/approx_linear_ablation.sh 2 $dataset $model $level $sample_ratio $seed deter
#     done
# done

for dataset in ${dataset_buf[@]}
do
    for seed in ${seed_buf[@]}
    do
    bash scripts/approx_linear_roberta.sh 2 $dataset $model $level $sample_ratio $seed
    # bash scripts/lora_approx_linear.sh 2 $dataset $model $level $sample_ratio $seed
    done
done


