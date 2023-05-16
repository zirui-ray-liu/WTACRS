


level=2
sample_ratio=0.1
model="t5-base"
seed_buf=(0 1)       
dataset_buf=("cola" "mrpc" "stsb" "sst2") # ("mnli")  "rte"

# for seed in ${seed_buf[@]}   
# do
# bash scripts/approx_linear.sh 1 rte $model $level $sample_ratio $seed
# done

# for seed in ${seed_buf[@]}
# do
# bash scripts/lora_approx_linear.sh 7 cola $model $level $sample_ratio $seed
# done

# for dataset in ${dataset_buf[@]}
# do
#     for seed in ${seed_buf[@]}
#     do
#     bash scripts/approx_linear.sh 6 $dataset $model $level $sample_ratio $seed
#     done
# done

for dataset in ${dataset_buf[@]}
do
    for seed in ${seed_buf[@]}
    do
    bash scripts/approx_linear_ablation.sh 5 $dataset $model $level $sample_ratio $seed deter
    done
done




