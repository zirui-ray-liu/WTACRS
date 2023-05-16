level=2
sample_ratio=0.3


model="roberta-base"
# model="t5-small"
seed_buf=(0)
dataset_buf=("rte") # "rte" "mrpc" "stsb" "cola" "sst2" "qnli" "qqp" "mnli") # ("rte") # "mrpc" "stsb" "cola" "sst2")  #

# for seed in ${seed_buf[@]}
# do
# bash scripts/approx_linear.sh 1 rte $model $level $sample_ratio $seed
# done

# for seed in ${seed_buf[@]}
# do
# bash scripts/lora_approx_linear.sh 1 rte $model $level $sample_ratio $seed
# done

for dataset in ${dataset_buf[@]}     
do
    for seed in ${seed_buf[@]}
    do
    bash scripts/approx_linear_bert.sh 1 $dataset $model $level $sample_ratio $seed
    # bash scripts/lora_approx_linear.sh 0 $dataset $model $level $sample_ratio $seed
    done
done

# for dataset in ${dataset_buf[@]}
# do
#     for seed in ${seed_buf[@]}
#     do
#     bash scripts/approx_linear_ablation.sh 0 $dataset $model $level $sample_ratio $seed randomize
#     done
# done



