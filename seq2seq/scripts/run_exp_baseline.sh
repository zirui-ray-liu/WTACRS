# model="t5-base"
model="roberta-base"
dataset_buf=("rte") # "rte" "mrpc" "stsb" "cola" "sst2" "qnli" "qqp" "mnli")
seed_buf=(0)

# for seed in ${seed_buf[@]}
# do
# bash scripts/baseline.sh 2 mnli $model $seed
# done

# for seed in ${seed_buf[@]}
# do
# bash scripts/lora_baseline.sh 6 mnli $model $seed
# done


# for seed in ${seed_buf[@]}
# do
#     for dataset in ${dataset_buf[@]}
#     do
#     bash scripts/ladder_side_tuning_baseline.sh 0 $dataset $model $seed
#     done
# done

for seed in ${seed_buf[@]}
do
    for dataset in ${dataset_buf[@]}
    do
    bash scripts/baseline_roberta.sh 0 $dataset $model $seed
    done
done
