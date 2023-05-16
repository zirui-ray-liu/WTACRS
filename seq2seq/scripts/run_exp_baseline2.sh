
model="t5-base"
dataset_buf=("qqp" "mnli")       # ("rte" "mrpc" "stsb" "cola" "sst2" "qnli") #
seed_buf=(0 1)

# seed_buf=(1 2)
# for seed in ${seed_buf[@]}
# do
# bash scripts/baseline.sh 3 qqp $model $seed
# done

# for seed in ${seed_buf[@]}
# do
# bash scripts/lora_baseline.sh 7 qqp $model $seed
# done

for seed in ${seed_buf[@]}
do
    for dataset in ${dataset_buf[@]}
    do
    bash scripts/ladder_side_tuning_baseline.sh 1 $dataset $model $seed
    done
done








