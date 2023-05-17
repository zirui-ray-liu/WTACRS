# model="t5-base"
# model="roberta-base"
# model="bert-base-cased"
model="t5-3b"
dataset_buf=("rte") # ("rte" "mrpc" "stsb" "cola") # "sst2" "qnli") # "qqp" "mnli")
seed_buf=(10)
batchsize_buf=(8)   

for seed in ${seed_buf[@]}
do
    for dataset in ${dataset_buf[@]}
    do
        for batchsize in ${batchsize_buf[@]}
        do

            # bash scripts/baseline_bert.sh 0 $dataset $model $seed
            # bash scripts/lora_baseline_bert.sh 0 $dataset $model $seed
            # bash scripts/ladder_side_tuning_baseline.sh 0 $dataset $model $seed
            # bash scripts/baseline.sh 0 $dataset $model $seed
            # bash scripts/lora_baseline.sh 0 $dataset $model $seed
            bash scripts/baseline_3b.sh 0 $dataset $model $seed
            # bash scripts/lora_baseline_3b.sh 0 $dataset $model $seed $batchsize

        done
    done
done
