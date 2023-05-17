level=2
sample_ratio=0.1
# model="roberta-base"
# model="t5-small"
# model="bert-base-cased"
model="t5-3b"
seed_buf=(9)
dataset_buf=("rte") # "rte" "mrpc" "stsb" "cola") # "sst2" "qnli" "qqp" "mnli") # ("rte") # "mrpc" "stsb" "cola" "sst2") #
batchsize_buf=(38 32 16 8)

for seed in ${seed_buf[@]}
do
    for dataset in ${dataset_buf[@]}
    do
        for batchsize in ${batchsize_buf[@]}
        do
            # bash scripts/approx_linear_bert.sh 3 $dataset $model $level $sample_ratio $seed
            # bash scripts/lora_approx_linear_bert.sh 0 $dataset $model $level $sample_ratio $seed
            # bash scripts/approx_linear.sh 1 rte $model $level $sample_ratio $seed
            bash scripts/lora_approx_linear_3b.sh 0 $dataset $model $level $sample_ratio $seed $batchsize
        done
    done
done

# for dataset in ${dataset_buf[@]}
# do
#     for seed in ${seed_buf[@]}
#     do
#     bash scripts/approx_linear_ablation.sh 0 $dataset $model $level $sample_ratio $seed randomize
#     done
# done



