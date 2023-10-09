# for T5 experiment
GPUID=0
sample_ratio=0.3
level=2
# model=t5-base # t5-large, t5-3b
# for dataset in "rte" "mrpc" "stsb" "cola" "sst2" "qnli" "qqp" "mnli";
# do
#     for seed in 0 1 2;
#     do
#     bash scripts/lora_approx_linear.sh $GPUID $dataset $model $level $sample_ratio $seed 32
#     done
# done

# for Bert experiment
model=bert-base-uncased # bert-large-uncased
for dataset in "rte" "mrpc" "stsb" "cola" "sst2" "qnli" "qqp" "mnli";
do
    for seed in 0 1 2;
    do
    bash scripts/lora_approx_linear_bert.sh $GPUID $dataset $model $level $sample_ratio $seed
    done
done