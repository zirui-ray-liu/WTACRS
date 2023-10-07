level=2
sample_ratio=0.3
# model="t5-large"
# model="bert-large-cased"
model="facebook/opt-350m"
seed_buf=(0)
dataset_buf=("mnli") # ("sst2" "qnli") # ("rte" "mrpc" "stsb" "cola") # 

for seed in ${seed_buf[@]}
    do
    for dataset in ${dataset_buf[@]}
    do
    # bash scripts/lora_approx_linear_bert.sh 0 $dataset $model $level $sample_ratio $seed
    # bash scripts/approx_linear_bert.sh 0 $dataset $model $level $sample_ratio $seed
    # bash scripts/approx_linear.sh 0 $dataset $model $level $sample_ratio $seed
    # bash scripts/lora_approx_linear.sh 0 $dataset $model $level $sample_ratio $seed 64
    bash scripts/lora_approx_linear_opt.sh 0 $dataset $model $level $sample_ratio $seed
    done
done



# level=2
# sample_ratio=0.3
# # model="t5-large"
# model="bert-large-cased"
# seed_buf=(4)
# dataset_buf=("mnli") #  "qnli"

# # for seed in ${seed_buf[@]}
# # do
# #     bash scripts/approx_linear.sh 0 qnli $model $level $sample_ratio $seed
# # done

# # for seed in ${seed_buf[@]}
# # do
# # bash scripts/lora_approx_linear.sh 0 stsb $model $level $sample_ratio $seed
# # done

# for seed in ${seed_buf[@]}
#     do
#     for dataset in ${dataset_buf[@]}
#     do
#     bash scripts/lora_approx_linear_bert.sh 0 $dataset $model $level $sample_ratio $seed
#     # bash scripts/approx_linear_bert.sh 0 $dataset $model $level $sample_ratio $seed       
#     # bash scripts/approx_linear.sh 0 $dataset $model $level $sample_ratio $seed
#     done
# done