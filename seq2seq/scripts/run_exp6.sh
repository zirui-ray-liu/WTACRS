level=2
sample_ratio=0.1
model="t5-3b"
# model="bert-large-cased"
# model="t5-large"
# model="t5-base"
seed_buf=(9)
dataset_buf=("rte")
batchsize_buf=(100)

# for seed in ${seed_buf[@]}
# do
# bash scripts/approx_linear.sh 0 rte $model $level $sample_ratio $seed
# done

# for seed in ${seed_buf[@]}
# do
# bash scripts/lora_approx_linear.sh 0 cola $model $level $sample_ratio $seed
# done

for seed in ${seed_buf[@]}
do
    for dataset in ${dataset_buf[@]}
    do
        for batchsize in ${batchsize_buf[@]}
        do
            bash scripts/lora_approx_linear_4gpu.sh 0 $dataset $model $level $sample_ratio $seed 32 $batchsize
            # bash scripts/approx_linear_4gpu.sh 0 $dataset $model $level $sample_ratio $seed $batchsize      
        done
    done
done