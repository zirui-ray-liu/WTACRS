level=2
sample_ratio=0.3
model="t5-large"
seed_buf=(0 1 2)

# for seed in ${seed_buf[@]}
# do
# bash scripts/approx_linear.sh 0 cola $model $level $sample_ratio $seed
# done

for seed in ${seed_buf[@]}
do
echo $seed
# bash scripts/lora_approx_linear.sh 0 sst2 $model $level $sample_ratio $seed
done

