level=2
sample_ratio=0.3
model="t5-base"

# bash scripts/approx_linear.sh 2 cola $model $level $sample_ratio

seed_buf=(0 1 2)

for seed in ${seed_buf[@]}
do
bash scripts/lora_approx_linear.sh 6 cola $model $level $sample_ratio $seed 1 1 1
done


