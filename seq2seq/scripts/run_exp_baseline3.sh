

model="t5-base"
# bash scripts/baseline.sh 2 cola $model

seed_buf=(0 1 2)

for seed in ${seed_buf[@]}
do
bash scripts/lora_baseline.sh 4 cola $model $seed
done
