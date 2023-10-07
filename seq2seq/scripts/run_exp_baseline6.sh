


model="t5-large"

seed_buf=(0 1 2)

for seed in ${seed_buf[@]}
do
bash scripts/lora_baseline.sh 0 qqp $model $seed
done