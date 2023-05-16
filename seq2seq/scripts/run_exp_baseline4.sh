


model="t5-base"
# bash scripts/baseline.sh 3 stsb $model

seed_buf=(0 1 2)

for seed in ${seed_buf[@]}
do
bash scripts/lora_baseline.sh 6 sst2 $model $seed
done