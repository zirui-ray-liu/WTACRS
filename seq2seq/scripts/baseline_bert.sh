# This scripts trains full finetuning method.
# For smaller datasets of GLUE (mrpc, cola, and stsb), we set the `num_train_epochs` to 20,
# for other larger datasets in GLUE we used `num_train_epochs` of 3. 
folder_name=all_output_logs/
if [ ! -d ${folder_name} ] ; then
    mkdir -p ${folder_name}
fi

echo $1 $2 $3 $4

gpuid=$1
dataset=$2
model=$3
seed=$4

exp_tag=${dataset}_${model}_fp

config_file_name=configs/baseline_bert.json
update_file_name=configs/baseline/baseline_${exp_tag}.json

source scripts/env_baseline_bert.sh
python scripts/update_scripts_for_given_input.py $config_file_name "" $update_file_name

# Hyper-parameter for Setting
python scripts/update_scripts_for_given_input.py $update_file_name task_name str $dataset $update_file_name
python scripts/update_scripts_for_given_input.py $update_file_name eval_dataset_name str $dataset $update_file_name
python scripts/update_scripts_for_given_input.py $update_file_name test_dataset_name str $dataset $update_file_name
python scripts/update_scripts_for_given_input.py $update_file_name split_validation_test bool false $update_file_name
python scripts/update_scripts_for_given_input.py $update_file_name pad_to_max_length bool true $update_file_name
python scripts/update_scripts_for_given_input.py $update_file_name metric_for_best_model str ${metric_for_best_model[$dataset]} $update_file_name
python scripts/update_scripts_for_given_input.py $update_file_name load_best_model_at_end bool true $update_file_name

# Hyper-parameter for Training
python scripts/update_scripts_for_given_input.py $update_file_name model_name_or_path str $model $update_file_name
python scripts/update_scripts_for_given_input.py $update_file_name tokenizer_name str $model $update_file_name
python scripts/update_scripts_for_given_input.py $update_file_name learning_rate float ${lr[$dataset]} $update_file_name
python scripts/update_scripts_for_given_input.py $update_file_name num_train_epochs int ${num_epochs[$dataset]} $update_file_name
python scripts/update_scripts_for_given_input.py $update_file_name seed int $seed $update_file_name
python scripts/update_scripts_for_given_input.py $update_file_name per_device_train_batch_size int ${batch_size[$dataset]} $update_file_name
python scripts/update_scripts_for_given_input.py $update_file_name warmup_steps int 0 $update_file_name

# Run Experiment
python scripts/update_scripts_for_given_input.py $update_file_name output_dir   str outputs/full_finetuning_${exp_tag}_sd${seed} $update_file_name

CUDA_VISIBLE_DEVICES=$gpuid python run_glue.py  $update_file_name