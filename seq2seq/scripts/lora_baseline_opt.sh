# This scripts trains Adapters method.
# For smaller datasets of GLUE (mrpc, cola, and stsb), we set the `num_train_epochs` to 20,
# for other larger datasets in GLUE we used `num_train_epochs` of 3. For all datasets we tried
# with the adapter's bottleneck size of `task_reduction_factor`=[32, 16, 8], and report the 
# results on the test set for the model performing the best on the validation set.

folder_name=all_output_logs/
if [ ! -d ${folder_name} ] ; then
    mkdir -p ${folder_name}
fi

echo $1 $2 $3 $4

gpuid=$1
dataset=$2
model=$3
seed=$4

exp_tag=${dataset}_${model/\//-}_LoRA

config_file_name=configs/lora.json
update_file_name=configs/baseline/lora_baseline_${exp_tag}.json

source scripts/env_lora_opt.sh
python scripts/update_scripts_for_given_input.py $config_file_name "" $update_file_name

lora_dim=32

# Hyper-parameter for Setting
python scripts/update_scripts_for_given_input.py $update_file_name task_name str $dataset $update_file_name
python scripts/update_scripts_for_given_input.py $update_file_name eval_dataset_name str $dataset $update_file_name
python scripts/update_scripts_for_given_input.py $update_file_name test_dataset_name str $dataset $update_file_name
python scripts/update_scripts_for_given_input.py $update_file_name split_validation_test bool false $update_file_name
python scripts/update_scripts_for_given_input.py $update_file_name pad_to_max_length bool true $update_file_name

# Hyper-parameter for Training
python scripts/update_scripts_for_given_input.py $update_file_name model_name_or_path str $model $update_file_name
python scripts/update_scripts_for_given_input.py $update_file_name tokenizer_name str $model $update_file_name
python scripts/update_scripts_for_given_input.py $update_file_name learning_rate float ${lr[$dataset]} $update_file_name
python scripts/update_scripts_for_given_input.py $update_file_name num_train_epochs int ${num_epochs[$dataset]} $update_file_name
python scripts/update_scripts_for_given_input.py $update_file_name seed int $seed $update_file_name
python scripts/update_scripts_for_given_input.py $update_file_name per_device_train_batch_size int 100 $update_file_name

python scripts/update_scripts_for_given_input.py $update_file_name task_adapter_layers_encoder eval None $update_file_name
python scripts/update_scripts_for_given_input.py $update_file_name trainable_encoder_layers eval None $update_file_name
python scripts/update_scripts_for_given_input.py $update_file_name task_adapter_layers_decoder eval None $update_file_name
python scripts/update_scripts_for_given_input.py $update_file_name trainable_decoder_layers eval None

# Hyper-parameter for LoRA
python scripts/update_scripts_for_given_input.py $update_file_name lora_dim int ${lora_dim}

# Run Experiment
python scripts/update_scripts_for_given_input.py $update_file_name output_dir  str outputs/full_finetuning_${exp_tag}_sd${seed} $update_file_name

CUDA_VISIBLE_DEVICES=$gpuid python opt_run_glue.py  $update_file_name


