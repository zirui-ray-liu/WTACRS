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

exp_tag=${dataset}_${model}_LST

config_file_name=configs/side_transformers.json
update_file_name=configs/baseline/lst_baseline_${exp_tag}.json

source scripts/env.sh
python scripts/update_scripts_for_given_input.py $config_file_name "" $update_file_name

r=8
lr=3e-4

encoder_side_layers="0,1,2,3,4,5,6,7,9,11,13,15,16,17,18,19,20,21,22,23" # "0,1,2,3,5,6,7,9,10,11"

# Hyper-parameter for Setting
python scripts/update_scripts_for_given_input.py $update_file_name model_name_or_path str $model $update_file_name
python scripts/update_scripts_for_given_input.py $update_file_name tokenizer_name str $model $update_file_name
python scripts/update_scripts_for_given_input.py $update_file_name task_name str $dataset $update_file_name
python scripts/update_scripts_for_given_input.py $update_file_name eval_dataset_name str $dataset $update_file_name
python scripts/update_scripts_for_given_input.py $update_file_name test_dataset_name str $dataset $update_file_name
python scripts/update_scripts_for_given_input.py $update_file_name split_validation_test bool false $update_file_name
python scripts/update_scripts_for_given_input.py $update_file_name pad_to_max_length bool true $update_file_name

# Hyper-parameter for Training
python scripts/update_scripts_for_given_input.py $update_file_name seed int $seed $update_file_name

# Hyper-parameter for LST
python scripts/update_scripts_for_given_input.py $update_file_name use_gate str "learnable" $update_file_name
python scripts/update_scripts_for_given_input.py $update_file_name task_reduction_factor int ${r} $update_file_name
python scripts/update_scripts_for_given_input.py $update_file_name load_side_pretrained_weights str "" $update_file_name # fisher-v2
python scripts/update_scripts_for_given_input.py $update_file_name learning_rate float ${lr} $update_file_name
python scripts/update_scripts_for_given_input.py $update_file_name num_train_epochs int ${num_epochs[$2]} $update_file_name # 
python scripts/update_scripts_for_given_input.py $update_file_name add_bias_sampling bool true $update_file_name
python scripts/update_scripts_for_given_input.py $update_file_name create_side_lm bool false $update_file_name
python scripts/update_scripts_for_given_input.py $update_file_name freeze_side_lm bool false $update_file_name
python scripts/update_scripts_for_given_input.py $update_file_name add_residual_after bool false $update_file_name
python scripts/update_scripts_for_given_input.py $update_file_name encoder_side_layers list $encoder_side_layers $update_file_name
python scripts/update_scripts_for_given_input.py $update_file_name decoder_side_layers list $encoder_side_layers $update_file_name

python scripts/update_scripts_for_given_input.py $update_file_name output_dir str outputs/lst_${exp_tag}_r${r}_sd${seed} $update_file_name

CUDA_VISIBLE_DEVICES=$gpuid python run_seq2seq.py  $update_file_name

