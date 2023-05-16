# This scripts trains full finetuning method.
# For smaller datasets of GLUE (mrpc, cola, and stsb), we set the `num_train_epochs` to 20,
# for other larger datasets in GLUE we used `num_train_epochs` of 3. 
folder_name=all_output_logs/
if [ ! -d ${folder_name} ] ; then
    mkdir -p ${folder_name}
fi

echo $1 $2 $3 $4 $5 $6 $7

gpuid=$1
dataset=$2
model=$3
level=$4
sample_ratio=$5
seed=$6
deter_ratio=$7


exp_tag=${dataset}_${model}_level${level}_s${sample_ratio}_ablation_fixed_deter_${deter_ratio}

config_file_name=configs/approx_linear.json
update_file_name=configs/approx_linear/approx_linear_${exp_tag}.json

source scripts/env_approx.sh
python scripts/update_scripts_for_given_input.py $config_file_name "" $update_file_name

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

# Hyper-parameter for Approx
python scripts/update_scripts_for_given_input.py $update_file_name apply_sampling bool true $update_file_name
python scripts/update_scripts_for_given_input.py $update_file_name deter_adaptive bool false $update_file_name
python scripts/update_scripts_for_given_input.py $update_file_name sample_replacement bool true $update_file_name
python scripts/update_scripts_for_given_input.py $update_file_name mix_replacement bool true $update_file_name
python scripts/update_scripts_for_given_input.py $update_file_name k_sampling    int 0 $update_file_name
python scripts/update_scripts_for_given_input.py $update_file_name q_sampling    int 0 $update_file_name
python scripts/update_scripts_for_given_input.py $update_file_name v_sampling    int 0 $update_file_name
python scripts/update_scripts_for_given_input.py $update_file_name o_sampling    int 1 $update_file_name
python scripts/update_scripts_for_given_input.py $update_file_name wi_0_sampling int 1 $update_file_name
python scripts/update_scripts_for_given_input.py $update_file_name wi_1_sampling int 1 $update_file_name
python scripts/update_scripts_for_given_input.py $update_file_name wo_sampling   int 1 $update_file_name
python scripts/update_scripts_for_given_input.py $update_file_name score_sampling   int 1 $update_file_name
python scripts/update_scripts_for_given_input.py $update_file_name attout_sampling   int 1 $update_file_name
python scripts/update_scripts_for_given_input.py $update_file_name softmax_prune_ratio   float 0 $update_file_name
python scripts/update_scripts_for_given_input.py $update_file_name inplace_layernorm   bool true $update_file_name
python scripts/update_scripts_for_given_input.py $update_file_name sampling_ratio float $sample_ratio $update_file_name
python scripts/update_scripts_for_given_input.py $update_file_name deter_ratio float $deter_ratio $update_file_name

# Run Experiment
python scripts/update_scripts_for_given_input.py $update_file_name output_dir  str outputs/full_finetuning_${exp_tag}_sd${seed} $update_file_name

CUDA_VISIBLE_DEVICES=$1 python run_seq2seq.py  $update_file_name

