level=$1
update_file_name=$2

config_file_name=configs/approx_linear.json

if [ $level == 0 ]
then
    python scripts/update_scripts_for_given_input.py $update_file_name apply_sampling bool false $update_file_name
    
elif [ $level == 1 ]
then
    python scripts/update_scripts_for_given_input.py $update_file_name apply_sampling bool true $update_file_name
    python scripts/update_scripts_for_given_input.py $update_file_name k_sampling    int 0 $update_file_name
    python scripts/update_scripts_for_given_input.py $update_file_name q_sampling    int 0 $update_file_name
    python scripts/update_scripts_for_given_input.py $update_file_name v_sampling    int 0 $update_file_name
    python scripts/update_scripts_for_given_input.py $update_file_name o_sampling    int 0 $update_file_name
    python scripts/update_scripts_for_given_input.py $update_file_name wi_0_sampling int 0 $update_file_name
    python scripts/update_scripts_for_given_input.py $update_file_name wi_1_sampling int 0 $update_file_name
    python scripts/update_scripts_for_given_input.py $update_file_name wo_sampling   int 0 $update_file_name
    python scripts/update_scripts_for_given_input.py $update_file_name score_sampling   int 0 $update_file_name
    python scripts/update_scripts_for_given_input.py $update_file_name attout_sampling   int 0 $update_file_name
    python scripts/update_scripts_for_given_input.py $update_file_name softmax_prune_ratio   float 0 $update_file_name
    python scripts/update_scripts_for_given_input.py $update_file_name inplace_layernorm   bool true $update_file_name
    
elif [ $level == 2 ]
then
    python scripts/update_scripts_for_given_input.py $update_file_name apply_sampling bool true $update_file_name
    python scripts/update_scripts_for_given_input.py $update_file_name deter_adaptive bool true $update_file_name
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
    
else
   echo "No Implementation for Such Level."
fi


