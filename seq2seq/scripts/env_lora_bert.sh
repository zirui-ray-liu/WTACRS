declare -A num_epochs
declare -A lr
declare -A batch_size
declare -A metric_for_best_model

num_epochs["rte"]=20
num_epochs["qnli"]=20
num_epochs["mnli"]=10
num_epochs["qqp"]=10
num_epochs["cola"]=40
num_epochs["sst2"]=20
num_epochs["mrpc"]=20
num_epochs["stsb"]=10

lr["rte"]=2e-4
lr["qnli"]=2e-4
lr["mnli"]=2e-4
lr["qqp"]=2e-4
lr["cola"]=3e-4
lr["sst2"]=2e-4
lr["mrpc"]=2e-4
lr["stsb"]=2e-4

batch_size["rte"]=128
batch_size["qnli"]=128
batch_size["mnli"]=128
batch_size["qqp"]=128
batch_size["cola"]=128
batch_size["sst2"]=128
batch_size["mrpc"]=128
batch_size["stsb"]=16

metric_for_best_model["rte"]="accuracy"
metric_for_best_model["qnli"]='accuracy'
metric_for_best_model["mnli"]='accuracy'
metric_for_best_model["qqp"]='f1'
metric_for_best_model["cola"]="matthews_correlation"
metric_for_best_model["sst2"]="accuracy"
metric_for_best_model["mrpc"]="f1"
metric_for_best_model["stsb"]="pearson"




