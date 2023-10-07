declare -A num_epochs
declare -A lr
declare -A sample_ratio
declare -A mix_replacement

num_epochs["rte"]=40
num_epochs["qnli"]=10
num_epochs["mnli"]=10
num_epochs["qqp"]=10
num_epochs["cola"]=20
num_epochs["sst2"]=10
num_epochs["mrpc"]=20
num_epochs["stsb"]=20
num_epochs["superglue-boolq"]=20
num_epochs["superglue-multirc"]=10
num_epochs["superglue-wic"]=20
num_epochs["superglue-cb"]=20
num_epochs["superglue-copa"]=20
num_epochs["superglue-record"]=10

lr["rte"]=3e-5
lr["qnli"]=3e-6
lr["mnli"]=3e-5
lr["qqp"]=3e-5
lr["cola"]=3e-5
lr["sst2"]=3e-5
lr["mrpc"]=3e-5
lr["stsb"]=3e-5

sample_ratio["rte"]=0.5
sample_ratio["qnli"]=0.1
sample_ratio["mnli"]=0.3
sample_ratio["qqp"]=0.3
sample_ratio["cola"]=0.1
sample_ratio["sst2"]=0.1
sample_ratio["mrpc"]=0.1
sample_ratio["stsb"]=0.1



