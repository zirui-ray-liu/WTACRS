#!/bin/bash
#SBATCH -A vxc204_aisc
#SBATCH -p aisc
#SBATCH --gpus=1
#SBATCH -c 16
#SBATCH --mem=128gb
#SBATCH --time=48:00:00
module load Python/3.8.6-GCCcore-10.2.0
module load CUDA/11.7.0
source /mnt/rstor/CSE_CSDS_VXC204/sxz517/venv_vault/lm/bin/activate
bash /mnt/rstor/CSE_CSDS_VXC204/sxz517/FT_ApproxMatmul/seq2seq/scripts/run_exp.sh