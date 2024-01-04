#!/bin/bash

#SBATCH --job-name=test_lyj
#SBATCH --output=test_lyj.out.%j
#SBATCH --error=test_lyj.out.%j
#SBATCH --time=3:00:00
#SBATCH --account=cml-zhou
#SBATCH --partition=cml-dpart
#SBATCH --qos=cml-medium
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G


# checking gpu status
nvidia-smi

# cd ../..
source /fs/nexus-scratch/yliang17/miniconda3/bin/activate ldm
# echo $CONDA_DEFAULT_ENV

python main.py --base configs/stable-diffusion/v1-finetune.yaml \
               -t \
               --actual_resume models/ldm/sd_14/sd-v1-4.ckpt \
               -n animal_test \
               --gpus "0," \
               --data_root "data/Tayassu_pecari/" \
               --init_word animal