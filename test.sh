#!/bin/bash

#SBATCH --job-name=test_lyj
#SBATCH --output=test_lyj.out.%j
#SBATCH --error=test_lyj.out.%j
#SBATCH --time=1:00:00
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

python main.py --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml \
               -t \
               --actual_resume models/ldm/text2img-large/model.ckpt \
               -n argusianus_argus \
               --gpus "0," \
               --data_root "../data/train/argusianus_argus/" \
               --init_word argus