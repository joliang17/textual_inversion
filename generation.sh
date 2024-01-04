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

python scripts/txt2img.py --ddim_eta 0.0 \
                          --n_samples 4 \
                          --n_iter 2 \
                          --scale 10.0 \
                          --strength 0.5 \
                          --ddim_steps 50 \
                          --o_weight 0.1 \
                          --init_img /fs/nexus-scratch/yliang17/Research/diffusion/gene_diffcls/output/tayassu_pecari/gpt_camera_03/8fa67602-21bc-11ea-a13a-137349068a90.jpg \
                          --embedding_path logs/Tayassu_pecari2024-01-03T04-53-48_animal_test/checkpoints/embeddings_gs-6099.pt \
                          --ckpt_path models/ldm/text2img-large/model.ckpt \
                          --prompt "a photo of *" \
                          --outdir "outputs/img2img-samples/10_50_10"

python scripts/txt2img.py --ddim_eta 0.0 \
                          --n_samples 4 \
                          --n_iter 2 \
                          --scale 5.0 \
                          --strength 0.5 \
                          --ddim_steps 50 \
                          --o_weight 0.5 \
                          --init_img /fs/nexus-scratch/yliang17/Research/diffusion/gene_diffcls/output/tayassu_pecari/gpt_camera_03/8fa67602-21bc-11ea-a13a-137349068a90.jpg \
                          --embedding_path logs/Tayassu_pecari2024-01-03T04-53-48_animal_test/checkpoints/embeddings_gs-6099.pt \
                          --ckpt_path models/ldm/text2img-large/model.ckpt \
                          --prompt "a photo of *" \
                          --outdir "outputs/img2img-samples/5_50_50"

python scripts/txt2img.py --ddim_eta 0.0 \
                          --n_samples 4 \
                          --n_iter 2 \
                          --scale 5.0 \
                          --strength 0.5 \
                          --ddim_steps 50 \
                          --o_weight 0.8 \
                          --init_img /fs/nexus-scratch/yliang17/Research/diffusion/gene_diffcls/output/tayassu_pecari/gpt_camera_03/8fa67602-21bc-11ea-a13a-137349068a90.jpg \
                          --embedding_path logs/Tayassu_pecari2024-01-03T04-53-48_animal_test/checkpoints/embeddings_gs-6099.pt \
                          --ckpt_path models/ldm/text2img-large/model.ckpt \
                          --prompt "a photo of *" \
                          --outdir "outputs/img2img-samples/5_50_80"



