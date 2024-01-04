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

run_generation() {
    local scale=$1
    local scale_str=$2
    local o_weight=$3
    local o_str=$4
    local noise=$5
    local noise_str=$6

    echo "run generation with ${scale} - ${o_weight} - ${noise}"

    python scripts/txt2img.py --ddim_eta 0.0 \
                            --n_samples 4 \
                            --n_iter 2 \
                            --scale ${scale} \
                            --strength ${noise} \
                            --ddim_steps 50 \
                            --o_weight ${o_weight} \
                            --W 512 \
                            --H 320 \
                            --C 4 \
                            --f 8 \
                            --init_img /fs/nexus-scratch/yliang17/Research/diffusion/gene_diffcls/output/tayassu_pecari/gpt_camera_03/8fa67602-21bc-11ea-a13a-137349068a90.jpg \
                            --embedding_path logs/Tayassu_pecari2024-01-04T07-28-05_animal_test/checkpoints/embeddings_gs-6099.pt \
                            --ckpt_path models/ldm/sd_14/sd-v1-4.ckpt \
                            --config configs/stable-diffusion/v1-inference.yaml \
                            --prompt "a photo of *" \
                            --ori_prompt "white-lipped peccary in the wild" \
                            --outdir "outputs/sd_img2img-samples/${scale_str}_${noise_str}_${o_str}"

}

SCALE=10.0
SCALE_STR="10"
NOISE=0.3
NOISE_STR="30"

O_WEIGHT=0.1
O_STR="01"
run_generation ${SCALE} ${SCALE_STR} ${O_WEIGHT} ${O_STR} ${NOISE} ${NOISE_STR}


SCALE=5.0
SCALE_STR="5"

O_WEIGHT=0
O_STR="00"
run_generation ${SCALE} ${SCALE_STR} ${O_WEIGHT} ${O_STR} ${NOISE} ${NOISE_STR}

O_WEIGHT=0.1
O_STR="01"
run_generation ${SCALE} ${SCALE_STR} ${O_WEIGHT} ${O_STR} ${NOISE} ${NOISE_STR}

O_WEIGHT=0.3
O_STR="03"
run_generation ${SCALE} ${SCALE_STR} ${O_WEIGHT} ${O_STR} ${NOISE} ${NOISE_STR}

O_WEIGHT=0.5
O_STR="05"
run_generation ${SCALE} ${SCALE_STR} ${O_WEIGHT} ${O_STR} ${NOISE} ${NOISE_STR}

O_WEIGHT=0.8
O_STR="08"
run_generation ${SCALE} ${SCALE_STR} ${O_WEIGHT} ${O_STR} ${NOISE} ${NOISE_STR}
