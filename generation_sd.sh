#!/bin/bash

#SBATCH --job-name=test_lyj
#SBATCH --output=test_lyj.out.%j
#SBATCH --error=test_lyj.out.%j
#SBATCH --time=5:00:00
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
                            --init_img /fs/nexus-scratch/yliang17/Research/diffusion/gene_diffcls/data/test/argusianus_argus/8a6c4cac-21bc-11ea-a13a-137349068a90.jpg \
                            --embedding_path logs/argusianus_argus2024-01-11T06-11-59_animal_argus_test/checkpoints/embeddings_gs-3499.pt \
                            --ckpt_path models/ldm/sd_14/sd-v1-4.ckpt \
                            --config configs/stable-diffusion/v1-inference.yaml \
                            --prompt "a photo of *" \
                            --ori_prompt "a photo of great argus in the wild" \
                            --outdir "outputs/sd_img2img-samples_new/${scale_str}_${noise_str}_${o_str}"

}

SCALE=5.0
SCALE_STR="5"

NOISE=0.1
NOISE_STR="10"
O_WEIGHT=1
O_STR="10"
run_generation ${SCALE} ${SCALE_STR} ${O_WEIGHT} ${O_STR} ${NOISE} ${NOISE_STR}

NOISE=0.9
NOISE_STR="90"
O_WEIGHT=1
O_STR="10"
run_generation ${SCALE} ${SCALE_STR} ${O_WEIGHT} ${O_STR} ${NOISE} ${NOISE_STR}


SCALE=3.0
SCALE_STR="3"

NOISE=0.1
NOISE_STR="10"
O_WEIGHT=1
O_STR="10"
run_generation ${SCALE} ${SCALE_STR} ${O_WEIGHT} ${O_STR} ${NOISE} ${NOISE_STR}

NOISE=0.3
NOISE_STR="30"
O_WEIGHT=1
O_STR="10"
run_generation ${SCALE} ${SCALE_STR} ${O_WEIGHT} ${O_STR} ${NOISE} ${NOISE_STR}

NOISE=0.5
NOISE_STR="50"
O_WEIGHT=1
O_STR="10"
run_generation ${SCALE} ${SCALE_STR} ${O_WEIGHT} ${O_STR} ${NOISE} ${NOISE_STR}

NOISE=0.9
NOISE_STR="90"
O_WEIGHT=1
O_STR="10"
run_generation ${SCALE} ${SCALE_STR} ${O_WEIGHT} ${O_STR} ${NOISE} ${NOISE_STR}


SCALE=10.0
SCALE_STR="10"

NOISE=0.1
NOISE_STR="10"
O_WEIGHT=1
O_STR="10"
run_generation ${SCALE} ${SCALE_STR} ${O_WEIGHT} ${O_STR} ${NOISE} ${NOISE_STR}

NOISE=0.3
NOISE_STR="30"
O_WEIGHT=1
O_STR="10"
run_generation ${SCALE} ${SCALE_STR} ${O_WEIGHT} ${O_STR} ${NOISE} ${NOISE_STR}

NOISE=0.5
NOISE_STR="50"
O_WEIGHT=1
O_STR="10"
run_generation ${SCALE} ${SCALE_STR} ${O_WEIGHT} ${O_STR} ${NOISE} ${NOISE_STR}

NOISE=0.9
NOISE_STR="90"
O_WEIGHT=1
O_STR="10"
run_generation ${SCALE} ${SCALE_STR} ${O_WEIGHT} ${O_STR} ${NOISE} ${NOISE_STR}

# NOISE=0.5
# NOISE_STR="50"
# O_WEIGHT=1
# O_STR="10"
# run_generation ${SCALE} ${SCALE_STR} ${O_WEIGHT} ${O_STR} ${NOISE} ${NOISE_STR}
