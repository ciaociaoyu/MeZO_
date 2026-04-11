#!/bin/bash
#SBATCH --job-name=sst5_opt1p3b_100
#SBATCH --partition=gpu_p
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=120G
#SBATCH --gres=gpu:H100:1
#SBATCH --time=04:00:00
#SBATCH --output=jobs/%x_%j.out

ml jq

source ~/.bashrc
conda activate mezo-env

ROOT_DIR=/scratch/jy03364/MeZO_
cd "${ROOT_DIR}/large_models"

MODEL=opt-1.3b \
TASK=SST5 \
MODE=ft \
SEED=0 \
DATA_SEED=0 \
DATASET_MODE=fewshot \
K=16 \
BS=1 \
LR=1e-6 \
EPS=1e-3 \
EVAL=128 \
STEPS=100 \
EVAL_STEPS=25 \
bash mezo.sh \
  --no_auto_device \
  --gradient_accumulation_steps 8 \
  --logging_steps 1
