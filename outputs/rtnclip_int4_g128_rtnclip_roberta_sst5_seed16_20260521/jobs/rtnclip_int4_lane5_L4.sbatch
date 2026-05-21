#!/usr/bin/env bash
#SBATCH --job-name=rtnclip-i4-l5-L4
#SBATCH --output=/scratch/jy03364/MeZO_/outputs/rtnclip_int4_g128_rtnclip_roberta_sst5_seed16_20260521/jobs/%x_%j.out
#SBATCH --error=/scratch/jy03364/MeZO_/outputs/rtnclip_int4_g128_rtnclip_roberta_sst5_seed16_20260521/jobs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:L4:1
set -euo pipefail
cd "/scratch/jy03364/MeZO_"
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "ciao"
fi
export DATALOADER_SHUFFLE=True
export TOKENIZERS_PARALLELISM=false
export REQUESTED_GPU_TYPE="L4"
export FALLBACK_USED="1"
python tools/rtnclip_roberta_sst5_batch.py \
  --output_root "/scratch/jy03364/MeZO_/outputs/rtnclip_int4_g128_rtnclip_roberta_sst5_seed16_20260521" \
  --manifest "/scratch/jy03364/MeZO_/outputs/rtnclip_int4_g128_rtnclip_roberta_sst5_seed16_20260521/lane_manifests/lane5.csv" \
  --eval_every 1000 \
  --checkpoint_steps 1000 \
  --eval_batch_size 64 \
  --eval_batches -1 \
  --diag_every 100 \
  --quant_log_every 1000 \
  --log_every 100 \
  run-manifest
