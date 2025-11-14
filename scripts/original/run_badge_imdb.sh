#!/bin/bash
#SBATCH --job-name=badge_imdb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/run_badge_imdb/%j.out
#SBATCH --error=logs/run_badge_imdb/%j.err
#SBATCH --partition=h200

set -euo pipefail

source /local/scratch/zyu273/miniconda3/etc/profile.d/conda.sh
conda activate /local/scratch/zyu273/alkd/env
export CUBLAS_WORKSPACE_CONFIG=:4096:8

git pull origin &&

python -u run_badge_agnews.py \
    --dataset imdb \
    --batch_size 64 \
    --epochs 3 \
    --query_size 1000 \
    --rounds 20 \
    --device cuda

