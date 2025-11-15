#!/bin/bash
#SBATCH --job-name=badge_yelp_llm_cbm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/run_badge_yelp_llm_cbm/%j.out
#SBATCH --error=logs/run_badge_yelp_llm_cbm/%j.err
#SBATCH --partition=c64-m512*

set -euo pipefail

source /local/scratch/zyu273/miniconda3/etc/profile.d/conda.sh &&
conda activate /local/scratch/zyu273/alkd/env &&
export CUBLAS_WORKSPACE_CONFIG=:4096:8 &&

git pull origin &&

python -u run_badge_llm_cbm.py \
    --dataset yelp_polarity \
    --model_name FacebookAI/roberta-base \
    --max_length 128 \
    --seed_size 50 \
    --query_size 5 \
    --rounds 100 \
    --epochs 5 \
    --batch_size 16 \
    --lr 0.1 \
    --alpha_concept 1.0

