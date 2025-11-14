#!/bin/bash
#SBATCH --job-name=badge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/run_badge_agnews/%j.out
#SBATCH --error=logs/run_badge_agnews/%j.err
#SBATCH --partition=h200

set -euo pipefail

source /local/scratch/zyu273/miniconda3/etc/profile.d/conda.sh
conda activate /local/scratch/zyu273/alkd/env
export CUBLAS_WORKSPACE_CONFIG=:4096:8

git pull origin &&


python -u run_badge_agnews.py --dataset ag_news --batch_size 64 --epochs 3 --query_size 1000 --rounds 20 --device cuda
# python -u run_badge_agnews_llm.py --model_name FacebookAI/roberta-base --max_length 128 --seed_size 50 --query_size 5 --rounds 100 --epochs 5 --batch_size 16 --lr 0.1
# python -u run_badge_agnews_llm_cbm.py --model_name FacebookAI/roberta-base --max_length 128 --seed_size 50 --query_size 5 --rounds 100 --epochs 5 --batch_size 16 --lr 0.1 --alpha_concept 0 --concept_l1 0
