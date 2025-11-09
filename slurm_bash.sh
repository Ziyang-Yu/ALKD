#!/bin/bash
#SBATCH --job-name=badge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/badge_%j.out
#SBATCH --error=logs/badge_%j.err
#SBATCH --partition=h200


set -euo pipefail

source /local/scratch/zyu273/miniconda3/etc/profile.d/conda.sh
# conda create -p /local/scratch/zyu273/alkd/env python=3.10 -y
conda activate /local/scratch/zyu273/alkd/env
# pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# pip install -U scikit-learn
# pip install numpy pandas tqdm
# pip install openml
# python run.py --alg badge --did 0 --lr 0.001  --model resnet --data data --nQuery 10000 --nStart 10000 --nEnd 50000 --nEmb 256 --trunc -1 --aug 1 --dummy 1 --data CIFAR10 --lr 0.001 

# python run.py --alg badge --did 0 --lr 0.001  --model resnet --data data --nQuery 10000 --nStart 10000 --nEnd 50000 --nEmb 256 --trunc -1 --aug 1 --dummy 1 --data CIFAR10 --lr 0.001 
# python run.py --model mlp --nQuery 10000 --did 6 --alg bait
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -u run_badge_agnews.py --batch_size 64 --epochs 3 --query_size 1000 --rounds 20 --device cuda
