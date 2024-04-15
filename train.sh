#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 3-0:0:0

#SBATCH --gres=gpu:a6000:1
#SBATCH --job-name=test_model
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G

#SBATCH -o /home/tahboub.h/logs/%j.log
#SBATCH -e /home/tahboub.h/logs/%j.err

module load anaconda3/2022.05
conda activate color
python ddcolor/train.py
