#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 7-0:0:0

#SBATCH --gres=gpu:a6000:1
#SBATCH --job-name=test_model
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G

#SBATCH -o /home/tahboub.h/logs/%j.log
#SBATCH -e /home/tahboub.h/logs/%j.err

module load cuda/12.1
source "/shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh"
conda activate color
cd ~/image-colorization
srun python ddcolor/train.py
