#!/bin/bash

#SBATCH --time=70:00:00 # walltime
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --account=lonepeak-gpu
#SBATCH --partition=lonepeak-gpu
#SBATCH --gres=gpu:1080ti:1
#SBATCH -o runs/ooc_basic_vit_base.out # Standard output 
#SBATCH -e runs/ooc_basic_vit_base.err # Standard error


~/miniconda3/envs/dash/bin/python train_ooc.py