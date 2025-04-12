#!/bin/bash

#SBATCH --time=24:00:00 # walltime
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --account=kingspeak-gpu
#SBATCH --partition=kingspeak-gpu
#SBATCH --gres=gpu:titanx:1
#SBATCH -o runs/slurm_log.out # Standard output 
#SBATCH -e runs/slurm_log.err # Standard error


~/miniconda3/envs/dash/bin/python train.py