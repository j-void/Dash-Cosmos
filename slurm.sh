#!/bin/bash

#SBATCH --time=24:00:00 # walltime
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --account=kingspeak-gpu
#SBATCH --partition=kingspeak-gpu
#SBATCH --gres=gpu:titanx:1
#SBATCH -o slurm_log.out # Standard output 
#SBATCH -e slurm_log.err # Standard error


~/miniconda3/envs/dash/bin/python train.py