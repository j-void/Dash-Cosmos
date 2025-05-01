#!/bin/bash

#SBATCH --time=70:00:00 # walltime
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=16G
#SBATCH --account=lonepeak-gpu
#SBATCH --partition=lonepeak-gpu
#SBATCH --gres=gpu:1080ti:1
#SBATCH -o runs/ooc_basic_synth.out # Standard output 
#SBATCH -e runs/ooc_basic_synth.err # Standard error
#SBATCH --open-mode=append


~/miniconda3/envs/dash/bin/python train_ooc_synth.py
