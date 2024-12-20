#!/bin/bash -l
#Configuration Options
#SBATCH --account=defake
#SBATCH --partition=tier3
#SBATCH --job-name=ucf_train_with_noise_final
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-user=slack:@dp7144
#SBATCH --mail-type=ALL
#SBATCH --time=0-15:00:00
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1

#Load Software
spack env activate default-ml-x86_64-24071101

#Your code
python training/train.py --detector_path ./training/config/detector/ucf.yaml

