#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -p gputest
#SBATCH -t 00:15:00
#SBATCH --gpus-per-node=v100:4
#SBATCH --account=project_2001659

module purge
module load tensorflow

export NCCL_DEBUG=INFO

cd tensorflow-train
srun python3 main.py
