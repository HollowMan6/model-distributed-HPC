#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -p dev-g
#SBATCH -t 3:00:00
#SBATCH --gpus-per-node=mi250:8
#SBATCH --account=project_462000007

module purge
module use /appl/local/csc/modulefiles
module load tensorflow

export NCCL_DEBUG=INFO

cd tensorflow-train
srun python3 main.py
