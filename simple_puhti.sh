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
module load pytorch

export NCCL_DEBUG=INFO

MASTER_ADDR=`perl -le '$_=$ENV{"SLURM_JOB_NODELIST"}; s/,.*//; s/-.*//; s/\[//; print'`
MASTER_PORT=6777

srun torchrun --nnodes=2 \
    --nproc_per_node=4 \
    --node_rank $SLURM_PROCID \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --tee 3 \
    --log_dir ./logs \
    simple.py
 