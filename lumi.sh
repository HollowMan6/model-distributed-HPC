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
module load pytorch

export NCCL_DEBUG=INFO
export TORCH_CPP_LOG_LEVEL=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

MASTER_ADDR=`perl -le '$_=$ENV{"SLURM_JOB_NODELIST"}; s/,.*//; s/-.*//; s/\[//; print'`
MASTER_PORT=6777

srun torchrun --nnodes=2 \
    --nproc_per_node=8 \
    --node_rank $SLURM_PROCID \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --tee 3 \
    --log_dir ./logs \
    primitives.py
 