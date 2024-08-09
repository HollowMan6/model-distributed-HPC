#!/bin/bash
#SBATCH --time=5-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu-h100-80g
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:4

cd /scratch/work/jiangs5
source venv/bin/activate

module load model-huggingface/all
# this will set HF_HOME to /scratch/shareddata/dldata/huggingface-hub-cache

export TRANSFORMERS_OFFLINE=1

clean_up_logs() {
  while true; do
    rm -rf /tmp/$SLURM_JOB_USER/$SLURM_JOB_ID/ray/*/logs/* || true
    sleep 1800
  done
}

# Function to check if a port is open
wait_for_port() {
    local host=$1
    local port=$2
    while ! nc -z $host $port; do
        echo "Waiting for $host:$port to be open..."
        sleep 1
    done
}

port_forwarding() {
  # # Wait for ports 1234 and 8000 to be open
  # wait_for_port 127.0.0.1 1234
  wait_for_port 127.0.0.1 8000

  # Execute the SSH command
  ssh -fN -R 33333:127.0.0.1:8000 mine
}

port_forwarding &
clean_up_logs &

# --disable-log-requests
# tensor-parallel-size should match requested GPU number
python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-Large-Instruct-2407 \
  --tensor-parallel-size 4
