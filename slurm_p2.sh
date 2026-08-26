#!/bin/bash
#SBATCH --job-name=p2_llm_reduction
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --output=logs/p2_%j.out
#SBATCH --error=logs/p2_%j.err

# Protocol 2: LLM-Assisted Topic Reduction
# This script runs the vLLM server and Protocol 2 experiments

echo "=========================================="
echo "Protocol 2: LLM-Assisted Topic Reduction"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

source .env && export HF_TOKEN

# Load modules (adjust to your OSCAR setup)
module load python/3.10
module load cuda/12.1

# Activate virtual environment
source ~/venv/bin/activate

# Set environment variables
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=0

# Create logs directory if it doesn't exist
mkdir -p logs
MODEL_ID = "google/gemma-3-12b-it"

echo "Starting vLLM server..."
echo "Model: $MODEL_ID"

# Start vLLM server in background
vllm serve $MODEL_ID \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.8 &

VLLM_PID=$!
echo "vLLM server started (PID: $VLLM_PID)"

# Wait for vLLM server to load model (60 seconds should be enough)
echo "Waiting for model to load (60 seconds)..."
sleep 60

# Verify server is running
if ! ps -p $VLLM_PID > /dev/null; then
    echo "ERROR: vLLM server failed to start"
    exit 1
fi

echo ""
echo "=========================================="
echo "Running Protocol 2 Experiments"
echo "=========================================="
echo ""

# Run Protocol 2 with 20 Newsgroups
echo "Dataset: 20 Newsgroups"
python experiments/run_p2_with_llm.py --config experiments/configs/p2_20NG.yaml

# Optionally run with AG News (comment out if not needed)
# echo ""
# echo "Dataset: AG News"
# python experiments/run_p2_with_llm.py --config experiments/configs/p2_AG.yaml

echo ""
echo "=========================================="
echo "Experiments Complete"
echo "=========================================="

# Cleanup: Kill vLLM server
echo "Shutting down vLLM server..."
kill $VLLM_PID
wait $VLLM_PID 2>/dev/null

echo "Job finished successfully"