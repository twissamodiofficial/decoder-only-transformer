#!/bin/bash
#SBATCH --job-name=gpt_clone_train
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err
#SBATCH --partition=MGPU-TC2           # REQUIRED - specify partition
#SBATCH --qos=normal                   # REQUIRED - specify QoS
#SBATCH --nodes=1                      # REQUIRED
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=06:00:00

# Load modules
module load anaconda/25.5.1
# module load cuda/12.8.0   # not required - pip-installed PyTorch bundles its own CUDA runtime

# Activate environment
eval "$(conda shell.bash hook)"
conda activate gpt-clone   # replace with your actual environment name if different

# Run training
LATEST_CKPT=$(ls -v modules/checkpoints/epoch_*.pt 2>/dev/null | tail -n 1)

if [ -n "$LATEST_CKPT" ]; then
    echo "Resuming from $LATEST_CKPT"
    python -u -m modules.training.training --attention mha --resume "$LATEST_CKPT"
else
    echo "No checkpoint found, starting fresh"
    python -u -m modules.training.training --attention mha
fi

echo "Job completed for modules.training.training"