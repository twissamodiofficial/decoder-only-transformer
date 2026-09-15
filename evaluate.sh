#!/bin/bash
#SBATCH --job-name=gpt_clone_evaluate
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err
#SBATCH --partition=MGPU-TC2           # REQUIRED - specify partition
#SBATCH --qos=normal                   # REQUIRED - specify QoS
#SBATCH --nodes=1                      # REQUIRED
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=01:00:00

# Load modules
module load anaconda/25.5.1

# Activate environment
eval "$(conda shell.bash hook)"
conda activate gpt-clone   # replace with your actual environment name if different

# Run test-set evaluation for all three attention variants
python -u -m modules.training.evaluate

echo "Job completed for modules.training.evaluate"