#!/bin/bash
#SBATCH --job-name=NemaSeg_Nextflow_test
#SBATCH --partition=a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/eande106/ZihaoJohnLi/datasets/Nextflow_test/Nextflow_test_%j.out
#SBATCH --error=/scratch/eande106/ZihaoJohnLi/datasets/Nextflow_test/Nextflow_test_%j.err

# --- 1. Load Software ---
module purge
module load gcc/9.3.0
module load anaconda3/2024.02-1
module load cuda/13.0.2 

# --- 2. Activate Environment ---
source activate /scratch/eande106/conda_envs/nemaseg

echo "--- Diagnostic Check ---"
echo "Python Path: $(which python)"
echo "Checking for torch in environment..."
pip list | grep torch

# --- 3. Set Variables ---
export YOLO_CONFIG_DIR=/scratch/eande106/conda_envs/nemaseg/yolo_config
export PYTHONNOUSERSITE=1

# --- 4. GPU & Resource Diagnostics ---
echo "--- Hardware Assignment ---"
nvidia-smi

echo "--- Python/Torch Connection ---"
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"

# --- 5. Main Execution ---
cd /scratch/eande106/ZihaoJohnLi/NemaSeg_Project/Repo/NemaSeg

# Running the script
# python detect_and_crop_rois.py
# python skeletonize_worms.py
# python run_pipeline.py --local /path/to/project
python run_pipeline.py /scratch/eande106/ZihaoJohnLi/datasets/Nextflow_test