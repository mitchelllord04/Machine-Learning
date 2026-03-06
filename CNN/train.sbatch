#!/bin/bash
#SBATCH -J CIFAR10CNN
#SBATCH -N 1 --ntasks-per-node=1
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=40G
#SBATCH -t 1:00:00
#SBATCH -o Output-%j.out

cd $SLURM_SUBMIT_DIR

module load python

VENV_DIR="venv"

# Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR" || exit 1
fi

source "$VENV_DIR/bin/activate"

# Install dependencies only once
if [ ! -f "$VENV_DIR/.deps_installed" ]; then
  python3 -m pip install --upgrade pip
  pip install -r requirements.txt
  touch "$VENV_DIR/.deps_installed"
fi

echo "Starting training..."
python3 -u train.py