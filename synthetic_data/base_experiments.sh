#!/bin/bash
#SBATCH  -t 16:00:00
#SBATCH -p gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --mem 100GB
#SBATCH --cpus-per-task 4
#SBATCH --job-name=test-hypothesis-1
#SBATCH --output=./out/synthetic_data_experiments.out
#SBATCH --error=./out/synthetic_data_experiments.err
conda activate in-context-learning
echo "Running experiments for sequential composition"
PYTHON_SCRIPT="sequential_compositional_generalization.py"
cmd="python $PYTHON_SCRIPT --composition_type sequential"
echo "Running: $cmd"
eval $cmd
echo "DONE" 

# run parallel experiments
echo "Running experiments for parallel composition"
PYTHON_SCRIPT="parallel_compositional_generalization.py"
cmd="python $PYTHON_SCRIPT --composition_type parallel"
echo "Running: $cmd"
eval $cmd
echo "DONE"

