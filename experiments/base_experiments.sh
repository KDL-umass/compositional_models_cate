#!/bin/bash
#SBATCH  -t 16:00:00
#SBATCH -p gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --mem 100GB
#SBATCH --cpus-per-task 4
#SBATCH --job-name=test-hypothesis-1
#SBATCH --output=./out/test-hypothesis_%j.out
#SBATCH --error=./out/test-hypothesis_%j.err
conda activate in-context-learning
cd /work/pi_jensen_umass_edu/ppruthi_umass_edu/compositional_models_cate/experiments

# Path to your Python script
# PYTHON_SCRIPT="test_base_hypotheses_modularized.py"
PYTHON_SCRIPT="parallel_compositional_generalization.py"
cmd="python $PYTHON_SCRIPT"
echo "Running: $cmd"
eval $cmd
# print DONE
echo "DONE"

# Common arguments that are always true


# # Function to run experiments
# run_experiments() {
#     local systematic=$1
#     local covariates_shared=$2
#     local use_subset_features=$3
#     local feature_dim_list=("${@:4}")

#     for feature_dim in "${feature_dim_list[@]}"; do
#         for module_function_type in "linear" "quadratic" "mlp"; do
#             for data_dist in "normal" "uniform"; do
#                 for bias_strength in {0..10}; do
#                     # Determine the split_type based on systematic
#                     local split_type="ood"
#                     if [ "$systematic" = "false" ]; then
#                         split_type="iid"
#                     fi

#                     cmd="python $PYTHON_SCRIPT \
#                         --num_modules 6 \
#                         --num_feature_dimensions $feature_dim \
#                         --module_function_type $module_function_type \
#                         --data_dist $data_dist \
#                         --covariates_shared $covariates_shared \
#                         --underlying_model_class MLP \
#                         --use_subset_features $use_subset_features \
#                         --systematic $systematic \
#                         --split_type $split_type \
#                         --bias_strength $bias_strength"
#                     echo "Running: $cmd"
#                     eval $cmd
#                 done
#             done
#         done
#     done
# }

# # Feature dimension lists
# feature_dim_list=(1 2 3 4 5 6 7 8 9 10)
# feature_dim_list_10=(1 2 5 10 20 30 40 50 60 70 80 90 100)

# # Run experiments for systematic true and false
# for systematic in true; do
#     echo "Running experiments with systematic=$systematic"
    
#     # Covariates shared true
#     echo "Running experiments with covariates_shared=true, use_subset_features=true"
#     run_experiments $systematic true true "${feature_dim_list_10[@]}"
    
#     echo "Running experiments with covariates_shared=true, use_subset_features=false"
#     run_experiments $systematic true false "${feature_dim_list_10[@]}"

#     # Covariates shared false
#     echo "Running experiments with covariates_shared=false, use_subset_features=false"
#     run_experiments $systematic false false "${feature_dim_list[@]}"
# done

# echo "All experiments completed."