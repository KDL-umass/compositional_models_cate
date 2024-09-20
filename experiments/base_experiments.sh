#!/bin/bash
#SBATCH  -t 48:00:00
#SBATCH -p gpu-preempt
#SBATCH --gres=gpu:2
#SBATCH --mem 100GB
#SBATCH --constraint vram80
#SBATCH --cpus-per-task 4
#SBATCH --job-name=test-hypothesis-1
#SBATCH --output=./out/test-hypothesis_%j.out
#SBATCH --error=./out/test-hypothesis_%j.err
conda init bash
conda activate in-context-learning
cd /work/pi_jensen_umass_edu/ppruthi_umass_edu/compositional_models_cate/experiments

# Knobs 
# Composition Type: Additive Parallel
# Structure: Fixed Structure
# Module Function Type: Linear Outcome, Quadratic Outcome
# Vary dimensionality per module: 2 to 10
# Vary number of modules: 1 to 10
# Data distribution: Nornal, Uniform

# Base command
base_cmd="python test_base_hypothesis_1.py"
module_function_types=("mlp" "quadratic")
data_dists=("normal")
feature_dim_list=(2 3 4 5 6 7 8 9 10)
feature_dim_list_10=(2 10 20 30 40 50 60 70 80 90 100)
# feature_dim_list_10=(2)
num_modules_list=(1 2 3 4 5 6 7 8 9 10)
num_modules_list_10=(1 10 20 30 40 50 60 70 80 90 100)


for feature_dim in ${feature_dim_list_10[@]};
do

    # Vary module_function_type
    for module_function_type in ${module_function_types[@]}; 
    do
        # Vary data_dist
        for data_dist in ${data_dists[@]}; 
        do
            $base_cmd --num_modules 10 --num_feature_dimensions $feature_dim --module_function_type $module_function_type --data_dist $data_dist --covariates_shared False --underlying_model_class "MLP" --use_subset_features False --run_env "unity"
        done
    done
done


# # test command 
# python test_base_hypothesis_1.py --num_modules 5 --num_feature_dimensions 2 --module_function_type "mlp" --data_dist "uniform" --covariates_shared False --underlying_model_class "MLP" --run_env "unity"

# Vary num_modules from 1 10 20 for shared covariates
# for num_modules in ${num_modules_list_10[@]};
# do

#     # Vary module_function_type
#     for module_function_type in ${module_function_types[@]}; 
#     do
#         # Vary data_dist
#         for data_dist in ${data_dists[@]}; 
#         do
#             $base_cmd --num_modules $num_modules --module_function_type $module_function_type --data_dist $data_dist --covariates_shared True --num_feature_dimensions 10 --underlying_model_class "MLP" --use_subset_features True
#         done
#     done
# done

# # Vary num_modules from 2 to 10 for non-shared covariates
# for num_modules in ${num_modules_list[@]};
# do

#     # Vary module_function_type
#     for module_function_type in ${module_function_types[@]}; 
#     do
#         # Vary data_dist
#         for data_dist in ${data_dists[@]}; 
#         do
#             $base_cmd --num_modules $num_modules --module_function_type $module_function_type --data_dist $data_dist --covariates_shared False --num_feature_dimensions 10 --underlying_model_class "MLP"
#         done
#     done
# done

# # Vary num_features from 2 to 100 by 10 for shared covariates


# for feature_dim in ${feature_dim_list_10[@]};
# do

#     # Vary module_function_type
#     for module_function_type in ${module_function_types[@]}; 
#     do
#         # Vary data_dist
#         for data_dist in ${data_dists[@]}; 
#         do
#             $base_cmd --num_modules 10 --num_feature_dimensions $feature_dim --module_function_type $module_function_type --data_dist $data_dist --covariates_shared True --underlying_model_class "MLP" --use_subset_features False
#         done
#     done
# done

# # # # Vary num_features from 2 to 10 for num_modules = 10
# for feature_dim in ${feature_dim_list[@]};
# do

#     # Vary module_function_type
#     for module_function_type in ${module_function_types[@]}; 
#     do
#         # Vary data_dist
#         for data_dist in ${data_dists[@]}; 
#         do
#             $base_cmd --num_modules 10 --num_feature_dimensions $feature_dim --module_function_type $module_function_type --data_dist $data_dist --covariates_shared False --underlying_model_class "MLP"
#         done
#     done
# done



# # Vary heterogeneity from 0 to 1 in 0.1 increments
# # based on number of modules vary the heterogeneity
# # num_modules=10
# # heterogeneity_values=(0 1)
# # for heterogeneity in ${heterogeneity_values[@]}; do
# #     $base_cmd --num_modules 10 --feature_dim 5 --noise_level 0 --heterogeneity $heterogeneity
# # done

# # # Vary noise level from 0 to 1 in 0.1 increments
# # for noise_level in $(seq 0 0.1 1); do
# #     $base_cmd --num_modules 5 --feature_dim 5 --noise_level $noise_level --heterogeneity 0
# # done