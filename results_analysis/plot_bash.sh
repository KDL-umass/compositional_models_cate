#!/bin/bash

base_cmd="python plot_experiment_results.py"
experiments=("feature_dim")
data_dists=("normal")
module_function_types=("mlp")
covariates_shared=("True")


for experiment in ${experiments[@]};
do
    for data_dist in ${data_dists[@]};
    do
        for module_function_type in ${module_function_types[@]};
        do
            for covariate_shared in ${covariates_shared[@]};
            do
                $base_cmd --experiment $experiment --data_dist $data_dist --module_function_type $module_function_type --covariates_shared $covariate_shared  --underlying_model_class "MLP" --use_subset_features True --run_env "unity" --metric pehe
                $base_cmd --experiment $experiment --data_dist $data_dist --module_function_type $module_function_type --covariates_shared $covariate_shared --underlying_model_class "MLP" --use_subset_features False --run_env "unity" --metric pehe
       
            done
        done
    done
done

