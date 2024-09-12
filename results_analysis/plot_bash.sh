#!/bin/bash

base_cmd="python plot_experiment_results.py"
experiments=("num_modules" "feature_dim")
data_dists=("normal" "uniform")
module_function_types=("linear" "quadratic")
covariates_shared=("True" "False")

for experiment in ${experiments[@]};
do
    for data_dist in ${data_dists[@]};
    do
        for module_function_type in ${module_function_types[@]};
        do
            for covariate_shared in ${covariates_shared[@]};
            do
                $base_cmd --experiment $experiment --data_dist $data_dist --module_function_type $module_function_type --covariates_shared $covariate_shared
            done
        done
    done
done