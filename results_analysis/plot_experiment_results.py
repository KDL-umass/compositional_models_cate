import os
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Parameters
# add experiment as argument
parser = argparse.ArgumentParser(description="Plot experiment results")
parser.add_argument("--experiment", type=str, default="num_modules", help="Experiment to plot")
parser.add_argument("--data_dist", type=str, default="uniform", help="Data distribution")
parser.add_argument("--module_function_type", type=str, default="linear", help="Module function type")
parser.add_argument("--composition_type", type=str, default="parallel", help="Composition type")
parser.add_argument("--covariates_shared", type=str, default="False", help="Covariates shared")
parser.add_argument("--run_env", type=str, default="local", help="Run environment")
parser.add_argument("--underlying_model_class", type=str, default="MLP", help="Model class")
parser.add_argument("--use_subset_features", type=str, default="False", help="Use subset of features")
parser.add_argument("--systematic", type=str, default="False", help="Use systematic features")

args = parser.parse_args()
experiment = args.experiment

# Directory where the results are stored
data_dist = args.data_dist
module_function_type = args.module_function_type
composition_type = args.composition_type
covariates_shared = args.covariates_shared
run_env = args.run_env
underlying_model_class = args.underlying_model_class
systematic = args.systematic
use_subset_features = args.use_subset_features
# if covariates_shared == "True": 
#     covariates_shared = True
# else:
#     covariates_shared = False
if run_env == "local":
    base_dir = "/Users/ppruthi/research/compositional_models/compositional_models_cate/domains"
else:
    base_dir = "/work/pi_jensen_umass_edu/ppruthi_umass_edu/compositional_models_cate/domains"


results_path = f"{base_dir}/synthetic_data/results/results_{data_dist}_{module_function_type}_{composition_type}_covariates_shared_{covariates_shared}_underlying_model_{underlying_model_class}_use_subset_features_{args.use_subset_features}_systematic_{systematic}"

# Lists to store the data for plotting
# heterogeneity_values = []
if experiment == "num_modules":
    if covariates_shared == "True":
        num_modules_values = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    else:
        num_modules_values = list(np.arange(1, 11))
    feature_dim_values = [10]
    varying_values = num_modules_values
elif experiment == "feature_dim":
    if covariates_shared == "True":
        feature_dim_values = [2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    else:
        feature_dim_values = list(np.arange(2, 11))
    num_modules_values = [10]
    
    varying_values = feature_dim_values

pehe_baseline_values = []
pehe_additive_values = []
pehe_moe_values = []

print(num_modules_values)
print(feature_dim_values)
# Collect data from result files
for num_modules in num_modules_values:
    for feature_dim in feature_dim_values:
        results_file = f"{results_path}/results_{num_modules}_{feature_dim}.json"
        print(f"Checking for {results_file}")
        if os.path.exists(results_file):
            print(f"Reading results from {results_file}")
            with open(results_file, 'r') as f:
                results = json.load(f)
            

            pehe_baseline_values.append(results['pehe_baseline'])
            pehe_additive_values.append(results['pehe_additive'])
            pehe_moe_values.append(results['pehe_moe'])

# # Create the plot
plt.figure(figsize=(10, 6))
plt.plot(varying_values, pehe_baseline_values, 'b-', label='Unitary Model')
plt.plot(varying_values, pehe_additive_values, 'r-', label='Additive Parallel Composition Model')
plt.plot(varying_values, pehe_moe_values, 'g-', label='Mixture of Experts Model')

plt.xlabel(f'{experiment}')
plt.ylabel('PEHE (Precision in Estimation of Heterogeneous Effect)')
plt.title(f'Performance vs {experiment}')
plt.legend()
plt.grid(True)

# Save the plot
plot_dir = f"plots_{data_dist}_{module_function_type}_{composition_type}_covariates_shared_{covariates_shared}_underlying_model_{underlying_model_class}"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
plot_file = f"{plot_dir}/plot_{experiment}.png"

plt.savefig(plot_file)
plt.close()

print(f"Plot saved as {plot_file}")