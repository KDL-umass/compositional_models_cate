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
parser.add_argument("--covariates_shared", type=lambda x: (str(x).lower() == 'true'), default=False, help="Covariates shared")
parser.add_argument("--run_env", type=str, default="local", help="Run environment")
parser.add_argument("--underlying_model_class", type=str, default="MLP", help="Model class")
parser.add_argument("--use_subset_features", type=lambda x: (str(x).lower() == 'true'), default=False, help="Use subset features")
parser.add_argument("--systematic", type=lambda x: (str(x).lower() == 'true'), default=False, help="Generate trees systematically")
parser.add_argument("--metric", type=str, default="pehe", help="Metric to plot")
parser.add_argument("--scale", action="store_true", help="Scale data")
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
metric = args.metric
scale = args.scale
# if covariates_shared == "True": 
#     covariates_shared = True
# else:
#     covariates_shared = False
print(args)
if run_env == "local":
    base_dir = "/Users/ppruthi/research/compositional_models/compositional_models_cate/domains"
else:
    base_dir = "/work/pi_jensen_umass_edu/ppruthi_umass_edu/compositional_models_cate/domains"


results_path = f"{base_dir}/synthetic_data/results/results_{data_dist}_{module_function_type}_{composition_type}_covariates_shared_{covariates_shared}_underlying_model_{underlying_model_class}_use_subset_features_{args.use_subset_features}_systematic_{systematic}"

# Lists to store the data for plotting
# heterogeneity_values = []
if experiment == "num_modules":
    if covariates_shared == 'true':
        num_modules_values = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    else:
        num_modules_values = list(np.arange(1, 11))
    feature_dim_values = [10]
    varying_values = num_modules_values
elif experiment == "feature_dim":
    if covariates_shared == True:
        feature_dim_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    else:
        feature_dim_values = list(np.arange(2, 11))
    num_modules_values = [10]
    
    varying_values = feature_dim_values

metric_baseline_values_train = []
metric_additive_values_train = []
metric_moe_values_train = []
metric_baseline_values_test = []
metric_additive_values_test = []
metric_moe_values_test = []
module_pehe_decomposition_train_sum = []
module_pehe_decomposition_test_sum = []
module_pehe_decomposition_train_cov = []
module_pehe_decomposition_test_cov = []

print(num_modules_values)
print(feature_dim_values)
# Collect data from result files
for num_modules in num_modules_values:
    for feature_dim in feature_dim_values:
        
        results_file = f"{results_path}/results_{num_modules}_{feature_dim}_scale_{scale}.json"
        print(f"Checking for {results_file}")
        if os.path.exists(results_file):
            print(f"Reading results from {results_file}")
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            if metric == "pehe":
                idx = 0
            else:
                idx = 1
            metric_baseline_values_train.append(results['Baseline_train'][idx])
            metric_additive_values_train.append(results['Additive_train'][idx])
            metric_moe_values_train.append(results['MoE_train'][idx])
            metric_baseline_values_test.append(results['Baseline_test'][idx])
            metric_additive_values_test.append(results['Additive_test'][idx])
            metric_moe_values_test.append(results['MoE_test'][idx])
            module_pehe_decomposition_train_sum.append(results['Module_PEHE_decomposition_train'][0])
            module_pehe_decomposition_train_cov.append(results['Module_PEHE_decomposition_train'][1])
            module_pehe_decomposition_test_sum.append(results['Module_PEHE_decomposition_test'][0])
            module_pehe_decomposition_test_cov.append(results['Module_PEHE_decomposition_test'][1])


# # Create the plot
plt.figure(figsize=(10, 6))
# plt.plot(varying_values, metric_baseline_values_train, 'b-', label='Baseline Model (Train)')
# plt.plot(varying_values, metric_additive_values_train, 'r-', label='Additive Model (Train)')
# plt.plot(varying_values, metric_moe_values_train, 'g-', label='MoE Model (Train)')
if systematic == True:
    label_add = " (OOD)"
else:
    label_add = ""

if systematic == True:
    # also plot train 
    train_label_add = " (IID)"
    plt.plot(varying_values, metric_baseline_values_train, 'b--', label='Baseline Model (Train)' + train_label_add)
    plt.plot(varying_values, metric_additive_values_train, 'r--', label='Additive Model (Train)' + train_label_add)
    plt.plot(varying_values, metric_moe_values_train, 'g--', label='MoE Model (Train)' + train_label_add)
plt.plot(varying_values, metric_baseline_values_test, 'b-', label='Baseline Model (Test)' + label_add)
plt.plot(varying_values, metric_additive_values_test, 'r-', label='Additive Model (Test)' + label_add)
plt.plot(varying_values, metric_moe_values_test, 'g-', label='MoE Model (Test)' + label_add)
# For module pehe decomposition, have base color as red only but with different line styles than the additive model
# plt.plot(varying_values, module_pehe_decomposition_test_sum, 'r--', label='Module PEHE Decomposition (Sum) (Test)' + label_add)
# plt.plot(varying_values, module_pehe_decomposition_test_cov, 'r-.', label='Module PEHE Decomposition (Cov) (Test)' + label_add)

plt.xlabel(f'{experiment}')
if metric == "pehe":
    plt.ylabel('PEHE (Precision in Estimation of Heterogeneous Effect)')
else:
    plt.ylabel('R^2')
plt.title(f'Data Dist = {data_dist}, Module Function = {module_function_type}, Covariates Shared = {covariates_shared}, Underlying Model = {underlying_model_class}, Use Subset Features = {use_subset_features}')
# legend outside the plot but do not cut off the plot
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True)

# Save the plot
plot_dir = f"plots/plots_{data_dist}_{module_function_type}_{composition_type}_covariates_shared_{covariates_shared}_underlying_model_{underlying_model_class}_use_subset_features_{args.use_subset_features}_systematic_{systematic}"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
plot_file = f"{plot_dir}/plot_{experiment}_{metric}.png"

plt.savefig(plot_file, bbox_inches='tight')
plt.close()

print(f"Plot saved as {plot_file}")