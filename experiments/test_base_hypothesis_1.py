# Imports 
import pandas as pd
import numpy as np
import os
import json
import warnings
import argparse
import sys
sys.path.append('../')
from domains.samplers import *
from models.MoE import *
from models.utils import *
from models.additive_parallel_comp_model import get_additive_model_effects
from sklearn.metrics import r2_score
warnings.filterwarnings('ignore')

# Define the argument parser
parser = argparse.ArgumentParser(description="Train modular neural network architectures and baselines for causal effect estimation")
parser.add_argument("--num_modules", type=int, default=1, help="Number of modules")
parser.add_argument("--num_feature_dimensions", type=int, default=10, help="Number of feature dimensions")
parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples")
parser.add_argument("--module_function_type", type=str, default="linear", help="Module function type")
parser.add_argument("--composition_type", type=str, default="parallel", help="Composition type")
parser.add_argument("--resample", type=bool, default=True, help="Resample data")
parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
parser.add_argument("--fixed_structure", type=bool, default=True, help="Fixed structure flag")
parser.add_argument("--data_dist", type=str, default="uniform", help="Data distribution")
parser.add_argument("--heterogeneity", type=float, default=1.0, help="Heterogeneity")
parser.add_argument("--split_type", type=str, default="iid", help="Split type")
# hidden_dim
parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension")
# epochs
parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
# batch_size
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
# output_dim
parser.add_argument("--output_dim", type=int, default=1, help="Output dimension")
# covariates_shared
parser.add_argument("--covariates_shared", type=str, default="False", help="Covariates shared")
# model_class
parser.add_argument("--underlying_model_class", type=str, default="MLP", help="Model class")
# run_env
parser.add_argument("--run_env", type=str, default="local", help="Run environment")

# parse arguments
args = parser.parse_args()
num_modules = args.num_modules
max_depth = num_modules
feature_dim = args.num_feature_dimensions
num_samples = args.num_samples
composition_type = args.composition_type
module_function_type = args.module_function_type
resample = args.resample
fixed_structure = args.fixed_structure
num_trees = num_samples
seed = args.seed
data_dist = args.data_dist
split_type = args.split_type    
hidden_dim = args.hidden_dim
epochs = args.epochs
batch_size = args.batch_size
output_dim = args.output_dim
data_dist = args.data_dist
covariates_shared = args.covariates_shared
underlying_model_class = args.underlying_model_class
if covariates_shared == "True":
    covariates_shared = True
else:
    covariates_shared = False

domain = "synthetic_data"
# setup directories
run_env = args.run_env

if run_env == "local":
    base_dir = "/Users/ppruthi/research/compositional_models/compositional_models_cate/domains"
else:
    base_dir = "/work/pi_jensen_umass_edu/ppruthi_umass_edu/compositional_models_cate/domains"

main_dir = f"{base_dir}/{domain}"
csv_path = f"{main_dir}/csvs/fixed_structure_{fixed_structure}_outcomes_{composition_type}"
obs_data_path = f"{main_dir}/observational_data/fixed_structure_{fixed_structure}_outcomes_{composition_type}"

# simulate data
sampler = SyntheticDataSampler(num_modules, feature_dim, composition_type, fixed_structure, max_depth, num_trees, seed, data_dist, module_function_type, resample=resample,heterogeneity=args.heterogeneity, covariates_shared=covariates_shared)
sampler.simulate_data()

# read the data
data_path = f"{csv_path}/{domain}_data_high_level_features.csv"
data = pd.read_csv(data_path)

# create observational data
results = sampler.create_observational_data(biasing_covariate="feature_sum", bias_strength=0)
df_sampled = pd.read_csv(f"{obs_data_path}/feature_sum_0/df_sampled.csv")

# split the data into features, treatment, and outcome
sampler.create_iid_ood_split(split_type = args.split_type)

# load train and test data
train_test_qids_path = f"{csv_path}/{args.split_type}/train_test_split_qids.json"
with open(train_test_qids_path, "r") as f:
    train_test_qids = json.load(f)
train_qids = train_test_qids["train"]
test_qids = train_test_qids["test"]

train_df = df_sampled[df_sampled["query_id"].isin(train_qids)]
test_df = df_sampled[df_sampled["query_id"].isin(test_qids)]

# unitary Baseline
if covariates_shared:
    covariates = [x for x in train_df.columns if "module_1_feature" in x]
else:
    covariates = [x for x in train_df.columns if "feature" in x]
treatment = "treatment_id"
outcome = "query_output"

# X_test = test_df[covariates].values
if args.split_type == "iid":
    test_df = train_df

input_dim = len(covariates)
if underlying_model_class == "MLP":
    # make sure hidden dim is greater than input dim
    baseline_model = BaselineModel(input_dim + 1, (input_dim + 1)*2, output_dim)
else:
    baseline_model = BaselineLinearModel(input_dim + 1, output_dim)
print("Training Baseline Model")
print(train_df.head())
baseline_model, train_losses, val_losses = train_model(baseline_model, train_df, covariates, treatment, outcome, epochs, batch_size)
causal_effect_estimates = predict_model(baseline_model, test_df, covariates)
test_df.loc[:, "estimated_effect"] = causal_effect_estimates
baseline_estimated_effects = get_estimated_effects(test_df, train_qids)
baseline_causal_effect_dict_test = get_ground_truth_effects(data, train_qids)

# have combined df with ground truth and estimated effects
baseline_combined_df = pd.DataFrame({"ground_truth_effect": list(baseline_causal_effect_dict_test.values()), "estimated_effect": list(baseline_estimated_effects.values())})

# MoE Baseline
if underlying_model_class == "MLP":
    moe_model = MoE(input_dim+1, (input_dim + 1)*2, output_dim, num_modules)
else:
    moe_model = MoELinear(input_dim+1, output_dim, num_modules)
train_model(moe_model, train_df, covariates, treatment, outcome, epochs, batch_size)
moe_causal_effect_estimates = predict_model(moe_model, test_df, covariates)
test_df.loc[:, "estimated_effect"] = moe_causal_effect_estimates
moe_estimated_effects = get_estimated_effects(test_df, train_qids)
baseline_combined_df["estimated_effect_moe"] = list(moe_estimated_effects.values())

# Explicitly modular model+
print("Training Additive Model")
additive_combined_df = get_additive_model_effects(csv_path, obs_data_path, train_qids, test_qids, hidden_dim=hidden_dim, epochs=epochs, batch_size=batch_size, output_dim=output_dim, underlying_model_class=underlying_model_class)
# combine the two dataframes: baseline_combined_df and additive_combined_df on query_id
baseline_combined_df["query_id"] = list(baseline_combined_df.index)
additive_combined_df["query_id"] = list(additive_combined_df.index)
combined_df = pd.merge(baseline_combined_df, additive_combined_df, on="query_id", suffixes=("_baseline", "_additive"))

# Save Results
pehe_baseline = pehe(combined_df["ground_truth_effect_baseline"], combined_df["estimated_effect_baseline"])
pehe_additive = pehe(combined_df["ground_truth_effect_additive"], combined_df["estimated_effect_additive"])
pehe_moe = pehe(combined_df["ground_truth_effect_baseline"], combined_df["estimated_effect_moe"])

print(combined_df.head())

r2_baseline = get_r2_score(combined_df["ground_truth_effect_baseline"], combined_df["estimated_effect_baseline"])
r2_additive = get_r2_score(combined_df["ground_truth_effect_additive"], combined_df["estimated_effect_additive"])
r2_moe = get_r2_score(combined_df["ground_truth_effect_baseline"], combined_df["estimated_effect_moe"])


print(f"PEHE for baseline model: {pehe_baseline}")
print(f"PEHE for additive model: {pehe_additive}")
print(f"PEHE for MoE model: {pehe_moe}")

# save the results
results = {"pehe_baseline": pehe_baseline, "pehe_additive": pehe_additive, "pehe_moe": pehe_moe, "r2_baseline": r2_baseline, "r2_additive": r2_additive, "r2_moe": r2_moe}
results_path = f"{main_dir}/results/results_{data_dist}_{module_function_type}_{composition_type}_covariates_shared_{covariates_shared}_underlying_model_{underlying_model_class}"
os.makedirs(results_path, exist_ok=True)
results_file = f"{results_path}/results_{num_modules}_{feature_dim}.json"
print(f"Results saved at {results_file}")
with open(results_file, "w") as f:
    json.dump(results, f)