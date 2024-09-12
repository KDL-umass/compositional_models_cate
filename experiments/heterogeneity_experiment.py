# The goal of this experiment is to understand the tradeoffs between the additive model
# and the baseline model when the heterogeneity among the modules is modified for a given 
# d_j=dimension of the modules and k = number of modules, 
# and # the noise level is varied.
import sys
sys.path.append('../')
from domains.samplers import *
import pandas as pd
import numpy as np
import os
import json
import warnings
import argparse
from models.MoE import *


# ignore below warnings
# A value is trying to be set on a copy of a slice from a DataFrame.
# Try using .loc[row_indexer,col_indexer] = value instead
warnings.filterwarnings('ignore')


def get_ground_truth_effects(data, qids, treatment_col='treatment_id', outcome_col='query_output'):
    # get ground truth effects from data 
    # Group by query_id and treatment_id, then unstack to have treatments as columns
    grouped = data.groupby(['query_id', treatment_col])[outcome_col].first().unstack()

    # Calculate the causal effect (treatment 1 - treatment 0)
    causal_effect = grouped[1] - grouped[0]

    # Convert to dictionary
    causal_effect_dict = causal_effect.to_dict()

    causal_effect_dict_test = {k: v for k, v in causal_effect_dict.items() if k in qids}
    return causal_effect_dict_test

def get_estimated_effects(data, qids):
    estimated_effects = data.groupby("query_id")["estimated_effect"].first().to_dict()
    estimated_effects_test = {k: v for k, v in estimated_effects.items() if k in qids}
    return estimated_effects_test

# calculate pehe for the two models
def pehe(ground_truth, estimated):
    return np.sqrt(np.mean((ground_truth - estimated)**2))

# add arguments for num_modules, feature_dim, noise_level, and heterogeneity
def parse_arguments():
    parser = argparse.ArgumentParser(description="Train additive parallel model and baselines for causal effect estimation")
    parser.add_argument("--num_modules", type=int, default=10, help="Number of modules")
    parser.add_argument("--feature_dim", type=int, default=3, help="Feature dimension")
    parser.add_argument("--noise_level", type=float, default=0.1, help="Noise level")
    parser.add_argument("--heterogeneity", type=float, default=1.0, help="Heterogeneity")
    # split type
    parser.add_argument("--split_type", type=str, default="iid", help="Split type")
    # run environment
    parser.add_argument("--run_env", type=str, default="local", help="Run environment")
    return parser.parse_args()
args = parse_arguments()
# print(args)
print(f"Running experiment with num_modules: {args.num_modules}, feature_dim: {args.feature_dim}, noise_level: {args.noise_level}, heterogeneity: {args.heterogeneity}")
num_modules = args.num_modules
feature_dim = args.feature_dim
noise_level = args.noise_level
heterogeneity = args.heterogeneity
composition_type = "parallel"
fixed_structure = "True"
fixed_structure_bool = True if fixed_structure == "True" else False
max_depth = num_modules
num_trees = 5000
seed = 42
data_dist = "uniform"
module_function_type_list = ["quadratic_module"]
resample = True
domain = "synthetic_data"
run_env = args.run_env

if run_env == "local":
    base_dir = "/Users/ppruthi/research/compositional_models/compositional_models_cate/domains"
else:
    base_dir = "/work/pi_jensen_umass_edu/ppruthi_umass_edu/compositional_models_cate/domains"


main_dir = f"{base_dir}/{domain}"
csv_path = f"{main_dir}/csvs/fixed_structure_{fixed_structure}_outcomes_{composition_type}"
obs_data_path = f"{main_dir}/observational_data/fixed_structure_{fixed_structure}_outcomes_{composition_type}"
# simulate data
sampler = SyntheticDataSampler(num_modules, feature_dim, composition_type, fixed_structure_bool, max_depth, num_trees, seed, data_dist, module_function_type_list, resample=resample,heterogeneity=heterogeneity)
sampler.simulate_data()

# read the data
data_path = f"{csv_path}/{domain}_data_high_level_features.csv"
data = pd.read_csv(data_path)

# Create observational data
results = sampler.create_observational_data(biasing_covariate="feature_sum", bias_strength=0)
df_sampled = pd.read_csv(f"{obs_data_path}/feature_sum_0/df_sampled.csv")

# Split the data into features, treatment, and outcome
sampler.create_iid_ood_split(split_type = args.split_type)
train_test_qids_path = f"{csv_path}/{args.split_type}/train_test_split_qids.json"
with open(train_test_qids_path, "r") as f:
    train_test_qids = json.load(f)
train_qids = train_test_qids["train"]
test_qids = train_test_qids["test"]

train_df = df_sampled[df_sampled["query_id"].isin(train_qids)]
test_df = df_sampled[df_sampled["query_id"].isin(test_qids)]

# test one of the baselines 

covariates = [x for x in train_df.columns if "feature" in x]
treatment = "treatment_id"
outcome = "query_output"
# X_test = test_df[covariates].values
test_df = train_df
input_dim = len(covariates)
output_dim = 1
hidden_dim = 64
epochs = 100
batch_size = 64
baseline_model = BaselineModel(input_dim + 1, hidden_dim, output_dim)
train_model(baseline_model, train_df, covariates, treatment, outcome, epochs, batch_size)
causal_effect_estimates = predict_model(baseline_model, test_df, covariates)
test_df.loc[:, "estimated_effect"] = causal_effect_estimates


baseline_estimated_effects = get_estimated_effects(test_df, train_qids)
baseline_causal_effect_dict_test = get_ground_truth_effects(data, train_qids)


# have combined df with ground truth and estimated effects
baseline_combined_df = pd.DataFrame({"ground_truth_effect": list(baseline_causal_effect_dict_test.values()), "estimated_effect": list(baseline_estimated_effects.values())})

# Moe models
# have a MoE model with the same data
moe_model = MoE(input_dim+1, hidden_dim, output_dim, num_modules)
train_model(moe_model, train_df, covariates, treatment, outcome, epochs, batch_size)
moe_causal_effect_estimates = predict_model(moe_model, test_df, covariates)
test_df.loc[:, "estimated_effect"] = moe_causal_effect_estimates
moe_estimated_effects = get_estimated_effects(test_df, train_qids)
baseline_combined_df["estimated_effect_moe"] = list(moe_estimated_effects.values())

# read all the module files 
module_files = os.listdir(f"{csv_path}/")
module_files = [x for x in module_files if "module" in x]

# read all the module files
module_data = {}
for module_file in module_files:
    module_data[module_file] = pd.read_csv(f"{csv_path}/{module_file}")
    

query_id_treatment_id_json_path = f"{obs_data_path}/feature_sum_0/treatment_assignments.json"
with open(query_id_treatment_id_json_path, "r") as f:
    query_id_treatment_id = json.load(f)
train_data = {}
test_data = {}
for module_file in module_files:
    module_df = module_data[module_file]
    module_df["assigned_treatment_id"] = module_df["query_id"].apply(lambda x: query_id_treatment_id[str(x)])
    module_df = module_df[module_df["treatment_id"] == module_df["assigned_treatment_id"]]
    # drop the assigned treatment id
    module_df.drop("assigned_treatment_id", axis=1, inplace=True)
    # split the data into train and test
    train_data[module_file] = module_df[module_df["query_id"].isin(train_qids)]
    test_data[module_file] = module_df[module_df["query_id"].isin(test_qids)]
    
for module_file in module_files:
    train_df = train_data[module_file]
    covariates = [x for x in train_df.columns if "feature" in x]
    treatment = "treatment_id"
    outcome = "output"
    input_dim = len(covariates)
    output_dim = 1
    hidden_dim = 32
    epochs = 100
    expert_model = BaselineModel(input_dim + 1, hidden_dim, output_dim)
    train_model(expert_model, train_df, covariates, treatment, outcome, epochs, batch_size)
    causal_effect_estimates = predict_model(expert_model, train_df, covariates)
    train_df["estimated_effect"] = causal_effect_estimates
    train_data[module_file] = train_df

# now for each module, get the ground truth and estimated effects
additive_ground_truth_effects = {}
additive_estimated_effects = {}
for module_file in module_files:
    train_df = train_data[module_file]
    test_df = test_data[module_file]
    module_causal_effect_dict_test = get_ground_truth_effects(module_data[module_file], train_qids, treatment_col="treatment_id", outcome_col="output")
    module_estimated_effects = get_estimated_effects(train_df, train_qids)
    module_combined_df = pd.DataFrame({"ground_truth_effect": list(module_causal_effect_dict_test.values()), "estimated_effect": list(module_estimated_effects.values())})
    if len(additive_estimated_effects) == 0:
        additive_ground_truth_effects = module_causal_effect_dict_test
        additive_estimated_effects = module_estimated_effects
    else:
        # add the effects
        additive_ground_truth_effects = {k: v + module_causal_effect_dict_test[k] for k, v in additive_ground_truth_effects.items()}
        additive_estimated_effects = {k: v + module_estimated_effects[k] for k, v in additive_estimated_effects.items()}
    
additive_combined_df = pd.DataFrame({"ground_truth_effect": additive_ground_truth_effects, "estimated_effect": additive_estimated_effects})
# combine the two dataframes: baseline_combined_df and additive_combined_df on query_id
baseline_combined_df["query_id"] = list(baseline_combined_df.index)
additive_combined_df["query_id"] = list(additive_combined_df.index)
combined_df = pd.merge(baseline_combined_df, additive_combined_df, on="query_id", suffixes=("_baseline", "_additive"))



pehe_baseline = pehe(combined_df["ground_truth_effect_baseline"], combined_df["estimated_effect_baseline"])
pehe_additive = pehe(combined_df["ground_truth_effect_additive"], combined_df["estimated_effect_additive"])
pehe_moe = pehe(combined_df["ground_truth_effect_baseline"], combined_df["estimated_effect_moe"])
print(f"PEHE for baseline model: {pehe_baseline}")
print(f"PEHE for additive model: {pehe_additive}")
print(f"PEHE for MoE model: {pehe_moe}")
# save the results
results = {"pehe_baseline": pehe_baseline, "pehe_additive": pehe_additive, "pehe_moe": pehe_moe}
results_path = f"{main_dir}/results"
os.makedirs(results_path, exist_ok=True)
results_file = f"{results_path}/results_{num_modules}_{feature_dim}_{noise_level}_{heterogeneity}.json"
with open(results_file, "w") as f:
    json.dump(results, f)

