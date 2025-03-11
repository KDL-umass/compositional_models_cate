from prepare_maths_expression_data import get_obs_df
import argparse
import pandas as pd
from baselines import random_forest, non_param_DML, neural_network, run_catenets
import numpy as np
from sklearn.metrics import r2_score
import os
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='Run baselines for math evaluation domain')
parser.add_argument("--sample_sizes", type=list, help="List of sample sizes", default=[5000])
parser.add_argument('--data_folder', type=str, help='Path to the data', default="data_manjaro")
parser.add_argument('--sampling', type=str, help='Sampling type', required=False, default="observational")
parser.add_argument('--biasing_covariate', type=str, help='Biasing covariate', required=False, default="matrix_size")
parser.add_argument('--treatment_ids', type=list, help='Treatment ids', required=False, default=[0, 1])
parser.add_argument('--plot', type=bool, help='Plot', required=False, default=True)
parser.add_argument('--transform', type=str, help='transformation_type', required=False, default="log")
parser.add_argument("--model", type=str, help="Model to use for estimation", default="random_forest")
parser.add_argument("--num_trials", type=int, help="Number of trials per bias strength and ", default=1)
args = parser.parse_args()
np.random.seed(42)

# set relative path 
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = "{}/data/csvs".format(ROOT_DIR)
plot_dir = "{}/plots".format(ROOT_DIR)
results_dir = "{}/results".format(ROOT_DIR)
obs_data_dir = "{}/obs_data".format(data_dir)

module_files = os.listdir(data_dir)
operators = [file.split(".")[0].split("_")[1] for file in module_files if file.endswith(".csv") and "high_level" not in file]

domain = "matrix_evaluation"
sampling = args.sampling
biasing_covariate = args.biasing_covariate
transform = args.transform
num_trials = args.num_trials
sample_sizes = args.sample_sizes
treatment_ids = args.treatment_ids
model = args.model

if not os.path.exists(obs_data_dir):
    os.makedirs(obs_data_dir)
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

pehe_results = pd.DataFrame(columns=[ "bias_strength", "trial"])
# bias_strengths = list(np.arange(0, 20, 0.5))
bias_strengths = [0]
# bias_strengths = [float(bias_strength) for bias_strength in bias_strengths]


for sample_size in sample_sizes:
    print("Sample Size: {}".format(sample_size))
    for bias_strength in bias_strengths:
        print("Bias Strength: {}".format(bias_strength))
        
        df_sampled = pd.read_csv("{}/df_sampled_{}_{}_{}.csv".format(obs_data_dir, sampling, biasing_covariate, bias_strength))
        df_cf_sampled = pd.read_csv("{}/df_cf_sampled_{}_{}_{}.csv".format(obs_data_dir, sampling, biasing_covariate, bias_strength))
        df = pd.read_csv("{}/df_{}_{}_{}.csv".format(obs_data_dir, sampling, biasing_covariate, bias_strength))
        
        
        for trial in range(num_trials):
            print("Trial: {}".format(trial))
            
            # Sample different treatments per query ID
            query_treatment_ids = df_sampled[['query_id', 'treatment_id']]
            results_outcomes = pd.DataFrame(columns=["query_id", "y1_gt", "y0_gt"])
            
            for op in operators:
                print(op)
                df_op, df_obs_op, df_cf_op = get_obs_df(query_treatment_ids=query_treatment_ids, operator_name=op)
                print(df_op.columns)
                file_path = "{}/query_ids_{}_{}_{}_{}.json".format(obs_data_dir, sampling, biasing_covariate, bias_strength, trial)
                with open(file_path, 'r') as json_file:
                    data = json.load(json_file)
                train_query_ids = data["train_query_ids"][0]
                test_query_ids = data["test_query_ids"][0]
                print(len(train_query_ids), len(test_query_ids))
                
                df_obs_op_train, df_obs_op_test = df_obs_op[df_obs_op["query_id"].isin(train_query_ids)], df_obs_op[df_obs_op["query_id"].isin(test_query_ids)]
                df_cf_op_train, df_cf_op_test = df_cf_op[df_cf_op["query_id"].isin(train_query_ids)], df_cf_op[df_cf_op["query_id"].isin(test_query_ids)]
                df_op_train, df_op_test = df_op[df_op["query_id"].isin(train_query_ids)], df_op[df_op["query_id"].isin(test_query_ids)]

                if len(df_obs_op_train) > sample_size:
                    df_obs_op_train = df_obs_op_train.sample(n=sample_size, random_state=42)
                    df_cf_op_train = df_cf_op_train[df_cf_op_train["query_id"].isin(df_obs_op_train["query_id"])]
                    df_op_train = df_op_train[df_op_train["query_id"].isin(df_obs_op_train["query_id"])]

                
                arg_norm_columns = [col for col in df_op.columns if "norm" in col]
                shape_columns = [col for col in df_op.columns if "shape" in col]
                covariates = ["matrix_size"] + shape_columns

                treatment = "treatment_id"
                outcome = "output"

                
                df_control = df_op_test[df_op_test[treatment] == 0]
                df_treatment = df_op_test[df_op_test[treatment] == 1]

                y1_gt = df_treatment[outcome].values
                y0_gt = df_control[outcome].values
                
                df_outcomes = pd.DataFrame(columns=["query_id", "y1_gt", "y0_gt"])
                df_outcomes["query_id"] = df_treatment["query_id"].values
                df_outcomes["y1_gt"] = y1_gt
                df_outcomes["y0_gt"] = y0_gt
                df_outcomes = df_outcomes.groupby("query_id").sum().reset_index()

                if transform == "log":
                    df_obs_op_train[outcome] = df_obs_op_train[outcome].apply(lambda x: np.log(x+1))
                    df_cf_op_train[outcome] = df_cf_op_train[outcome].apply(lambda x: np.log(x+1))
                    df_op_train[outcome] = df_op_train[outcome].apply(lambda x: np.log(x+1))
                    df_obs_op_test[outcome] = df_obs_op_test[outcome].apply(lambda x: np.log(x+1))
                    df_cf_op_test[outcome] = df_cf_op_test[outcome].apply(lambda x: np.log(x+1))
                    df_op_test[outcome] = df_op_test[outcome].apply(lambda x: np.log(x+1))
                

                if len(df_obs_op_test) != 0:
                    if model == "random_forest":
                        y1, y0 = random_forest(df_obs_op_train, covariates, treatment, outcome, X_test=df_obs_op_test[covariates].values)
                    elif model == "neural_network":
                        y1, y0 = neural_network(df_obs_op_train, covariates, treatment, outcome, X_test=df_obs_op_test[covariates].values)
                        y1 = y1.squeeze()
                        y0 = y0.squeeze()
                    elif model in ["tnet", "snet", "snet1", "snet2", "drnet"]:
                        cate_pred_t, y1, y0 = run_catenets(df_obs_op_train, covariates, treatment, outcome, X_test=df_obs_op_test[covariates].values, baseline=model)
                        y1 = y1.squeeze()
                        y0 = y0.squeeze()
                        
                        
                else:
                    continue
                
                if transform == "log":
                    y1 = np.exp(y1) - 1
                    y0 = np.exp(y0) - 1

                df_estimates = pd.DataFrame(columns=["query_id", "y1_est", "y0_est"])
                df_estimates["query_id"] = df_obs_op_test["query_id"].values
                df_estimates["y1_est"] = y1
                df_estimates["y0_est"] = y0
                df_estimates = df_estimates.groupby("query_id").sum().reset_index()
                
                df_outcomes = pd.merge(df_outcomes, df_estimates, on="query_id", how="inner")
            
                results_outcomes = pd.concat([results_outcomes, df_outcomes])
            
            results_outcomes = results_outcomes.groupby("query_id").sum().reset_index()
            
            if transform == "log":
                results_outcomes["y1_gt"] = results_outcomes["y1_gt"].replace([np.inf, -np.inf], np.nan)
                results_outcomes["y0_gt"] = results_outcomes["y0_gt"].replace([np.inf, -np.inf], np.nan)
                results_outcomes["y1_est"] = results_outcomes["y1_est"].replace([np.inf, -np.inf], np.nan)
                results_outcomes["y0_est"] = results_outcomes["y0_est"].replace([np.inf, -np.inf], np.nan)

                results_outcomes["y1_gt"] = results_outcomes["y1_gt"].apply(lambda x: np.log(x+1))
                results_outcomes["y0_gt"] = results_outcomes["y0_gt"].apply(lambda x: np.log(x+1))
                results_outcomes["y1_est"] = results_outcomes["y1_est"].apply(lambda x: np.log(x+1))
                results_outcomes["y0_est"] = results_outcomes["y0_est"].apply(lambda x: np.log(x+1))
                
            results_outcomes["ite"] = results_outcomes["y1_gt"] - results_outcomes["y0_gt"]
            results_outcomes["estimates"] = results_outcomes["y1_est"] - results_outcomes["y0_est"]
            pehe = np.mean((results_outcomes["ite"] - results_outcomes["estimates"]) ** 2)
            
            pehe_results = pd.concat([pehe_results, pd.DataFrame([[bias_strength, sample_size, trial, pehe]], columns=["bias_strength", "sample_size", "trial", "pehe"])])
pehe_results.to_csv("{}/pehe_results_parallel_additive_model_{}_{}_{}.csv".format(results_folder, sampling, biasing_covariate, model, num_trials), index=False)