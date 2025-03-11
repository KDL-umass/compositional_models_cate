import sys
from prepare_maths_expression_data import generate_high_level_observational_dataset
import argparse
import pandas as pd
from baselines import random_forest, s_learner, x_learner, non_param_DML, neural_network, t_learner, run_catenets
import numpy as np
from sklearn.metrics import r2_score
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import json

parser = argparse.ArgumentParser(description='Run baselines for math evaluation domain')
parser.add_argument("--sample_sizes", type=list, help="List of sample sizes", default=[5000])
parser.add_argument('--sampling', type=str, help='Sampling type', required=False, default="observational")
parser.add_argument('--biasing_covariate', type=str, help='Biasing covariate', required=False, default="matrix_size")
parser.add_argument('--treatment_ids', type=list, help='Treatment ids', required=False, default=[0, 1])
parser.add_argument('--load_from_disk', type=bool, help='Load from disk', required=False, default=False)
parser.add_argument('--transform', type=str, help='transformation_type', required=False, default="log")
parser.add_argument("--num_trials", type=int, help="Number of trials per bias strength", default=1)

args = parser.parse_args()

# setup directories
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = "{}/data/csvs".format(ROOT_DIR)
plot_dir = "{}/plots".format(ROOT_DIR)
results_dir = "{}/results".format(ROOT_DIR)
obs_data_dir = "{}/obs_data".format(data_dir)

np.random.seed(42)

domain = "matrix_evaluation"

sampling = args.sampling
biasing_covariate = args.biasing_covariate
load_from_disk = args.load_from_disk
transform = args.transform
num_trials = args.num_trials
sample_sizes = args.sample_sizes
treatment_ids = args.treatment_ids

if not os.path.exists(obs_data_dir):
    os.makedirs(obs_data_dir)
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

po_baselines = ["random_forest", "neural_network", "non_param_DML", "s_learner", "x_learner", "t_learner"]

pehe_results = pd.DataFrame(columns=["bias_strength", "trial"] + po_baselines)
r2_results = pd.DataFrame(columns=["bias_strength", "trial"] + po_baselines)
bias_strengths = list(np.arange(0, 20, 0.5))
bias_strengths = [float(bias_strength) for bias_strength in bias_strengths]
for sample_size in sample_sizes:
    print("Sample Size: {}".format(sample_size))
    for bias_strength in bias_strengths:
        print("Bias Strength: {}".format(bias_strength))
        if not load_from_disk:
            df, df_sampled, df_cf_sampled = generate_high_level_observational_dataset(treatment_ids=treatment_ids, 
                                                                                        sampling=sampling, 
                                                                                        biasing_covariate=biasing_covariate, 
                                                                                        bias_strength=bias_strength, 
                                                                                        plot_folder=plot_dir, 
                                                                                        )
            query_ids = df["query_id"].unique()
         
            df_sampled.to_csv("{}/df_sampled_{}_{}_{}.csv".format(obs_data_dir, sampling, biasing_covariate, bias_strength), index=False)
            df_cf_sampled.to_csv("{}/df_cf_sampled_{}_{}_{}.csv".format(obs_data_dir, sampling, biasing_covariate, bias_strength), index=False)
            df.to_csv("{}/df_{}_{}_{}.csv".format(obs_data_dir, sampling, biasing_covariate, bias_strength), index=False)
        else:
            df_sampled = pd.read_csv("{}/df_sampled_{}_{}_{}.csv".format(obs_data_dir, sampling, biasing_covariate, bias_strength))
            df_cf_sampled = pd.read_csv("{}/df_cf_sampled_{}_{}_{}.csv".format(obs_data_dir, sampling, biasing_covariate, bias_strength))
            df = pd.read_csv("{}/df_{}_{}_{}.csv".format(obs_data_dir, sampling, biasing_covariate, bias_strength))
        
        matrix_sizes = np.linspace(0, 1000, num=100, dtype=int)
        df_sampled = df_sampled[df_sampled["matrix_size"].isin(matrix_sizes)]
        df_cf_sampled = df_cf_sampled[df_cf_sampled["matrix_size"].isin(matrix_sizes)]
        df = df[df["matrix_size"].isin(matrix_sizes)]
        print(df.shape)
        for trial in range(num_trials):
            print("Trial: {}".format(trial))
            
            # # Sample different treatments per query ID
            # query_ids = df_sampled["query_id"].unique()
            # query_treatment_ids = pd.DataFrame({"query_id": query_ids})
            # query_treatment_ids["treatment_id"] = np.random.choice(treatment_ids, size=len(query_ids))
            
            all_query_ids = df["query_id"]
            train_query_ids, test_query_ids = train_test_split(all_query_ids, test_size=0.2, random_state=42)
            df_sampled_train, df_sampled_test = df_sampled[df_sampled["query_id"].isin(train_query_ids)], df_sampled[df_sampled["query_id"].isin(test_query_ids)]
            df_cf_sampled_train, df_cf_sampled_test = df_cf_sampled[df_cf_sampled["query_id"].isin(train_query_ids)], df_cf_sampled[df_cf_sampled["query_id"].isin(test_query_ids)]
            df_train, df_test = df[df["query_id"].isin(train_query_ids)], df[df["query_id"].isin(test_query_ids)]

            if len(df_sampled_train) > sample_size:
                df_sampled_train = df_sampled_train.sample(n=sample_size, random_state=42)
                df_cf_sampled_train = df_cf_sampled_train[df_cf_sampled_train["query_id"].isin(df_sampled_train["query_id"])]
                df_train = df_train[df_train["query_id"].isin(df_sampled_train["query_id"])]
            

            query_ids = {
                "train_query_ids": [df_sampled_train["query_id"].values.tolist()],
                "test_query_ids": [df_sampled_test["query_id"].values.tolist()]
            }
            train_query_ids = df_sampled_train["query_id"].values.tolist()
            test_query_ids = df_sampled_test["query_id"].values.tolist()
            print("Train Query IDs: {}".format(len(train_query_ids)))
            print("Test Query IDs: {}".format(len(test_query_ids)))
            
            file_path = "{}/query_ids_{}_{}_{}_{}.json".format(obs_data_dir, sampling, biasing_covariate, bias_strength, trial)
            with open(file_path, "w") as json_file:
                json.dump(query_ids, json_file, indent=4)
                        
            arg_norm_columns = [col for col in df.columns if "norm" in col]
            shape_columns = [col for col in df.columns if "shape" in col]
            num_columns = [col for col in df.columns if "num" in col]
            covariates = ["matrix_size"] + shape_columns + num_columns
            print(covariates)

            treatment = "treatment_id"
            outcome = "query_output"

            df_treatment = df_test[df_test[treatment] == 1]
            df_control = df_test[df_test[treatment] == 0]
            query_id_to_y1_gt = dict(zip(df_treatment["query_id"].values, df_treatment[outcome].values))
            query_id_to_y0_gt = dict(zip(df_control["query_id"].values, df_control[outcome].values))

            df_outcomes = pd.DataFrame(list(query_id_to_y1_gt.items()), columns=["query_id", "y1_gt_original"])
            df_outcomes["y0_gt_original"] = df_outcomes["query_id"].map(query_id_to_y0_gt)
            
            if transform == "log":
                df_train[outcome] = df_train[outcome].apply(lambda x: np.log(x+1))
                df_sampled_train[outcome] = df_sampled_train[outcome].apply(lambda x: np.log(x+1))
                df_cf_sampled_train[outcome] = df_cf_sampled_train[outcome].apply(lambda x: np.log(x+1))
                df_test[outcome] = df_test[outcome].apply(lambda x: np.log(x+1))
                df_sampled_test[outcome] = df_sampled_test[outcome].apply(lambda x: np.log(x+1))
                df_cf_sampled_test[outcome] = df_cf_sampled_test[outcome].apply(lambda x: np.log(x+1))
                
            # if plot:
            #     plot_outcome_distribution(df_sampled_train, df_train, outcome, plot_folder, "outcome_distribution_{}_{}_{}_{}_{}.png".format(sampling, biasing_covariate, bias_strength, trial))

            df_control = df_test[df_test[treatment] == 0]
            df_treatment = df_test[df_test[treatment] == 1]
            y1_gt = df_treatment[outcome].values
            y0_gt = df_control[outcome].values
            query_id_to_y1_gt = dict(zip(df_treatment["query_id"].values, y1_gt))
            query_id_to_y0_gt = dict(zip(df_control["query_id"].values, y0_gt))
            df_outcomes["y1_gt"] = df_outcomes["query_id"].map(query_id_to_y1_gt)
            df_outcomes["y0_gt"] = df_outcomes["query_id"].map(query_id_to_y0_gt)
            df_outcomes["ite"] = df_outcomes["y1_gt"] - df_outcomes["y0_gt"]

            pehe_dict = {}
            r2_dict = {}

            for baseline in po_baselines:
                if baseline == "non_param_DML":
                    estimates = non_param_DML(df_sampled_train, covariates, treatment, outcome, X_test=df_sampled_test[covariates].values)
                    query_id_to_ite_estimates = dict(zip(df_sampled_test["query_id"].values, estimates))
                    df_outcomes["ite_estimates_{}".format(baseline)] = df_outcomes["query_id"].map(query_id_to_ite_estimates)
                if baseline == "x_learner":
                    estimates = x_learner(df_sampled_train, covariates, treatment, outcome, X_test=df_sampled_test[covariates].values)
                    query_id_to_ite_estimates = dict(zip(df_sampled_test["query_id"].values, estimates))
                    df_outcomes["ite_estimates_{}".format(baseline)] = df_outcomes["query_id"].map(query_id_to_ite_estimates)
                if baseline == "s_learner":
                    estimates = s_learner(df_sampled_train, covariates, treatment, outcome, X_test=df_sampled_test[covariates].values)
                    query_id_to_ite_estimates = dict(zip(df_sampled_test["query_id"].values, estimates))
                    df_outcomes["ite_estimates_{}".format(baseline)] = df_outcomes["query_id"].map(query_id_to_ite_estimates)
                if baseline == "t_learner":
                    estimates = t_learner(df_sampled_train, covariates, treatment, outcome, X_test=df_sampled_test[covariates].values)
                    query_id_to_ite_estimates = dict(zip(df_sampled_test["query_id"].values, estimates))
                    df_outcomes["ite_estimates_{}".format(baseline)] = df_outcomes["query_id"].map(query_id_to_ite_estimates)
                if baseline == "random_forest":
                    Y1, Y0 = random_forest(df_sampled_train, covariates, treatment, outcome, X_test=df_sampled_test[covariates].values)
                    query_id_to_y1 = dict(zip(df_sampled_test["query_id"].values, Y1))
                    df_outcomes["y1_est_{}".format(baseline)] = df_outcomes["query_id"].map(query_id_to_y1)
                    query_id_to_y0 = dict(zip(df_sampled_test["query_id"].values, Y0))
                    df_outcomes["y0_est_{}".format(baseline)] = df_outcomes["query_id"].map(query_id_to_y0)
                    df_outcomes["ite_estimates_{}".format(baseline)] = df_outcomes["y1_est_{}".format(baseline)] - df_outcomes["y0_est_{}".format(baseline)]
                if baseline == "neural_network":
                    Y1, Y0 = neural_network(df_sampled_train, covariates, treatment, outcome, X_test=df_sampled_test[covariates].values)
                    Y1 = Y1.squeeze()
                    Y0 = Y0.squeeze()
                    query_id_to_y1 = dict(zip(df_sampled_test["query_id"].values, Y1))
                    df_outcomes["y1_est_{}".format(baseline)] = df_outcomes["query_id"].map(query_id_to_y1)
                    query_id_to_y0 = dict(zip(df_sampled_test["query_id"].values, Y0))
                    df_outcomes["y0_est_{}".format(baseline)] = df_outcomes["query_id"].map(query_id_to_y0)
                    df_outcomes["ite_estimates_{}".format(baseline)] = df_outcomes["y1_est_{}".format(baseline)] - df_outcomes["y0_est_{}".format(baseline)]
                if baseline in ["tnet", "snet", "snet1", "snet2", "drnet"]:
                    try:
                        cate_pred_t, Y1, Y0 = run_catenets(df_sampled_train, covariates, treatment, outcome, X_test=df_sampled_test[covariates].values, baseline=baseline)
                        Y1 = Y1.squeeze()
                        Y0 = Y0.squeeze()
                        query_id_to_y1 = dict(zip(df_sampled_test["query_id"].values, Y1))
                        df_outcomes["y1_est_{}".format(baseline)] = df_outcomes["query_id"].map(query_id_to_y1)
                        query_id_to_y0 = dict(zip(df_sampled_test["query_id"].values, Y0))
                        df_outcomes["y0_est_{}".format(baseline)] = df_outcomes["query_id"].map(query_id_to_y0)
                        df_outcomes["ite_estimates_{}".format(baseline)] = df_outcomes["y1_est_{}".format(baseline)] - df_outcomes["y0_est_{}".format(baseline)]
                    except:
                        cate_pred_t = run_catenets(df_sampled_train, covariates, treatment, outcome, X_test=df_sampled_test[covariates].values, baseline=baseline)
                        cate_pred_t = cate_pred_t.squeeze()
                        query_id_to_cate_pred_t = dict(zip(df_sampled_test["query_id"].values, cate_pred_t))
                        df_outcomes["ite_estimates_{}".format(baseline)] = df_outcomes["query_id"].map(query_id_to_cate_pred_t)
                    
                
                pehe = np.mean((df_outcomes["ite"] - df_outcomes["ite_estimates_{}".format(baseline)]) ** 2)
                r2 = r2_score(df_outcomes["ite"], df_outcomes["ite_estimates_{}".format(baseline)])
                # general_scatter_plot(df_outcomes["ite"].values, df_outcomes["ite_estimates_{}".format(baseline)].values, "True ITE", "Predicted ITE", "Test Predictions vs Test Outputs for ITE", "{}/test_predictions_vs_outputs_ite_{}_{}.png".format(plot_folder, baseline, bias_strength))
                pehe_dict[baseline] = pehe
                r2_dict[baseline] = r2
            print("Bias Strength: {}, PEHE: {} R2 {}".format(bias_strength, pehe_dict, r2_dict))

        
            pehe_results = pd.concat([pehe_results, pd.DataFrame([[bias_strength, sample_size, trial] + [pehe_dict[baseline] for baseline in po_baselines]], columns=[ "bias_strength", "sample_size", "trial"] + po_baselines)])
            r2_results = pd.concat([r2_results, pd.DataFrame([[bias_strength, sample_size, trial] + [r2_dict[baseline] for baseline in po_baselines]], columns=[ "bias_strength", "sample_size", "trial"] + po_baselines)])
            

# Save the PEHE results dataframe
pehe_results.to_csv("{}/pehe_results_po_model_{}_{}_{}.csv".format(results_dir, sampling, biasing_covariate, sample_size), index=False)
r2_results.to_csv("{}/r2_results_po_model_{}_{}_{}.csv".format(results_dir, sampling, biasing_covariate, sample_size), index=False)