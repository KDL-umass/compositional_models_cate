import numpy as np
import pandas as pd
from prepare_query_plan_data import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
from catenets.models.jax import TNet, SNet, DRNet, SNet1, SNet2

import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

import argparse
import glob
import re
# from bartpy.sklearnmodel import SklearnModel
from econml.dml import CausalForestDML
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import sys 

# define root dir as one level up
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# have one path up
ROOT_DIR = os.path.dirname(ROOT_DIR)
DATA_DIR = "{}/data_gen/queries/data".format(ROOT_DIR)
print(DATA_DIR)
import sys
sys.path.append("{}/data_gen".format(ROOT_DIR))

sys.path.append(DATA_DIR)
from plan_tree import PlanExplainer, PlanTree
from treatment_implementation import generate_basic_config, get_indiv_treatement_idx

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# set random seed
np.random.seed(42)
def plot_cate_results(results, plot_dir, plot_name):
    results = results[results['operator'].isin(['High Level', 'Component Level Model'])]
    # # for each noise variance, plot r2 score vs. prob
    mean = results.groupby(['noise_variance', 'identifier', 'operator']).mean().reset_index()
    std = results.groupby(['noise_variance', 'identifier', 'operator']).std().reset_index()
    noise_variance = 0.0
    mean_noise = mean[mean['noise_variance'] == noise_variance]
    std_noise = std[std['noise_variance'] == noise_variance]
    # except "Seq Scan"
    mean_noise = mean_noise[mean_noise['operator'] != 'Seq Scan']
    std_noise = std_noise[std_noise['operator'] != 'Seq Scan']
    # mean_noise = mean_noise.reset_index()
    plt.figure(figsize=(10,10))
    sns.lineplot(x=mean_noise["identifier"], y=mean_noise['r2_score'], hue = mean_noise['operator'], marker = 'o')
    # error bars
    plt.errorbar(mean_noise["identifier"], mean_noise['r2_score'], yerr = std_noise['r2_score'], linestyle = 'None')
    plt.title('R2 Score vs. Bias Strength for real world data {} query plans'.format(results["train_size"].unique()[0]))
    plt.xlabel('Observational Bias Strength')
    plt.ylabel('R2 Score')
    plt.savefig("{}/{}.png".format(plot_dir, plot_name))
    # plt.close()
# if plan changes the compare plans returns false else true
def compare_query_plans(query_id_1, treatment_id_1, query_id_2, treatment_id_2, data_folder_name, verbose = False):
    config_name = "config.json"
    dbname = "mathso"
    
    config_path = "{}/queries/jsons".format(DATA_DIR)
    config_file_path = "{}/{}".format(config_path, config_name)
    config = generate_basic_config(config_file_path, dbname, data_folder_name)
    plans = []
    for query_id, treatment_id in [(query_id_1, treatment_id_1), (query_id_2, treatment_id_2)]:
        query_id = query_id
        treatment_id = treatment_id
        index_level, memory_level, page_cost_level = get_indiv_treatement_idx(treatment_id)
        run_id = 0
        
        path = "{}/queries/data/{}/post_processed_queries/index_{}_memory_{}_page_{}/postgres_query_{}_{}_{}.json".format(DATA_DIR, data_folder_name, index_level, memory_level, page_cost_level, query_id, treatment_id, run_id)
        if not os.path.exists(path):
            print("Path {} doesn't exist".format(path))
            return None
        query_json = json.load(open(path, "r"))
        query_plan = query_json["json_result"][0]["Plan"]
        plan_explainer = PlanExplainer(query_id, treatment_id, run_id, query_plan, set_total_time =False)

        if verbose:
            print(plan_explainer.print_tree(plan_explainer.plan_tree, print_keys=[""]))
            print(path)
        plans.append(plan_explainer.plan_tree)
    
    # compare plans
    # print("Comparing plans")
    return plan_explainer.compare_plans(plans[0], plans[1])

def plot_gt_ce_df_orig(df, df_sampled, plot_folder, prob_value, plot_obs = True):
    # # print(df[['query_id', 'treatment_id', 
    # #    'Sort_sort_method', 'Aggregate_strategy','total_execution_time']])
    treatments = list(df['treatment_id'].unique())
    treatment_str = '_'.join([str(treatment) for treatment in treatments])

    # if "Sort_sort_method" in df.columns:
    #     sort_method_names = {0: 'external merge', 1: 'external sort', 2: 'quicksort', 3: 'top-N heapsort'}
    #     df["Sort_sort_method"] = df["Sort_sort_method"].apply(lambda x: sort_method_names[x])
    
    # if "Aggregate_strategy" in df.columns:
    #     aggregate_strategy = {0: 'Hashed', 1: 'Plain', 2: 'Sorted'}
    #     df["Aggregate_strategy"] = df["Aggregate_strategy"].apply(lambda x: aggregate_strategy[x])
    
    plt.figure(figsize=(10, 10))
    # make melted dataframe with different execution times for different memory levels
    df_melted = df.melt(id_vars=["query_id", "treatment_id"], value_vars=["total_execution_time", "Sort_execution_time", "Aggregate_execution_time", "Hash_execution_time", "Hash Join_execution_time", "Seq Scan_execution_time"], var_name="operation", value_name="execution_time")
    df_melted["operation"] = df_melted["operation"].str.replace("_execution_time", "")

    if plot_obs:
        df_sampled_melted = df_sampled.melt(id_vars=["query_id", "treatment_id"], value_vars=["total_execution_time", "Sort_execution_time", "Aggregate_execution_time", "Hash_execution_time", "Hash Join_execution_time", "Seq Scan_execution_time"], var_name="operation", value_name="execution_time")
        df_sampled_melted["operation"] = df_sampled_melted["operation"].str.replace("_execution_time", "")

        # combine df_melted and df_sampled_melted
        df_melted['sampled'] = "random"
        df_sampled_melted['sampled'] = "observational"
        
        df_melted = pd.concat([df_melted, df_sampled_melted])
        # combine treatment and sampled
        df_melted["treatment_sampled"] = df_melted.apply(lambda x: "{}_{}".format(x['treatment_id'], x['sampled']), axis = 1)

        # sort by treatment sampled
        df_melted = df_melted.sort_values(by=['treatment_sampled'])

    # plot boxplot with different sampling strategies side by side for different operations
    # x = operation 
    # y = execution time
    # hue = treatment
    # col = sampled
    if plot_obs:
        sns.boxplot(x="operation", y="execution_time", hue="treatment_sampled", data=df_melted)
    else:
        sns.boxplot(x="operation", y="execution_time", hue="treatment_id", data=df_melted)
    

    
    plt.title("Execution time for different operations for {} queries".format(len(df['query_id'].unique())))
    plot_dir = 'plots/{}/{}'.format(plot_folder, "ground_truth_plots")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plt.savefig("{}/boxplot_experimental_estimates_{}_prob_{}.png".format(plot_dir, treatment_str, prob_value))
    plt.figure(figsize=(10, 10))

    # plot count of different sort methods
    if 3 in treatments or 6 in treatments:
        if "Sort_sort_method" in df.columns:
            sns.countplot(x="treatment_id", hue="Sort_sort_method", data=df)
            plt.title("Sort Sort Method for different treatments for {} queries".format(len(df['query_id'].unique())))  
            # sns.barplot(x="treatment_id", y="Sort_sort_method",  data=df)
            plt.savefig("plots/{}/barplot_sort_method_{}.png".format(plot_folder, treatment_str))

        # aggregate strategy
        if "Aggregate_strategy" in df.columns:
            plt.figure(figsize=(10, 10))
            sns.countplot(x="treatment_id", hue="Aggregate_strategy", data=df)
            plt.title("Aggregate Strategy for different treatments for {} queries".format(len(df['query_id'].unique())))
            plt.savefig("plots/{}/barplot_aggregate_strategy_{}.png".format(plot_folder, treatment_str))

        if "Hash_hash_buckets" in df.columns:
            plt.figure(figsize=(10, 10))
            sns.boxplot(x="treatment_id", y="Hash_hash_buckets", data=df)
            plt.title("Hash Buckets for different treatments for {} queries".format(len(df['query_id'].unique())))
            plt.savefig("plots/{}/boxplot_hash_buckets_{}.png".format(plot_folder, treatment_str))
       
    if 9 in treatments or 18 in treatments or 24 in treatments:
        # plot counts of index scan and seq scan
        plt.figure(figsize=(10, 10))
        sns.countplot(x="treatment_id", y = "num_Index Scan", data=df)
        sns.countplot(x="treatment_id", y = "num_Seq Scan", data=df)
        plt.title("Index Scan Conditional for different treatments for {} queries".format(len(df['query_id'].unique())))
        plt.savefig("plots/{}/barplot_index_scan_conditional_{}.png".format(plot_folder, treatment_str))


def ground_truth_ce(df, df_cf, treatment, outcome, log_transform = True):
    query_ids = df['query_id']
    treatment_ids = df[treatment]
    y = df[outcome]

    # df_cf[treatment] = df_cf[treatment].apply(lambda x: -1 if x == 0 else x)
    cf_query_ids = df_cf['query_id']
    cf_treatment_ids = df_cf[treatment]
    y_cf = df_cf[outcome]
    observed_y = pd.DataFrame( {'query_id': query_ids, 'treatment_id_obs': treatment_ids, 'y_obs_gt': y})
    # obs_predictions = observed_y.copy()
    # reverse log transform
    if log_transform:
        observed_y['y_obs_gt'] = observed_y['y_obs_gt'].apply(lambda x: 10**x - 1)
    observed_y = observed_y.groupby(['query_id', 'treatment_id_obs']).sum()
    # reset index
    observed_y = observed_y.reset_index()
    # set query id as index
    observed_y = observed_y.set_index('query_id')

    observed_y_cf = pd.DataFrame( {'query_id': cf_query_ids, 'treatment_id_cf': cf_treatment_ids, 'y_cf_gt': y_cf})
    
    # reverse log transform
    if log_transform:
        observed_y_cf['y_cf_gt'] = observed_y_cf['y_cf_gt'].apply(lambda x: 10**x - 1)
    # cf_predictions = observed_y_cf.copy()
    observed_y_cf = observed_y_cf.groupby(['query_id', 'treatment_id_cf']).sum()
    # reset index
    observed_y_cf = observed_y_cf.reset_index()
    # set query id as index
    observed_y_cf = observed_y_cf.set_index('query_id')


    # merge observed_y and observed_y_cf
    
    observed_y = pd.merge(observed_y, observed_y_cf, on = ['query_id'], how = 'outer')
    # if treatment_id_cf is nan, set it to 1 - treatment_id_obs
    observed_y['treatment_id_cf'] = observed_y.apply(lambda x: 1 - x['treatment_id_obs'] if np.isnan(x['treatment_id_cf']) else x['treatment_id_cf'], axis = 1)
    observed_y['treatment_id_obs'] = observed_y.apply(lambda x: 1 - x['treatment_id_cf'] if np.isnan(x['treatment_id_obs']) else x['treatment_id_obs'], axis = 1)
    # fill na with 0
    observed_y = observed_y.fillna(0)



    # compute ce
    observed_y['y1_gt'] = observed_y.apply(lambda x: x['y_obs_gt'] if x['treatment_id_obs'] == 1 else x['y_cf_gt'], axis = 1)
    observed_y['y0_gt'] = observed_y.apply(lambda x: x['y_obs_gt'] if x['treatment_id_obs'] == 0 else x['y_cf_gt'], axis = 1)
    observed_y['gt_ce'] = observed_y['y1_gt'] - observed_y['y0_gt']

    # log transform
    if log_transform:
        observed_y['y1_gt'] = observed_y['y1_gt'].apply(lambda x: np.log10(x + 1))
        observed_y['y0_gt'] = observed_y['y0_gt'].apply(lambda x: np.log10(x + 1))
        observed_y['gt_ce'] = observed_y['gt_ce'].apply(lambda x: np.log10(abs(x) + 1) * np.sign(x))
        observed_y["y_obs_gt"] = observed_y["y_obs_gt"].apply(lambda x: np.log10(x + 1))
        observed_y["y_cf_gt"] = observed_y["y_cf_gt"].apply(lambda x: np.log10(x + 1))
    gt_ce = observed_y[['treatment_id_obs', 'gt_ce',  'y1_gt', 'y0_gt']]
    gt_ce.columns = ['t',  'gt_ce',  'y1_gt', 'y0_gt']
    gt_ce = gt_ce.reset_index()

    # all_predictions = pd.concat([obs_predictions, cf_predictions])
    return gt_ce
    
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def create_model(input_dim, output_dim):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, input_dim=input_dim, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(output_dim)
    ])
    
    optimizer = tf.keras.optimizers.Adam(0.01)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    
    return model

def rf_ce(df, df_cf, treatment, outcome, covariates, plot_folder = 'plots', operator = 'high_level', log_transform = True, save_plots = False, model_type = "random_forest"):
    # treatment_ids = [0, 3, 6]
    # treatment = ['treatment_id']
    # outcome = ['total_execution_time']
    # covariates = ['num_Seq Scan', 'num_Hash', 'num_Hash Join', 'num_Sort',
    #    'num_Aggregate', 'num_complex_ops', 'Seq Scan_input_rows',
    #    'Seq Scan_output_rows', 'Hash_input_rows', 'Hash_output_rows',
    #    'Hash Join_left_output_rows', 'Hash Join_right_output_rows',
    #    'Hash Join_output_rows', 'Sort_input_rows', 'Sort_output_rows',
    #    'Aggregate_input_rows', 'Aggregate_output_rows']
    # df = df_sampled

    # replace treatment_id 0 with -1
    # df[treatment] = df[treatment].apply(lambda x: -1 if x == 0 else x)
    query_ids = df['query_id']
    treatment_ids = df['treatment_id']
    X = df[covariates + [treatment]]
    y = df[outcome]
    print(X.shape, y.shape)
    print(len(covariates))

    
    # df_cf[treatment] = df_cf[treatment].apply(lambda x: -1 if x == 0 else x)
    cf_query_ids = df_cf['query_id']
    cf_treatment_ids = df_cf['treatment_id']
    X_cf = df_cf[covariates + [treatment]]
    y_cf = df_cf[outcome]

    X_cov = X[covariates].values

    if model_type == "neural_network":
        model = create_model(X.shape[1], 1)
    elif model_type == "random_forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == "tnet":
        model = TNet()
    elif model_type == "snet":
        model = SNet(penalty_orthogonal=0.01)
    if model_type in ["tnet", "snet"]:
        
        model.fit(X_cov, y.values, X[treatment].values)
        cate_pred, y0_pred, y1_pred = model.predict(X_cov, return_po=True)
        y0_pred = y0_pred.squeeze()
        y1_pred = y1_pred.squeeze()
        y_obs = []
        for i, t in enumerate(X[treatment].values):
            if t == 0:
                y_obs.append(y0_pred[i])
            else:
                y_obs.append(y1_pred[i])
        # y_obs = np.array(y_obs)
        # # squeeze
        # y_obs = y_obs.squeeze()
    else:
        # if model_type == "neural_network":
        #     # add treatment to X
        #     X = np.concatenate([X_cov, X[treatment].values.reshape(-1, 1)], axis = 1)
        model.fit(X, y)
        X_obs = X.copy()
        y_obs = model.predict(X_obs)
        # squeeze
        y_obs = y_obs.squeeze()

    # counterfactuals
    # print(query_ids.shape, treatment_ids.shape, y_obs.shape, y.shape)
    observed_y = pd.DataFrame( {'query_id': query_ids, 'treatment_id_obs': treatment_ids, 'y_obs': y_obs, 'y_obs_gt': y})
    obs_predictions = observed_y.copy()
    

    # reverse log transform
    if log_transform:
        observed_y['y_obs'] = observed_y['y_obs'].apply(lambda x: 10**x - 1)
        observed_y['y_obs_gt'] = observed_y['y_obs_gt'].apply(lambda x: 10**x - 1)
    observed_y = observed_y.groupby(['query_id', 'treatment_id_obs']).sum()
    # reset index
    observed_y = observed_y.reset_index()
    # set query id as index
    observed_y = observed_y.set_index('query_id')
    X_cf_obs = X_cf.copy()
    if model_type in ["tnet", "snet"]:
        cate_pred, y0_pred, y1_pred = model.predict(X_cf_cov, return_po=True)
        y0_pred = y0_pred.squeeze()
        y1_pred = y1_pred.squeeze()
        y_cf_obs = []
        for i, t in enumerate(X_cf[treatment].values):
            if t == 0:
                y_cf_obs.append(y0_pred[i])
            else:
                y_cf_obs.append(y1_pred[i])
        
    else:    
        # if model_type == "neural_network":
        #     X_cf_obs = np.concatenate([X_cf_cov, X_cf[treatment].values.reshape(-1, 1)], axis = 1)
        y_cf_obs = model.predict(X_cf_obs)
        # squeeze
        y_cf_obs = y_cf_obs.squeeze()
    # print(y_cf_obs.shape, y_cf.shape)
    observed_y_cf = pd.DataFrame( {'query_id': cf_query_ids, 'treatment_id_cf': cf_treatment_ids, 'y_cf': y_cf_obs, 'y_cf_gt': y_cf})
    cf_predictions = observed_y_cf.copy()
    # reverse log transform
    if log_transform:
        observed_y_cf['y_cf'] = observed_y_cf['y_cf'].apply(lambda x: 10**x - 1)
        observed_y_cf['y_cf_gt'] = observed_y_cf['y_cf_gt'].apply(lambda x: 10**x - 1)
    observed_y_cf = observed_y_cf.groupby(['query_id', 'treatment_id_cf']).sum()
    # reset index
    observed_y_cf = observed_y_cf.reset_index()
    # set query id as index
    observed_y_cf = observed_y_cf.set_index('query_id')


    # merge observed_y and observed_y_cf such that query ids not in both have 0 for cf
    observed_y = pd.merge(observed_y, observed_y_cf, on = ['query_id'], how = 'outer')
    # fill na with 0
    # if treatment_id_cf is nan, set it to 1 - treatment_id_obs
    observed_y['treatment_id_cf'] = observed_y.apply(lambda x: 1 - x['treatment_id_obs'] if np.isnan(x['treatment_id_cf']) else x['treatment_id_cf'], axis = 1)
    observed_y['treatment_id_obs'] = observed_y.apply(lambda x: 1 - x['treatment_id_cf'] if np.isnan(x['treatment_id_obs']) else x['treatment_id_obs'], axis = 1)
    # fill na with 0
    observed_y = observed_y.fillna(0)
    observed_y = observed_y.fillna(0)

    # compute ce
    observed_y['y1_pred'] = observed_y.apply(lambda x: x['y_obs'] if x['treatment_id_obs'] == 1 else x['y_cf'], axis = 1)
    observed_y['y0_pred'] = observed_y.apply(lambda x: x['y_obs'] if x['treatment_id_obs'] == 0 else x['y_cf'], axis = 1)

    observed_y['y1_gt'] = observed_y.apply(lambda x: x['y_obs_gt'] if x['treatment_id_obs'] == 1 else x['y_cf_gt'], axis = 1)
    observed_y['y0_gt'] = observed_y.apply(lambda x: x['y_obs_gt'] if x['treatment_id_obs'] == 0 else x['y_cf_gt'], axis = 1)

    observed_y['pred_ce'] = observed_y['y1_pred'] - observed_y['y0_pred']
    observed_y['gt_ce'] = observed_y['y1_gt'] - observed_y['y0_gt']

    # log transform
    if log_transform:
        observed_y['y1_pred'] = observed_y['y1_pred'].apply(lambda x: np.log10(x + 1))
        observed_y['y0_pred'] = observed_y['y0_pred'].apply(lambda x: np.log10(x + 1))
        observed_y['pred_ce'] = observed_y['pred_ce'].apply(lambda x: np.log10(abs(x) + 1) * np.sign(x))
        observed_y['y1_gt'] = observed_y['y1_gt'].apply(lambda x: np.log10(x + 1))
        observed_y['y0_gt'] = observed_y['y0_gt'].apply(lambda x: np.log10(x + 1))
        observed_y['gt_ce'] = observed_y['gt_ce'].apply(lambda x: np.log10(abs(x) + 1) * np.sign(x))
        observed_y["y_obs"] = observed_y["y_obs"].apply(lambda x: np.log10(x + 1))
        observed_y["y_cf"] = observed_y["y_cf"].apply(lambda x: np.log10(x + 1))
        observed_y["y_obs_gt"] = observed_y["y_obs_gt"].apply(lambda x: np.log10(x + 1))
        observed_y["y_cf_gt"] = observed_y["y_cf_gt"].apply(lambda x: np.log10(x + 1))
    pred_ce = observed_y[['treatment_id_obs', 'pred_ce', 'gt_ce', 'y1_pred', 'y0_pred', 'y1_gt', 'y0_gt']]
    pred_ce.columns = ['t', 'pred_ce', 'gt_ce', 'y1_pred', 'y0_pred', 'y1_gt', 'y0_gt']
    pred_ce = pred_ce.reset_index()
    all_predictions = pd.concat([obs_predictions, cf_predictions])
    print(pred_ce.head())
    return pred_ce, all_predictions


def CATENets(df, df_cf, treatment, outcome, covariates, plot_folder = 'plots', operator = 'high_level', log_transform = True, save_plots = False, model_type = "TNet"):
    query_ids = df['query_id']
    treatment_ids = df['treatment_id']
    X = df[covariates + [treatment]]
    y = df[outcome]

    # save distribution of X and y
    X_cov = X[covariates + [treatment]]
    # append y
    X_cov['y'] = y
    
    if save_plots:
        # plot distribution of all covariates and outcome for different treatments
        plt.figure(figsize=(10, 10))
        # have kdplot for each covariate and outcome per treatment
        for covariate in covariates + ['y']:
            t0 = X_cov[X_cov[treatment] == 0][covariate]
            t1 = X_cov[X_cov[treatment] == 1][covariate]
            plt.figure(figsize=(10, 10))
            sns.kdeplot(t0, label = 'treatment 0 {}'.format(covariate))
            sns.kdeplot(t1, label = 'treatment 1 {}'.format(covariate))
            plt.xlabel(covariate)
            plt.ylabel('Frequency')
            plt.legend()
            plt.savefig("plots/{}/{}_{}_kdeplot_covariates.png".format(plot_folder, operator, covariate))

        # if operator == 'sort':
        #     print(t1.describe())
        #     print(t0.describe())
        #     # print("T1 {} T0 {}".format(t1.median(), t0.median()))
            
    # sns.boxplot(x="treatment_id", y="value", hue="variable", data=pd.melt(X_cov, id_vars=['y', 'treatment_id']))
    # plt.savefig("plots/{}/{}_boxplot_covariates.png".format(plot_folder, operator))
    

    # df_cf[treatment] = df_cf[treatment].apply(lambda x: -1 if x == 0 else x)
    cf_query_ids = df_cf['query_id']
    cf_treatment_ids = df_cf['treatment_id']
    X_cf = df_cf[covariates + [treatment]]
    y_cf = df_cf[outcome]

    # save distribution of X and y
    X_cf_cov = X_cf[covariates]
    # append y
    X_cf_cov['y'] = y_cf
    # plt.figure(figsize=(10, 10))
    # sns.boxplot(data=X_cf_cov)
    # plt.savefig("plots/{}/{}_boxplot_covariates_cf.png".format(plot_folder, operator))

    if model_type == "neural_network":
        model = create_model(X.shape[1], 1)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # counterfactuals
    X_obs = X.copy()
    y_obs = model.predict(X_obs)
    # squeeze
    y_obs = y_obs.squeeze()
    observed_y = pd.DataFrame( {'query_id': query_ids, 'treatment_id_obs': treatment_ids, 'y_obs': y_obs, 'y_obs_gt': y})
    obs_predictions = observed_y.copy()
    

    # reverse log transform
    if log_transform:
        observed_y['y_obs'] = observed_y['y_obs'].apply(lambda x: 10**x - 1)
        observed_y['y_obs_gt'] = observed_y['y_obs_gt'].apply(lambda x: 10**x - 1)
    observed_y = observed_y.groupby(['query_id', 'treatment_id_obs']).sum()
    # reset index
    observed_y = observed_y.reset_index()
    # set query id as index
    observed_y = observed_y.set_index('query_id')

    X_cf_obs = X_cf.copy()
    y_cf_obs = model.predict(X_cf_obs)
    # squeeze
    y_cf_obs = y_cf_obs.squeeze()
    observed_y_cf = pd.DataFrame( {'query_id': cf_query_ids, 'treatment_id_cf': cf_treatment_ids, 'y_cf': y_cf_obs, 'y_cf_gt': y_cf})
    cf_predictions = observed_y_cf.copy()
    # reverse log transform
    if log_transform:
        observed_y_cf['y_cf'] = observed_y_cf['y_cf'].apply(lambda x: 10**x - 1)
        observed_y_cf['y_cf_gt'] = observed_y_cf['y_cf_gt'].apply(lambda x: 10**x - 1)
    observed_y_cf = observed_y_cf.groupby(['query_id', 'treatment_id_cf']).sum()
    # reset index
    observed_y_cf = observed_y_cf.reset_index()
    # set query id as index
    observed_y_cf = observed_y_cf.set_index('query_id')


    # merge observed_y and observed_y_cf such that query ids not in both have 0 for cf
    observed_y = pd.merge(observed_y, observed_y_cf, on = ['query_id'], how = 'outer')
    # fill na with 0
    # if treatment_id_cf is nan, set it to 1 - treatment_id_obs
    observed_y['treatment_id_cf'] = observed_y.apply(lambda x: 1 - x['treatment_id_obs'] if np.isnan(x['treatment_id_cf']) else x['treatment_id_cf'], axis = 1)
    observed_y['treatment_id_obs'] = observed_y.apply(lambda x: 1 - x['treatment_id_cf'] if np.isnan(x['treatment_id_obs']) else x['treatment_id_obs'], axis = 1)
    # fill na with 0
    observed_y = observed_y.fillna(0)
    observed_y = observed_y.fillna(0)

    # compute ce
    observed_y['y1_pred'] = observed_y.apply(lambda x: x['y_obs'] if x['treatment_id_obs'] == 1 else x['y_cf'], axis = 1)
    observed_y['y0_pred'] = observed_y.apply(lambda x: x['y_obs'] if x['treatment_id_obs'] == 0 else x['y_cf'], axis = 1)

    observed_y['y1_gt'] = observed_y.apply(lambda x: x['y_obs_gt'] if x['treatment_id_obs'] == 1 else x['y_cf_gt'], axis = 1)
    observed_y['y0_gt'] = observed_y.apply(lambda x: x['y_obs_gt'] if x['treatment_id_obs'] == 0 else x['y_cf_gt'], axis = 1)

    observed_y['pred_ce'] = observed_y['y1_pred'] - observed_y['y0_pred']
    observed_y['gt_ce'] = observed_y['y1_gt'] - observed_y['y0_gt']

    # log transform
    if log_transform:
        observed_y['y1_pred'] = observed_y['y1_pred'].apply(lambda x: np.log10(x + 1))
        observed_y['y0_pred'] = observed_y['y0_pred'].apply(lambda x: np.log10(x + 1))
        observed_y['pred_ce'] = observed_y['pred_ce'].apply(lambda x: np.log10(abs(x) + 1) * np.sign(x))
        observed_y['y1_gt'] = observed_y['y1_gt'].apply(lambda x: np.log10(x + 1))
        observed_y['y0_gt'] = observed_y['y0_gt'].apply(lambda x: np.log10(x + 1))
        observed_y['gt_ce'] = observed_y['gt_ce'].apply(lambda x: np.log10(abs(x) + 1) * np.sign(x))
        observed_y["y_obs"] = observed_y["y_obs"].apply(lambda x: np.log10(x + 1))
        observed_y["y_cf"] = observed_y["y_cf"].apply(lambda x: np.log10(x + 1))
        observed_y["y_obs_gt"] = observed_y["y_obs_gt"].apply(lambda x: np.log10(x + 1))
        observed_y["y_cf_gt"] = observed_y["y_cf_gt"].apply(lambda x: np.log10(x + 1))
    pred_ce = observed_y[['treatment_id_obs', 'pred_ce', 'gt_ce', 'y1_pred', 'y0_pred', 'y1_gt', 'y0_gt']]
    pred_ce.columns = ['t', 'pred_ce', 'gt_ce', 'y1_pred', 'y0_pred', 'y1_gt', 'y0_gt']
    pred_ce = pred_ce.reset_index()
    all_predictions = pd.concat([obs_predictions, cf_predictions])
    return pred_ce, all_predictions


# Two issues with double ML (1) We transform the outcome variable to log space and then predict the outcome variable. (2) Some features change in the counterfactual.
# Best way is to study the properties outside the query execution domain and then apply whatever general principles are here. 
def double_ml(df, df_cf, treatment, outcome, covariates, plot_folder = 'plots', operator = 'high_level', log_transform = True, save_plots = False):
    query_ids = df['query_id']
    treatment_ids = df['treatment_id']
    X = df[covariates]
    T = df[treatment]
    y = df[outcome]

    cf_query_ids = df_cf['query_id']
    cf_treatment_ids = df_cf['treatment_id']
    X_cf = df_cf[covariates]
    T_cf = df_cf[treatment]
    y_cf = df_cf[outcome]

    # reverse log-transform y and y_cf
    if log_transform:
        y = y.apply(lambda x: 10**x - 1)
        y_cf = y_cf.apply(lambda x: 10**x - 1)

    est = CausalForestDML(model_y=GradientBoostingRegressor(),
                      model_t=GradientBoostingRegressor())
    est.fit(y, T, X=X, W=X)
    t0 = np.zeros_like(T)
    t1 = np.ones_like(T)
    point = est.effect(X, T0=t0, T1=t1)
    
    observed_y = pd.DataFrame( {'query_id': query_ids, 'treatment_id_obs': treatment_ids, 'y_obs_gt': y})
    
    
    observed_y = observed_y.groupby(['query_id', 'treatment_id_obs']).sum()
    # reset index
    observed_y = observed_y.reset_index()
    # set query id as index
    observed_y = observed_y.set_index('query_id')

    X_cf_obs = X_cf.copy()
    y_cf_obs = model.predict(X_cf_obs)
    observed_y_cf = pd.DataFrame( {'query_id': cf_query_ids, 'treatment_id_cf': cf_treatment_ids, 'y_cf': y_cf_obs, 'y_cf_gt': y_cf})
    cf_predictions = observed_y_cf.copy()
    # reverse log transform
    if log_transform:
        observed_y_cf['y_cf'] = observed_y_cf['y_cf'].apply(lambda x: 10**x - 1)
        observed_y_cf['y_cf_gt'] = observed_y_cf['y_cf_gt'].apply(lambda x: 10**x - 1)
    observed_y_cf = observed_y_cf.groupby(['query_id', 'treatment_id_cf']).sum()
    # reset index
    observed_y_cf = observed_y_cf.reset_index()
    # set query id as index
    observed_y_cf = observed_y_cf.set_index('query_id')


    # merge observed_y and observed_y_cf such that query ids not in both have 0 for cf
    observed_y = pd.merge(observed_y, observed_y_cf, on = ['query_id'], how = 'outer')
    # fill na with 0
    # if treatment_id_cf is nan, set it to 1 - treatment_id_obs
    observed_y['treatment_id_cf'] = observed_y.apply(lambda x: 1 - x['treatment_id_obs'] if np.isnan(x['treatment_id_cf']) else x['treatment_id_cf'], axis = 1)
    observed_y['treatment_id_obs'] = observed_y.apply(lambda x: 1 - x['treatment_id_cf'] if np.isnan(x['treatment_id_obs']) else x['treatment_id_obs'], axis = 1)
    # fill na with 0
    observed_y = observed_y.fillna(0)
    observed_y = observed_y.fillna(0)

    # compute ce
    observed_y['y1_pred'] = observed_y.apply(lambda x: x['y_obs'] if x['treatment_id_obs'] == 1 else x['y_cf'], axis = 1)
    observed_y['y0_pred'] = observed_y.apply(lambda x: x['y_obs'] if x['treatment_id_obs'] == 0 else x['y_cf'], axis = 1)

    observed_y['y1_gt'] = observed_y.apply(lambda x: x['y_obs_gt'] if x['treatment_id_obs'] == 1 else x['y_cf_gt'], axis = 1)
    observed_y['y0_gt'] = observed_y.apply(lambda x: x['y_obs_gt'] if x['treatment_id_obs'] == 0 else x['y_cf_gt'], axis = 1)

    observed_y['pred_ce'] = observed_y['y1_pred'] - observed_y['y0_pred']
    observed_y['gt_ce'] = observed_y['y1_gt'] - observed_y['y0_gt']

    # log transform
    if log_transform:
        observed_y['y1_pred'] = observed_y['y1_pred'].apply(lambda x: np.log10(x + 1))
        observed_y['y0_pred'] = observed_y['y0_pred'].apply(lambda x: np.log10(x + 1))
        observed_y['pred_ce'] = observed_y['pred_ce'].apply(lambda x: np.log10(abs(x) + 1) * np.sign(x))
        observed_y['y1_gt'] = observed_y['y1_gt'].apply(lambda x: np.log10(x + 1))
        observed_y['y0_gt'] = observed_y['y0_gt'].apply(lambda x: np.log10(x + 1))
        observed_y['gt_ce'] = observed_y['gt_ce'].apply(lambda x: np.log10(abs(x) + 1) * np.sign(x))
        observed_y["y_obs"] = observed_y["y_obs"].apply(lambda x: np.log10(x + 1))
        observed_y["y_cf"] = observed_y["y_cf"].apply(lambda x: np.log10(x + 1))
        observed_y["y_obs_gt"] = observed_y["y_obs_gt"].apply(lambda x: np.log10(x + 1))
        observed_y["y_cf_gt"] = observed_y["y_cf_gt"].apply(lambda x: np.log10(x + 1))
    pred_ce = observed_y[['treatment_id_obs', 'pred_ce', 'gt_ce', 'y1_pred', 'y0_pred', 'y1_gt', 'y0_gt']]
    pred_ce.columns = ['t', 'pred_ce', 'gt_ce', 'y1_pred', 'y0_pred', 'y1_gt', 'y0_gt']
    pred_ce = pred_ce.reset_index()
    all_predictions = pd.concat([obs_predictions, cf_predictions])
    return pred_ce, all_predictions


def rf_ce_all(df, df_cf, learned_structure, treatment,log_transform = True, model_type = "random_forest"):

    # query_ids = df['query_id']
    # treatment_ids = df['treatment_id']
    # X = df[covariates + [treatment]]
    # y = df[outcome]

    # # save distribution of X and y
    # X_cov = X[covariates + [treatment]]
    # # append y
    # X_cov['y'] = y
    
    # # df_cf[treatment] = df_cf[treatment].apply(lambda x: -1 if x == 0 else x)
    # cf_query_ids = df_cf['query_id']
    # cf_treatment_ids = df_cf['treatment_id']
    # X_cf = df_cf[covariates + [treatment]]
    # y_cf = df_cf[outcome]

    # # save distribution of X and y
    # X_cf_cov = X_cf[covariates]
    # # append y
    # X_cf_cov['y'] = y_cf
    # # plt.figure(figsize=(10, 10))
    # # sns.boxplot(data=X_cf_cov)
    # # plt.savefig("plots/{}/{}_boxplot_covariates_cf.png".format(plot_folder, operator))

    # think about this later
    observed_y = []
    observed_y_cf = []
    for k,v in learned_structure.items():
        y_i = df[k]
        y_cf_i = df_cf[k]
        X_i = df[v + [treatment]]
        X_cf_i = df_cf[v + [treatment]]
        if model_type == "BART":
            model = SklearnModel()
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_i, y_i)
        y_i_obs = model.predict(X_i)
        y_i_cf = model.predict(X_cf_i)
        y_i_obs = (10**y_i_obs) - 1
        y_i_cf = (10**y_i_cf) - 1
        y_1_obs_preds = pd.DataFrame({"query_id": df["query_id"], 'treatment_id_obs': df['treatment_id'], 'y_obs': y_i_obs, 'y_obs_gt': (10**y_i)-1})
        y_1_cf_preds = pd.DataFrame({"query_id": df_cf["query_id"], 'treatment_id_cf': df_cf['treatment_id'], "y_cf": y_i_cf, "y_cf_gt": (10**y_cf_i)-1})
        observed_y.append(y_1_obs_preds)
        observed_y_cf.append(y_1_cf_preds)
    observed_y = pd.concat(observed_y)
    observed_y_cf = pd.concat(observed_y_cf)
    # sum by query id
    
    

    # # reverse log transform
    # if log_transform:
    #     preds['predicted_et'] = preds['predicted_et'].apply(lambda x: np.log10(x + 1))
    #     preds['actual_et'] = preds['actual_et'].apply(lambda x: np.log10(x + 1))
    #     cf_preds['predicted_et'] = cf_preds['predicted_et'].apply(lambda x: np.log10(x + 1))
    #     cf_preds['actual_et'] = cf_preds['actual_et'].apply(lambda x: np.log10(x + 1))

    # # counterfactuals
    # X_obs = X.copy()
    # y_obs = model.predict(X_obs)
    # observed_y = pd.DataFrame( {'query_id': query_ids, 'treatment_id_obs': treatment_ids, 'y_obs': y_obs, 'y_obs_gt': y})
    # obs_predictions = observed_y.copy()
    

    # # reverse log transform
    # if log_transform:
    #     observed_y['y_obs'] = observed_y['y_obs'].apply(lambda x: 10**x - 1)
    #     observed_y['y_obs_gt'] = observed_y['y_obs_gt'].apply(lambda x: 10**x - 1)
    observed_y = observed_y.groupby(['query_id', 'treatment_id_obs']).sum()
    # reset index
    observed_y = observed_y.reset_index()
    # set query id as index
    observed_y = observed_y.set_index('query_id')

    # X_cf_obs = X_cf.copy()
    # y_cf_obs = model.predict(X_cf_obs)
    # observed_y_cf = pd.DataFrame( {'query_id': cf_query_ids, 'treatment_id_cf': cf_treatment_ids, 'y_cf': y_cf_obs, 'y_cf_gt': y_cf})
    # cf_predictions = observed_y_cf.copy()
    # # reverse log transform
    # if log_transform:
    #     observed_y_cf['y_cf'] = observed_y_cf['y_cf'].apply(lambda x: 10**x - 1)
    #     observed_y_cf['y_cf_gt'] = observed_y_cf['y_cf_gt'].apply(lambda x: 10**x - 1)
    observed_y_cf = observed_y_cf.groupby(['query_id', 'treatment_id_cf']).sum()
    # reset index
    observed_y_cf = observed_y_cf.reset_index()
    # set query id as index
    observed_y_cf = observed_y_cf.set_index('query_id')


    # merge observed_y and observed_y_cf such that query ids not in both have 0 for cf
    observed_y = pd.merge(observed_y, observed_y_cf, on = ['query_id'], how = 'outer')
    # fill na with 0
    # if treatment_id_cf is nan, set it to 1 - treatment_id_obs
    observed_y['treatment_id_cf'] = observed_y.apply(lambda x: 1 - x['treatment_id_obs'] if np.isnan(x['treatment_id_cf']) else x['treatment_id_cf'], axis = 1)
    observed_y['treatment_id_obs'] = observed_y.apply(lambda x: 1 - x['treatment_id_cf'] if np.isnan(x['treatment_id_obs']) else x['treatment_id_obs'], axis = 1)
    # fill na with 0
    observed_y = observed_y.fillna(0)
    observed_y = observed_y.fillna(0)

    # compute ce
    observed_y['y1_pred'] = observed_y.apply(lambda x: x['y_obs'] if x['treatment_id_obs'] == 1 else x['y_cf'], axis = 1)
    observed_y['y0_pred'] = observed_y.apply(lambda x: x['y_obs'] if x['treatment_id_obs'] == 0 else x['y_cf'], axis = 1)

    observed_y['y1_gt'] = observed_y.apply(lambda x: x['y_obs_gt'] if x['treatment_id_obs'] == 1 else x['y_cf_gt'], axis = 1)
    observed_y['y0_gt'] = observed_y.apply(lambda x: x['y_obs_gt'] if x['treatment_id_obs'] == 0 else x['y_cf_gt'], axis = 1)

    observed_y['pred_ce'] = observed_y['y1_pred'] - observed_y['y0_pred']
    observed_y['gt_ce'] = observed_y['y1_gt'] - observed_y['y0_gt']

    # log transform
    if log_transform:
        observed_y['y1_pred'] = observed_y['y1_pred'].apply(lambda x: np.log10(x + 1))
        observed_y['y0_pred'] = observed_y['y0_pred'].apply(lambda x: np.log10(x + 1))
        observed_y['pred_ce'] = observed_y['pred_ce'].apply(lambda x: np.log10(abs(x) + 1) * np.sign(x))
        observed_y['y1_gt'] = observed_y['y1_gt'].apply(lambda x: np.log10(x + 1))
        observed_y['y0_gt'] = observed_y['y0_gt'].apply(lambda x: np.log10(x + 1))
        observed_y['gt_ce'] = observed_y['gt_ce'].apply(lambda x: np.log10(abs(x) + 1) * np.sign(x))
        observed_y["y_obs"] = observed_y["y_obs"].apply(lambda x: np.log10(x + 1))
        observed_y["y_cf"] = observed_y["y_cf"].apply(lambda x: np.log10(x + 1))
        observed_y["y_obs_gt"] = observed_y["y_obs_gt"].apply(lambda x: np.log10(x + 1))
        observed_y["y_cf_gt"] = observed_y["y_cf_gt"].apply(lambda x: np.log10(x + 1))
    pred_ce = observed_y[['treatment_id_obs', 'pred_ce', 'gt_ce', 'y1_pred', 'y0_pred', 'y1_gt', 'y0_gt']]
    pred_ce.columns = ['t', 'pred_ce', 'gt_ce', 'y1_pred', 'y0_pred', 'y1_gt', 'y0_gt']
    pred_ce = pred_ce.reset_index()
    all_predictions = []
    return pred_ce, all_predictions

def rf_ce_0_1(df, treatment, outcome, covariates, log_transform = True, model_type = "random_forest"):
    # treatment_ids = [0, 3, 6]
    # treatment = ['treatment_id']
    # outcome = ['total_execution_time']
    # covariates = ['num_Seq Scan', 'num_Hash', 'num_Hash Join', 'num_Sort',
    #    'num_Aggregate', 'num_complex_ops', 'Seq Scan_input_rows',
    #    'Seq Scan_output_rows', 'Hash_input_rows', 'Hash_output_rows',
    #    'Hash Join_left_output_rows', 'Hash Join_right_output_rows',
    #    'Hash Join_output_rows', 'Sort_input_rows', 'Sort_output_rows',
    #    'Aggregate_input_rows', 'Aggregate_output_rows']
    # df = df_sampled

    # replace treatment_id 0 with -1
    # df[treatment] = df[treatment].apply(lambda x: -1 if x == 0 else x)
    query_ids = df['query_id']
    X = df[covariates + [treatment]]
    y = df[outcome]

    if model_type == "BART":
        model = SklearnModel()
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    X0 = X.copy()
    X0[treatment] = 0
    y0 = model.predict(X0)

    X1 = X.copy()
    X1[treatment] = 1
    y1 = model.predict(X1)

    pred_ce = []
    for index, row in X.iterrows():
        query_id = query_ids[index]
        if log_transform:
            y1_reverse = 10**y1[index] - 1
            y0_reverse = 10**y0[index] - 1
            ce = y1_reverse - y0_reverse
        else:
            y1_reverse = y1[index]
            y0_reverse = y0[index]
            ce = y1_reverse - y0_reverse
        pred_ce.append([query_id, row[treatment], y1_reverse, y0_reverse, ce])
    pred_ce = pd.DataFrame(pred_ce, columns = ['query_id', 'treatment_id', 'y1_pred', 'y0_pred', 'pred_ce'])

    # aggregate over query ids
    pred_ce = pred_ce.groupby(['query_id', 'treatment_id']).sum()
    pred_ce = pred_ce.reset_index()
    pred_ce['pred_ce'] = pred_ce['y1_pred'] - pred_ce['y0_pred']

    # log transform
    if log_transform:
        pred_ce['y1_pred'] = pred_ce['y1_pred'].apply(lambda x: np.log10(x + 1))
        pred_ce['y0_pred'] = pred_ce['y0_pred'].apply(lambda x: np.log10(x + 1))
        pred_ce['pred_ce'] = pred_ce['pred_ce'].apply(lambda x: np.log10(abs(x) + 1) * np.sign(x))
        
    return pred_ce

def get_sampled_from_gt_ce(gt_ce, covariates, treatment, outcome, query_treatment_ids = None):
    # for each row, sample a random treatment and compute the outcome
    sampled_ce = gt_ce.copy()
    cf_ce = gt_ce.copy()
    if query_treatment_ids is not None:
        sampled_ce = pd.merge(sampled_ce, query_treatment_ids, on = ['query_id'])
    else:
        sampled_ce[treatment] = np.random.randint(0, 2, len(sampled_ce))

    cf_ce[treatment] = 1 - sampled_ce['treatment_id']
    # y1 if treatment is 1, y0 if treatment is 0
    sampled_ce[outcome] = sampled_ce.apply(lambda x: x['y1'] if x[treatment] == 1 else x['y0'], axis = 1)
    cf_ce[outcome] = cf_ce.apply(lambda x: x['y1'] if x[treatment] == 1 else x['y0'], axis = 1)
    sampled_ce = sampled_ce[['query_id', treatment, outcome] + covariates]
    cf_ce = cf_ce[['query_id', treatment, outcome] + covariates]
    return sampled_ce, cf_ce

def plot_results(pred_ce, operator, plot_folder = 'plots', model_type = "random_forest"):
    print("Ground Truth CE vs. Predicted CE {} {} {}".format(operator, np.mean(pred_ce['gt_ce']), np.mean(pred_ce['pred_ce'])))
    # R2 score 2 decimal places
    
    print("R2 Score {} {:.2f}".format(operator, r2_score(pred_ce['gt_ce'], pred_ce['pred_ce'])))

    if not os.path.exists('plots/{}'.format(plot_folder)):
        os.makedirs('plots/{}'.format(plot_folder))
    
    # plot distribution of causal effects with density
    plt.figure(figsize=(10,10))
    sns.kdeplot(pred_ce['gt_ce'], label = 'Ground Truth CE')
    sns.kdeplot(pred_ce['pred_ce'], label = 'Predicted CE')
    plt.xlabel('Causal Effect')
    plt.ylabel('Frequency')
    plt.legend()
    plot_dir = 'plots/{}/{}'.format(plot_folder, "operator_plots")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plt.savefig('{}/causal_effect_estimation_{}_kdeplot.png'.format(plot_dir, operator))

    # scatter plot
    plt.figure(figsize=(10,10))
    plt.scatter(x=pred_ce['gt_ce'], y=pred_ce['pred_ce'])
    # 45 degree line with min and max values
    min_val = min(min(pred_ce['gt_ce']), min(pred_ce['pred_ce']))
    max_val = max(max(pred_ce['gt_ce']), max(pred_ce['pred_ce']))
    plt.plot([min_val, max_val], [min_val, max_val], color = 'black')
    
    plt.xlabel('Ground Truth CE')
    plt.ylabel('Predicted CE')
    # have R2 score in the title
    plt.title("R2 Score {:.2f} Operator {}".format(r2_score(pred_ce['gt_ce'], pred_ce['pred_ce']), operator))
    plt.savefig('{}/causal_effect_estimation_{}_scatter_{}.png'.format(plot_dir, operator, model_type))

def remove_outliers_from_ce(gt_ce, operator = 'high_level'):
    # remove outliers
    median = np.median(gt_ce['gt_ce'])
    std = np.std(gt_ce['gt_ce'])
    print("Median {} Std {}".format(median, std))

    # remove causal effects that are more than 3 std away from the median
    skip_query_ids = gt_ce[abs(gt_ce['gt_ce'] - median) > 2 * std]['query_id'].unique()
    gt_ce = gt_ce[abs(gt_ce['gt_ce'] - median) < 2 * std]
    return gt_ce, skip_query_ids


# define arguments 



def causal_effect_estimation(treatment_ids, data_folder_name, data_agg_type, sampling, num_trials, changed_plans, log_transform, biasing_covariate, experiment_type,noise_variances,remove_outliers,indep_sampling, filters = False, model_type = "random_forest"):
    # arguments
    # arguments
    plot_df = []
    tids_str = '_'.join([str(tid) for tid in treatment_ids])
    bias_strengths = list(np.arange(0, 11, 0.5))
    
    # directory to save plots
    
    data_dir = "all_csvs"

    plot_folder = '{}/{}/treatment_ids_{}/sampling_{}/changed_plans_{}_log_{}_biaising_covariate_{}'.format(experiment_type, data_folder_name, tids_str, sampling, changed_plans, log_transform, biasing_covariate)
    if not os.path.exists('plots/{}'.format(plot_folder)):
        os.makedirs('plots/{}'.format(plot_folder))

    results_folder = '{}/{}/treatment_ids_{}/sampling_{}/changed_plans_{}_log_{}_biaising_covariate_{}'.format(experiment_type, data_folder_name, tids_str, sampling, changed_plans, log_transform, biasing_covariate)
    if not os.path.exists('results/{}'.format(results_folder)):
        os.makedirs('results/{}'.format(results_folder))

    run_index_scan = False
    combined_seq_index = False
    num_indices_treatments = 0
    for t in treatment_ids:
        if t >= 9:
            num_indices_treatments += 1
    
    if num_indices_treatments == 0:
        run_index_scan = False
    elif num_indices_treatments == 1:
        combined_seq_index = True
    elif num_indices_treatments == 2:
        run_index_scan = True
            
    for trial in range(num_trials):
        for noise_variance in noise_variances:
           
            all_results = {}
            if sampling == "random_prob":
                identifiers = prob_values
            elif sampling == "observational":
                identifiers = bias_strengths
            for identifier in identifiers:
                if sampling == "observational":
                    bias_strength = identifier
                    prob = None
                    plot_obs = True
                else:
                    bias_strength = None
                    prob = identifier
                    plot_obs = False
                
                df_orig, df_sampled, df_cf = generate_high_level_observational_dataset(treatment_ids = treatment_ids, 
                                                                                    data_folder_name = data_folder_name, sampling = sampling, 
                                                                                    prob_value = prob, log_transform = log_transform, 
                                                                                    biasing_covariate = biasing_covariate, bias_strength = bias_strength, 
                                                                                    plot_folder = plot_folder, data_dir = data_dir, filters = filters)
                
                # print length of High Level Observational Dataset
                print("Length of High Level Observational Datasets {} {} {}".format(len(df_orig), len(df_sampled), len(df_cf)))
                
                # find query ids for which query plan structure doesn't change with external interventions
                query_ids = df_orig['query_id'].unique()
                treatments = df_orig['treatment_id'].unique()   
                diff_plans = 0
                changed_plan_query_ids = []
                unchanged_plan_query_ids = []
                for qid in query_ids:
                    if real:
                        flag = compare_query_plans(qid, treatments[0], qid, treatments[1], data_folder_name)
                    else:
                        flag = compare_query_plans_simulated(qid, treatments[0], qid, treatments[1])
                    if flag == None:
                        continue
                    if flag:
                        unchanged_plan_query_ids.append(qid)
                    else:
                        changed_plan_query_ids.append(qid)
                # save unchanged and changed plan query ids

                with open('{}/unchanged_plan_query_ids.txt'.format(results_folder), 'w') as f:
                    for item in unchanged_plan_query_ids:
                        f.write("%s\n" % item)
                with open('{}/changed_plan_query_ids.txt'.format(results_folder), 'w') as f:
                    for item in changed_plan_query_ids:
                        f.write("%s\n" % item)
                # candidate_query_ids = skip_query_ids
                print("Number of query ids with unchanged query plan structure {}".format(len(unchanged_plan_query_ids)))
                print("Number of query ids with changed query plan structure {}".format(len(changed_plan_query_ids)))
                print(df_orig["treatment_id"].value_counts())
                print("Length after filtering query ids {}".format(len(df_orig)))

                if changed_plans == "combined":
                    candidate_query_ids = changed_plan_query_ids + unchanged_plan_query_ids
                else:
                    if changed_plans:
                        candidate_query_ids = changed_plan_query_ids
                    else:
                        candidate_query_ids = unchanged_plan_query_ids

                    
                    if real:
                        df_orig = df_orig[df_orig['query_id'].isin(candidate_query_ids)]
                        df_sampled = df_sampled[df_sampled['query_id'].isin(candidate_query_ids)]
                        df_cf = df_cf[df_cf['query_id'].isin(candidate_query_ids)]
                        # reset index
                        df_orig = df_orig.reset_index(drop = True)
                        df_sampled = df_sampled.reset_index(drop = True)
                        df_cf = df_cf.reset_index(drop = True)

                
                print("Number of query ids {}".format(len(df_orig['query_id'].unique())))
                covariates = ['num_Seq Scan', 'num_Hash', 'num_Hash Join', 'num_Sort',
                'num_Aggregate', 'num_complex_ops', 'Seq Scan_input_rows',
                'Seq Scan_output_rows', 'Hash_input_rows', 'Hash_output_rows',
                'Hash Join_left_output_rows', 'Hash Join_right_output_rows',
                'Hash Join_output_rows', 'Sort_input_rows', 'Sort_output_rows',
                'Aggregate_input_rows', 'Aggregate_output_rows']
                treatment = 'treatment_id'
                outcome = 'total_execution_time'
                query_treatment_ids = df_sampled[['query_id', 'treatment_id']]

                treatment_id_map = {treatment_ids[i]: i for i in range(len(treatment_ids))}
                df_orig['treatment_id'] = df_orig['treatment_id'].map(treatment_id_map)
                df_sampled['treatment_id'] = df_sampled['treatment_id'].map(treatment_id_map)
                df_cf['treatment_id'] = df_cf['treatment_id'].map(treatment_id_map)
                
              
                gt_ce = ground_truth_ce(df_sampled, df_cf, treatment, outcome, log_transform=log_transform)

                if remove_outliers:
                    gt_ce, qids = remove_outliers_from_ce(gt_ce)
                    skip_query_ids.extend(qids)
                    


            

                # get seq scan data
                    # check if treatment 9 or 18 is present
                print("Combined Seq Index {}".format(combined_seq_index))
                print("Run Index Scan {}".format(run_index_scan))

                if combined_seq_index:
                    df_seq, df_seq_sampled, df_seq_cf = get_index_scan_obs_df( filters= filters, query_treatment_ids = query_treatment_ids, 
                                                                data_folder_name = data_folder_name, log_transform = log_transform, data_dir = data_dir, index_alone = False)
                else:
                    df_seq, df_seq_sampled, df_seq_cf = get_seq_obs_df( filters= filters, query_treatment_ids = query_treatment_ids, 
                                                                data_folder_name = data_folder_name, log_transform = log_transform, data_dir = data_dir)
                # seq scan
                df_seq['treatment_id'] = df_seq['treatment_id'].map(treatment_id_map)
                df_seq_sampled['treatment_id'] = df_seq_sampled['treatment_id'].map(treatment_id_map)
                df_seq_cf['treatment_id'] = df_seq_cf['treatment_id'].map(treatment_id_map)

                if 0 in treatment_ids and 18 in treatment_ids:
                    seq_covariates = ['input_rows', 'output_rows','scan_type']
                else:
                    seq_covariates = ['input_rows', 'output_rows']
                seq_treatment = 'treatment_id'
                seq_outcome = 'execution_time'
                print(len(df_seq), len(df_seq_sampled), len(df_seq_cf))
                gt_seq_ce = ground_truth_ce(df_seq_sampled, df_seq_cf,  seq_treatment, seq_outcome, log_transform=log_transform)
                if remove_outliers:
                    gt_seq_ce, qids = remove_outliers_from_ce(gt_seq_ce, operator = 'seq_scan')
                    skip_query_ids.extend(qids)

                if run_index_scan or not real:
                    df_index, df_index_sampled, df_index_cf = get_index_scan_obs_df( filters= filters, query_treatment_ids = query_treatment_ids,
                                                                                    data_folder_name = data_folder_name, log_transform = log_transform,
                                                                                    data_dir = data_dir, index_alone = True)
                
                    df_index['treatment_id'] = df_index['treatment_id'].map(treatment_id_map)
                    df_index_sampled['treatment_id'] = df_index_sampled['treatment_id'].map(treatment_id_map)
                    df_index_cf['treatment_id'] = df_index_cf['treatment_id'].map(treatment_id_map)
                    index_covariates = ['input_rows', 'output_rows']
                    index_treatment = 'treatment_id'
                    index_outcome = 'execution_time'
                    gt_index_ce = ground_truth_ce(df_index_sampled, df_index_cf, index_treatment, index_outcome, log_transform=log_transform)
                    if remove_outliers:
                        gt_index_ce, qids = remove_outliers_from_ce(gt_index_ce, operator = 'index_scan')
                        skip_query_ids.extend(qids)

                # get hash data
                df_hash, df_hash_sampled, df_hash_cf = get_hash_obs_df(filters= filters, query_treatment_ids = query_treatment_ids, 
                                                                    data_folder_name = data_folder_name, log_transform = log_transform, data_dir = data_dir)
                # # hash
                df_hash['treatment_id'] = df_hash['treatment_id'].map(treatment_id_map)
                df_hash_sampled['treatment_id'] = df_hash_sampled['treatment_id'].map(treatment_id_map)
                df_hash_cf['treatment_id'] = df_hash_cf['treatment_id'].map(treatment_id_map)
                hash_covariates = ['input_rows', 'output_rows']
                hash_treatment = 'treatment_id'
                hash_outcome = 'execution_time'
                gt_hash_ce  = ground_truth_ce(df_hash_sampled, df_hash_cf, hash_treatment, hash_outcome, log_transform=log_transform)
                if remove_outliers:
                    gt_hash_ce, qids = remove_outliers_from_ce(gt_hash_ce, operator = 'hash')
                    skip_query_ids.extend(qids)

                print(len(df_hash), len(df_hash_sampled))
                


                # get hash join data
                df_hash_join, df_hash_join_sampled, df_hash_join_cf = get_hash_join_obs_df(filters= filters, 
                                                                                        query_treatment_ids = query_treatment_ids, 
                                                                                        data_folder_name = data_folder_name, 
                                                                                        log_transform = log_transform, data_dir = data_dir)
                # hash join
                df_hash_join['treatment_id'] = df_hash_join['treatment_id'].map(treatment_id_map)
                df_hash_join_sampled['treatment_id'] = df_hash_join_sampled['treatment_id'].map(treatment_id_map)
                df_hash_join_cf['treatment_id'] = df_hash_join_cf['treatment_id'].map(treatment_id_map)
                hash_join_covariates = ['left_output_rows', 'right_output_rows', 'output_rows']
                hash_join_treatment = 'treatment_id'
                hash_join_outcome = 'execution_time'
                gt_hash_join_ce =  ground_truth_ce(df_hash_join_sampled, df_hash_join_cf, hash_join_treatment, hash_join_outcome, log_transform=log_transform)
                if remove_outliers:
                    gt_hash_join_ce, qids = remove_outliers_from_ce(gt_hash_join_ce, operator = 'hash_join')
                    skip_query_ids.extend(qids)

                print(len(df_hash_join), len(df_hash_join_sampled))

                # get sort data
                df_sort, df_sort_sampled, df_sort_cf = get_sort_obs_df( filters= filters, query_treatment_ids = query_treatment_ids, 
                                                                    data_folder_name = data_folder_name, log_transform = log_transform, data_dir = data_dir)
                # sort
                df_sort['treatment_id'] = df_sort['treatment_id'].map(treatment_id_map)
                df_sort_sampled['treatment_id'] = df_sort_sampled['treatment_id'].map(treatment_id_map)
                df_sort_cf['treatment_id'] = df_sort_cf['treatment_id'].map(treatment_id_map)
                sort_covariates = ['input_rows', 'output_rows']
                sort_treatment = 'treatment_id'
                sort_outcome = 'execution_time'
                gt_sort_ce = ground_truth_ce(df_sort_sampled, df_sort_cf, sort_treatment, sort_outcome, log_transform=log_transform)
                if remove_outliers:
                    gt_sort_ce, qids = remove_outliers_from_ce(gt_sort_ce, operator = 'sort')
                    skip_query_ids.extend(qids)
                print(len(df_sort), len(df_sort_sampled))

                # get aggregate data
                df_aggregate, df_aggregate_sampled, df_aggregate_cf = get_aggregate_obs_df(filters= filters, query_treatment_ids = query_treatment_ids,
                                                                                            data_folder_name = data_folder_name, log_transform = log_transform,
                                                                                            data_dir = data_dir)
                # aggregate
                df_aggregate['treatment_id'] = df_aggregate['treatment_id'].map(treatment_id_map)
                df_aggregate_sampled['treatment_id'] = df_aggregate_sampled['treatment_id'].map(treatment_id_map)
                df_aggregate_cf['treatment_id'] = df_aggregate_cf['treatment_id'].map(treatment_id_map)
                aggregate_covariates = ['input_rows', 'output_rows']
                aggregate_treatment = 'treatment_id'
                aggregate_outcome = 'execution_time'
                gt_aggregate_ce = ground_truth_ce(df_aggregate_sampled, df_aggregate_cf, aggregate_treatment, aggregate_outcome, log_transform=log_transform)
                if remove_outliers:
                    gt_aggregate_ce, qids = remove_outliers_from_ce(gt_aggregate_ce, operator = 'aggregate')
                    skip_query_ids.extend(qids)
                
                print("Length of skip query ids {}".format(len(skip_query_ids)))
                # print(skip_query_ids)
                # take intersection of query ids
                if remove_outliers:
                    # remove outliers by removing those rows from df
                    df_orig = df_orig[~df_orig['query_id'].isin(skip_query_ids)].reset_index(drop = True)
                    df_sampled = df_sampled[~df_sampled['query_id'].isin(skip_query_ids)].reset_index(drop = True)
                    df_cf = df_cf[~df_cf['query_id'].isin(skip_query_ids)].reset_index(drop = True)

                    df_seq = df_seq[~df_seq['query_id'].isin(skip_query_ids)].reset_index(drop = True)
                    df_seq_sampled = df_seq_sampled[~df_seq_sampled['query_id'].isin(skip_query_ids)].reset_index(drop = True)
                    df_seq_cf = df_seq_cf[~df_seq_cf['query_id'].isin(skip_query_ids)].reset_index(drop = True)

                    df_hash = df_hash[~df_hash['query_id'].isin(skip_query_ids)].reset_index(drop = True)
                    df_hash_sampled = df_hash_sampled[~df_hash_sampled['query_id'].isin(skip_query_ids)].reset_index(drop = True)
                    df_hash_cf = df_hash_cf[~df_hash_cf['query_id'].isin(skip_query_ids)].reset_index(drop = True)

                    if run_index_scan or not real:
                        df_index = df_index[~df_index['query_id'].isin(skip_query_ids)].reset_index(drop = True)
                        df_index_sampled = df_index_sampled[~df_index_sampled['query_id'].isin(skip_query_ids)].reset_index(drop = True)
                        df_index_cf = df_index_cf[~df_index_cf['query_id'].isin(skip_query_ids)].reset_index(drop = True)

                    df_hash_join = df_hash_join[~df_hash_join['query_id'].isin(skip_query_ids)].reset_index(drop = True)
                    df_hash_join_sampled = df_hash_join_sampled[~df_hash_join_sampled['query_id'].isin(skip_query_ids)].reset_index(drop = True)
                    df_hash_join_cf = df_hash_join_cf[~df_hash_join_cf['query_id'].isin(skip_query_ids)].reset_index(drop = True)

                    df_sort = df_sort[~df_sort['query_id'].isin(skip_query_ids)].reset_index(drop = True)
                    df_sort_sampled = df_sort_sampled[~df_sort_sampled['query_id'].isin(skip_query_ids)].reset_index(drop = True)
                    df_sort_cf = df_sort_cf[~df_sort_cf['query_id'].isin(skip_query_ids)].reset_index(drop = True)

                    df_aggregate = df_aggregate[~df_aggregate['query_id'].isin(skip_query_ids)].reset_index(drop = True)
                    df_aggregate_sampled = df_aggregate_sampled[~df_aggregate_sampled['query_id'].isin(skip_query_ids)].reset_index(drop = True)
                    df_aggregate_cf = df_aggregate_cf[~df_aggregate_cf['query_id'].isin(skip_query_ids)].reset_index(drop = True)

                    
                    print("Length of df_orig after removing outliers {}".format(len(df_orig)))  
                    print("Length of df_sampled after removing outliers {}".format(len(df_sampled)))
                    print("Length of df_cf after removing outliers {}".format(len(df_cf)))


                # print number of unique query ids
                print("Number of unique query ids in original dataset {} and sampled dataset {}".format(len(df_orig['query_id'].unique()), len(df_sampled['query_id'].unique()))
                )
            
                plot_gt_ce_df_orig(df_orig, df_sampled, plot_folder, prob_value = identifier, plot_obs = plot_obs)


                
                pred_ce, high_preds = rf_ce(df_sampled, df_cf, treatment, outcome, covariates, plot_folder = plot_folder, operator = 'high_level', log_transform = log_transform, model_type = model_type)
                plot_results(pred_ce, 'high_level', plot_folder = plot_folder, model_type = model_type)

                # pred_ce_bart, high_preds_bart = rf_ce(df_sampled, df_cf, treatment, outcome, covariates, plot_folder = plot_folder, operator = 'high_level', log_transform = log_transform, model_type = "BART")
                # plot_results(pred_ce_bart, 'high_level_bart', plot_folder = plot_folder)
                pred_ce_bart = np.zeros_like(pred_ce)
                
                pred_all_ce = np.zeros_like(pred_ce)
                pred_scm_ce = np.zeros_like(pred_ce)
                # seq scan
                pred_seq_ce, seq_preds = rf_ce(df_seq_sampled, df_seq_cf, seq_treatment, seq_outcome, seq_covariates, plot_folder= plot_folder, operator = 'seq_scan', log_transform = log_transform, model_type = model_type)
                plot_results(pred_seq_ce,'seq_scan', plot_folder = plot_folder, model_type = model_type)

                if run_index_scan:
                    pred_index_ce, index_preds = rf_ce(df_index_sampled, df_index_cf, index_treatment, index_outcome, index_covariates, plot_folder= plot_folder, operator = 'index_scan', log_transform = log_transform, model_type = model_type)
                    plot_results(pred_index_ce,'index_scan', plot_folder = plot_folder, model_type = model_type)
                else:
                    pred_index_ce = None
                    index_preds = None
                    df_index = None
                    df_index_sampled = None
                
                # hash
                pred_hash_ce, hash_preds = rf_ce(df_hash_sampled, df_hash_cf, hash_treatment, hash_outcome, hash_covariates, plot_folder= plot_folder, operator = 'hash', log_transform = log_transform, model_type = model_type)
                plot_results(pred_hash_ce, 'hash', plot_folder = plot_folder, model_type = model_type)

                # hash join
                pred_hash_join_ce, hj_preds = rf_ce(df_hash_join_sampled, df_hash_join_cf, hash_join_treatment, hash_join_outcome, hash_join_covariates, plot_folder= plot_folder, operator = 'hash_join', log_transform = log_transform, model_type = model_type)
                plot_results(pred_hash_join_ce, 'hash_join', plot_folder = plot_folder,model_type = model_type)

                # sort
                pred_sort_ce, sort_preds = rf_ce(df_sort_sampled, df_sort_cf, sort_treatment, sort_outcome, sort_covariates, plot_folder= plot_folder, operator = 'sort', log_transform = log_transform, model_type = model_type)
                plot_results(pred_sort_ce, 'sort', plot_folder = plot_folder, model_type = model_type)

                # aggregate
                pred_aggregate_ce, agg_preds = rf_ce(df_aggregate_sampled, df_aggregate_cf, aggregate_treatment, aggregate_outcome, aggregate_covariates, plot_folder= plot_folder, operator = 'aggregate', log_transform = log_transform, model_type = model_type)
                plot_results(pred_aggregate_ce, 'aggregate', plot_folder = plot_folder, model_type = model_type)

                features = ['query_id', 't', 'y1_pred', 'y0_pred', 'y1_gt', 'y0_gt', 'pred_ce', 'gt_ce', 'operator']
                
                pred_seq_ce['operator'] = 'Seq Scan'
                if run_index_scan or not real:
                    pred_index_ce['operator'] = 'Index Scan'
                pred_hash_ce['operator'] = 'Hash'
                pred_hash_join_ce['operator'] = 'Hash Join'
                pred_sort_ce['operator'] = 'Sort'
                pred_aggregate_ce['operator'] = 'Aggregate'
                pred_ce['operator'] = 'High Level'
                if run_index_scan or not real:
                    combined_predictions = pd.concat([pred_seq_ce[features], pred_index_ce[features], pred_hash_ce[features], pred_hash_join_ce[features], pred_sort_ce[features], pred_aggregate_ce[features]])
                else:
                    combined_predictions = pd.concat([pred_seq_ce[features], pred_hash_ce[features], pred_hash_join_ce[features], pred_sort_ce[features], pred_aggregate_ce[features]])
                combined_predictions = combined_predictions[['query_id', 't', 'y1_pred', 'y0_pred', 'y1_gt', 'y0_gt', 'pred_ce', 'gt_ce']]
                
                # convert y1 to 10**y1 - 1 and y0 to 10**y0 - 1
                if log_transform:
                    combined_predictions['y1_pred'] = combined_predictions['y1_pred'].apply(lambda x: 10**x - 1)
                    combined_predictions['y0_pred'] = combined_predictions['y0_pred'].apply(lambda x: 10**x - 1)
                    combined_predictions['y1_gt'] = combined_predictions['y1_gt'].apply(lambda x: 10**x - 1)
                    combined_predictions['y0_gt'] = combined_predictions['y0_gt'].apply(lambda x: 10**x - 1)
                    # sum y1 and y0 per query id
                combined_predictions = combined_predictions.groupby(['query_id', 't']).sum()
                combined_predictions = combined_predictions.reset_index()
                combined_predictions['pred_ce'] = combined_predictions['y1_pred'] - combined_predictions['y0_pred']
                combined_predictions['gt_ce'] = combined_predictions['y1_gt'] - combined_predictions['y0_gt']
                

                # log transform
                if log_transform:
                    combined_predictions['y1_pred'] = np.log10(combined_predictions['y1_pred'] + 1)
                    combined_predictions['y0_pred'] = np.log10(combined_predictions['y0_pred'] + 1)
                    combined_predictions['pred_ce'] = combined_predictions['pred_ce'].apply(lambda x: np.log10(abs(x) + 1) * np.sign(x))
                    combined_predictions['y1_gt'] = np.log10(combined_predictions['y1_gt'] + 1)
                    combined_predictions['y0_gt'] = np.log10(combined_predictions['y0_gt'] + 1)
                    combined_predictions['gt_ce'] = combined_predictions['gt_ce'].apply(lambda x: np.log10(abs(x) + 1) * np.sign(x))
                    
                combined_predictions = combined_predictions.sort_values(by = ['query_id'])
                
                combined_predictions = combined_predictions[combined_predictions['query_id'].isin(pred_ce['query_id'])]
                combined_predictions = combined_predictions.reset_index(drop = True)
                pred_ce = pred_ce[['query_id', 'y1_pred', 'y0_pred', 'y1_gt', 'y0_gt', 'pred_ce', 'gt_ce']]
                all_ce = pd.merge(combined_predictions, pred_ce, on = ['query_id'], suffixes = ('_combined', '_high_level'))
                
                # show 45 degree line
                min_val = min(min(all_ce['gt_ce_combined']), min(all_ce['pred_ce_combined']), min(all_ce['gt_ce_high_level']), min(all_ce['pred_ce_high_level']))
                max_val = max(max(all_ce['gt_ce_combined']), max(all_ce['pred_ce_combined']), max(all_ce['gt_ce_high_level']), max(all_ce['pred_ce_high_level']))
                plt.plot([min_val, max_val], [min_val, max_val], color = 'black')
                
                plt.savefig('plots/{}/causal_effect_estimation_combined_vs_high_level_{}.png'.format(plot_folder, identifier))


                if not indep_sampling:
                    plot_results(combined_predictions, 'component_level_model', plot_folder = plot_folder, model_type = model_type)

               
                results = pd.DataFrame()
                results['operator'] = ['Component Level Model', 'High Level', 'Seq Scan', 'Hash', 'Hash Join', 'Sort', 'Aggregate']
                if not changed_plans:
                    results['ground_truth'] = [np.mean(combined_predictions['gt_ce']), np.mean(pred_ce['gt_ce']), np.mean(pred_seq_ce['gt_ce']), np.mean(pred_hash_ce['gt_ce']), np.mean(pred_hash_join_ce['gt_ce']), np.mean(pred_sort_ce['gt_ce']), np.mean(pred_aggregate_ce['gt_ce'])]
                else:
                    results['ground_truth'] = [np.mean(combined_predictions['gt_ce']), np.mean(pred_ce['gt_ce']), np.mean(pred_seq_ce['gt_ce']), np.mean(pred_hash_ce['gt_ce']), np.mean(pred_hash_join_ce['gt_ce']), np.mean(pred_sort_ce['gt_ce']), np.mean(pred_aggregate_ce['gt_ce'])]
                results['predicted'] = [np.mean(combined_predictions['pred_ce']), np.mean(pred_ce['pred_ce']), np.mean(pred_seq_ce['pred_ce']), np.mean(pred_hash_ce['pred_ce']), np.mean(pred_hash_join_ce['pred_ce']), np.mean(pred_sort_ce['pred_ce']), np.mean(pred_aggregate_ce['pred_ce'])]
                if not changed_plans:
                    results['r2_score'] = [r2_score(combined_predictions['gt_ce'], combined_predictions['pred_ce']), r2_score(pred_ce['gt_ce'], pred_ce['pred_ce']), r2_score(pred_seq_ce['gt_ce'], pred_seq_ce['pred_ce']), r2_score(pred_hash_ce['gt_ce'], pred_hash_ce['pred_ce']), r2_score(pred_hash_join_ce['gt_ce'], pred_hash_join_ce['pred_ce']), r2_score(pred_sort_ce['gt_ce'], pred_sort_ce['pred_ce']), r2_score(pred_aggregate_ce['gt_ce'], pred_aggregate_ce['pred_ce'])]
                    results['mse'] = [mean_squared_error(combined_predictions['gt_ce'], combined_predictions['pred_ce']), mean_squared_error(pred_ce['gt_ce'], pred_ce['pred_ce']), mean_squared_error(pred_seq_ce['gt_ce'], pred_seq_ce['pred_ce']), mean_squared_error(pred_hash_ce['gt_ce'], pred_hash_ce['pred_ce']), mean_squared_error(pred_hash_join_ce['gt_ce'], pred_hash_join_ce['pred_ce']), mean_squared_error(pred_sort_ce['gt_ce'], pred_sort_ce['pred_ce']), mean_squared_error(pred_aggregate_ce['gt_ce'], pred_aggregate_ce['pred_ce'])]
                else:
                    results['r2_score'] = [r2_score(combined_predictions['gt_ce'], combined_predictions['pred_ce']), r2_score(pred_ce['gt_ce'], pred_ce['pred_ce']), r2_score(pred_seq_ce['gt_ce'], pred_seq_ce['pred_ce']), r2_score(pred_hash_ce['gt_ce'], pred_hash_ce['pred_ce']), r2_score(pred_hash_join_ce['gt_ce'], pred_hash_join_ce['pred_ce']), r2_score(pred_sort_ce['gt_ce'], pred_sort_ce['pred_ce']), r2_score(pred_aggregate_ce['gt_ce'], pred_aggregate_ce['pred_ce'])]
                    results['mse'] = [mean_squared_error(combined_predictions['gt_ce'], combined_predictions['pred_ce']), mean_squared_error(pred_ce['gt_ce'], pred_ce['pred_ce']), mean_squared_error(pred_seq_ce['gt_ce'], pred_seq_ce['pred_ce']), mean_squared_error(pred_hash_ce['gt_ce'], pred_hash_ce['pred_ce']), mean_squared_error(pred_hash_join_ce['gt_ce'], pred_hash_join_ce['pred_ce']), mean_squared_error(pred_sort_ce['gt_ce'], pred_sort_ce['pred_ce']), mean_squared_error(pred_aggregate_ce['gt_ce'], pred_aggregate_ce['pred_ce'])]
                results['train_size'] = [len(df_sampled), len(df_sampled), len(df_seq_sampled), len(df_hash_sampled), len(df_hash_join_sampled), len(df_sort_sampled), len(df_aggregate_sampled)]
                results["query_ids"] = [len(df_sampled['query_id'].unique()), len(df_sampled['query_id'].unique()), len(df_seq_sampled['query_id'].unique()), len(df_hash_sampled['query_id'].unique()), len(df_hash_join_sampled['query_id'].unique()), len(df_sort_sampled['query_id'].unique()), len(df_aggregate_sampled['query_id'].unique())]
                results["treatment_0_samples"] = [len(df_sampled[df_sampled['treatment_id'] == 0]), len(df_sampled[df_sampled['treatment_id'] == 0]), len(df_seq_sampled[df_seq_sampled['treatment_id'] == 0]), len(df_hash_sampled[df_hash_sampled['treatment_id'] == 0]), len(df_hash_join_sampled[df_hash_join_sampled['treatment_id'] == 0]), len(df_sort_sampled[df_sort_sampled['treatment_id'] == 0]), len(df_aggregate_sampled[df_aggregate_sampled['treatment_id'] == 0])]
                results["treatment_1_samples"] = [len(df_sampled[df_sampled['treatment_id'] == 1]), len(df_sampled[df_sampled['treatment_id'] == 1]), len(df_seq_sampled[df_seq_sampled['treatment_id'] == 1]), len(df_hash_sampled[df_hash_sampled['treatment_id'] == 1]), len(df_hash_join_sampled[df_hash_join_sampled['treatment_id'] == 1]), len(df_sort_sampled[df_sort_sampled['treatment_id'] == 1]), len(df_aggregate_sampled[df_aggregate_sampled['treatment_id'] == 1])]
                results["trt_0_ratio"] = results["treatment_0_samples"] / results["train_size"]
                if run_index_scan:
                    # add index scan to results
                    ix_results = pd.DataFrame()
                    ix_results['operator'] = ['Index Scan']
                    ix_results['ground_truth'] = [np.mean(pred_index_ce['gt_ce'])]
                    ix_results['predicted'] = [np.mean(pred_index_ce['pred_ce'])]
                    ix_results['r2_score'] = [r2_score(pred_index_ce['gt_ce'], pred_index_ce['pred_ce'])]
                    ix_results['mse'] = [mean_squared_error(pred_index_ce['gt_ce'], pred_index_ce['pred_ce'])]
                    ix_results['train_size'] = [len(df_index_sampled)]
                    ix_results["query_ids"] = [len(df_index_sampled['query_id'].unique())]
                    ix_results["treatment_0_samples"] = [len(df_index_sampled[df_index_sampled['treatment_id'] == 0])]
                    ix_results["treatment_1_samples"] = [len(df_index_sampled[df_index_sampled['treatment_id'] == 1])]
                    ix_results["trt_0_ratio"] = ix_results["treatment_0_samples"] / ix_results["train_size"]
                    results = pd.concat([results, ix_results])

                if log_transform:
                    rev_log_transform = results['ground_truth'].apply(lambda x: np.sign(x)* (10**abs(x) - 1))
                
                
                all_results[identifier] = results

    
            for identifier in identifiers:
                res = all_results[identifier]['r2_score'].values
                res_mse = all_results[identifier]['mse'].values
                tratio = all_results[identifier]['trt_0_ratio'].values
                train_size = all_results[identifier]['train_size'].values
                ground_truth = all_results[identifier]['ground_truth'].values
                predicted = all_results[identifier]['predicted'].values
                names = all_results[identifier]['operator'].values
                for i in range(len(res)):
                    plot_df.append([trial, noise_variance, identifier, names[i], ground_truth[i], predicted[i], res[i], res_mse[i], tratio[i], train_size[i]])

    plot_df = pd.DataFrame(plot_df, columns = ['trial', 'noise_variance', 'identifier', 'operator', 'ground_truth', 'predicted', 'r2_score', 'mse', 'trt_0_ratio', 'train_size'])
    # save plot_df
    filename = 'results/{}/cate_results_{}.csv'.format(results_folder,model_type)

    if os.path.exists(filename):
        # add number _v1, _v2, etc if number exists
        previous_versions = glob.glob('results/{}/cate_results_v*.csv'.format(results_folder))
        if len(previous_versions) > 0:
            version_numbers = [int(re.findall(r'\d+', x)[0]) for x in previous_versions]
            version_numbers.sort()
            version = version_numbers[-1] + 1
        else:
            version = 1
        filename = 'results/{}/cate_results_{}_v{}.csv'.format(results_folder, model_type, version)
    print("Saving results to {}".format(filename))
    plot_df.to_csv(filename, index = False)
    plot_df = plot_df[plot_df['operator'].isin(['High Level', 'Component Level Model'])]
    # # for each noise variance, plot r2 score vs. identifier
    plot_cate_results(plot_df, "plots/{}".format(plot_folder),"r2_score_cate")
        
   
    print(results_folder)
    print(plot_folder)
    return all_ce, combined_predictions, pred_ce, pred_seq_ce, pred_hash_ce, pred_hash_join_ce, pred_sort_ce, pred_aggregate_ce, pred_index_ce, df_orig, df_seq, df_index, df_hash, df_hash_join, df_sort, df_aggregate, plot_df, df_sampled, df_seq_sampled, df_index_sampled, df_hash_sampled, df_hash_join_sampled, df_sort_sampled, df_aggregate_sampled, results, plot_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # num_trials 
    parser.add_argument("--num_trials", help="num trials", type=int, default=1)
    parser.add_argument("--treatment_ids", help="treatment ids", nargs='+', type=int, default=[0, 6])
    parser.add_argument("--remove_outliers", help="remove outliers", action="store_true",default=False)
    parser.add_argument("--indep_sampling", help="indep sampling", action="store_true", default=False)
    parser.add_argument("--changed_plans", help="changed plans", type=int, default=0)
    parser.add_argument("--log_transform", help="log transform", action="store_true",default=True)
    parser.add_argument("--sampling", help="sampling", type=str, default="observational")
    parser.add_argument("--biasing_covariate", help="biasing covariate", type=str, default="total_output_rows")
    parser.add_argument("--data_folder_name", help="data folder name", type=str, default="mathso")
    parser.add_argument("--data_agg_type", help="data agg type", type=str, default="agg")
    parser.add_argument("--noise_variances", help="noise variances", nargs='+', type=float, default=[0.0])
    parser.add_argument("--experiment_type", help="experiment type", type=str, default="cate")
    parser.add_argument("--model_type", help="model_type", type=str, default="random_forest")
    # filters
    parser.add_argument("--filters", action="store_true", default=False)
    args = parser.parse_args()
    
    num_trials = args.num_trials
    
    treatment_ids = args.treatment_ids
    remove_outliers = args.remove_outliers
    indep_sampling = args.indep_sampling
    changed_plans = args.changed_plans
    log_transform = args.log_transform
    sampling = args.sampling
    biasing_covariate = args.biasing_covariate
    data_folder_name = args.data_folder_name
    data_agg_type = args.data_agg_type
    noise_variances = args.noise_variances
    experiment_type = args.experiment_type
    filters = args.filters
    model_type = args.model_type
    all_ce, combined_predictions, pred_ce, pred_seq_ce, pred_hash_ce, pred_hash_join_ce, pred_sort_ce, pred_aggregate_ce, pred_index_ce, df_orig, df_seq, df_index, df_hash, df_hash_join, df_sort, df_aggregate, plot_df, df_sampled, df_seq_sampled, df_index_sampled, df_hash_sampled, df_hash_join_sampled, df_sort_sampled, df_aggregate_sampled,results, plot_df = causal_effect_estimation(treatment_ids, data_folder_name, data_agg_type, sampling, num_trials, changed_plans, log_transform, biasing_covariate, experiment_type,noise_variances, remove_outliers, indep_sampling, filters = filters, model_type = model_type)
    