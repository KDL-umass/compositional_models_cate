import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from domains.tree_data_structures import ExpressionNode, QueryPlanNode, Node
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# load trees from JSON files.
# Return the tree-structured units grouped by depth because it will be used to create batches.
def load_dataset_depth_wise(domain, data_folder, outcomes_parallel=0, single_outcome=False, load_all=True):
    tree_depth_groups = {}
    files = [f for f in os.listdir(data_folder) if f.endswith('.json')]
    if not load_all:
        files = files[:10000]
    for file in files:
        with open(os.path.join(data_folder, file), "r") as f:
            data = json.load(f)
            input_tree_dict = data["json_tree"]
            input_tree = {
                "maths_evaluation": ExpressionNode,
                "query_execution": QueryPlanNode,
                "synthetic_data": Node
            }[domain].from_dict(input_tree_dict)
            
            query_id = data["query_id"]
            outputs = data["query_output"] if single_outcome else input_tree.return_outputs_as_list_postorder()
            depth = input_tree.get_depth()
            
            tree_depth_groups.setdefault(depth, {})[query_id] = (input_tree, outputs)
    return tree_depth_groups

import math

def replace_large_numbers(features, threshold=1e6):
    def process_value(value):
        if isinstance(value, (int, float)):
            if abs(value) > threshold or math.isinf(value):
                return 0
        if value is None:
            return 0
        return value

    return [process_value(feature) for feature in features]

def apply_scalers_to_node(node, input_scalers, output_scalers):
    module_name = node.module_name
    input_scaler = input_scalers.get(module_name)
    output_scaler = output_scalers.get(module_name)
    high_level_output_scaler = output_scalers.get("high_level")
    # Scale features
    if node.features is not None:
        # replace nan values with 0
        # replace nan, inf, -inf with 0
        node.features = replace_large_numbers(node.features)
        scaled_features = input_scaler.transform([node.features])[0]
        # if there are nan values, replace them with 0
        scaled_features = np.nan_to_num(scaled_features,posinf=0,neginf=0)
        node.features = scaled_features
    
    # Scale output (apply log transform and then scale)
    if node.output is not None:
        # log_output = np.log(node.output)
        # scaled_output = output_scaler.transform([[log_output]])[0][0]
        
        node.output = np.log(node.output)
    
    # Scale total_output (apply log transform and then scale)
    if node.total_output is not None:
        log_total_output = np.log(node.total_output)
        # scaled_total_output = high_level_output_scaler.transform([[log_total_output]])[0][0]
        node.total_output = log_total_output
    
    return node

def apply_scalers_recursive(node, input_scaler, output_scaler):
    # Apply scalers to current node
    node = apply_scalers_to_node(node, input_scaler, output_scaler)
    
    # Recursively apply scalers to children
    if node.children is not None:
        for i, child in enumerate(node.children):
            node.children[i] = apply_scalers_recursive(child, input_scaler, output_scaler)
    
    return node

def load_dataset(domain, data_folder, outcomes_parallel=0, single_outcome=False, load_all=True, input_scalers=None, output_scalers=None):
    tree_groups = {}
    files = [f for f in os.listdir(data_folder) if f.endswith('.json')]
    if not load_all:
        files = files[:1000]
    for file in files:
        with open(os.path.join(data_folder, file), "r") as f:
            data = json.load(f)
            input_tree_dict = data["json_tree"]
            input_tree = {
                "maths_evaluation": ExpressionNode,
                "query_execution": QueryPlanNode,
                "synthetic_data": Node
            }[domain].from_dict(input_tree_dict)
            
            query_id = data["query_id"]

            # apply scalers to the tree
            if input_scalers is not None and output_scalers is not None:
                input_tree = apply_scalers_recursive(input_tree, input_scalers, output_scalers)
            
            # data["query_output"] = output_scalers["high_level"].transform([[np.log(data["query_output"])]])[0][0]
            data["query_output"] = np.log(data["query_output"])

            outputs = data["query_output"] if single_outcome else input_tree.return_outputs_as_list_postorder()
            
            tree_groups[query_id] = (input_tree, outputs)
    return tree_groups

def train_test_split_tree_groups(tree_groups_0, tree_groups_1, split_info):
    train_dataset_0, train_dataset_1 = {}, {}
    test_dataset = []
    # split info contains the query ids for train and test {"train": [], "test": []}
    for query_id, (tree, outputs) in tree_groups_0.items():
        if query_id in split_info["train"]:
            train_dataset_0[query_id] = (tree, outputs)
            train_dataset_1[query_id] = (tree_groups_1[query_id][0], tree_groups_1[query_id][1])
        else:
            tree_0, outputs_0 = tree, outputs
            tree_1, outputs_1 = tree_groups_1[query_id][0], tree_groups_1[query_id][1]
            test_dataset.append((query_id, tree_0, outputs_0, tree_1, outputs_1))
    return train_dataset_0, train_dataset_1, test_dataset

# Return the observational dataset
def get_observational_tree_groups(tree_groups_0, tree_groups_1, query_id_to_treatment):
    sampled_tree_groups = {}
    for query_id, (tree, outputs) in tree_groups_0.items():
        # Get the corresponding tree for treatment 1
        treatment_id = query_id_to_treatment[query_id]
        if treatment_id == 0:
            sampled_tree_groups[query_id] = (tree, treatment_id, outputs)
        else:
            sampled_tree_groups[query_id] = (tree_groups_1[query_id][0], treatment_id, tree_groups_1[query_id][1])
    return sampled_tree_groups

def create_batch(train_dataset_tree_groups, query_depth_info, batch_size=32):
    # we have to create a batch of queries,groupped by depth 
    depth_query_ids = {}
    for query_id, depth in query_depth_info.items():
        if query_id in train_dataset_tree_groups:
            depth_query_ids.setdefault(depth, []).append(query_id)
    print(f"Depth query ids: {depth_query_ids.keys()}")
    train_dataset = []
    for depth, query_ids in depth_query_ids.items():
        items = [train_dataset_tree_groups[query_id] for query_id in query_ids]
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i+batch_size]
            trees, treatment_ids, outputs = zip(*[(item[0], item[1], item[2]) for item in batch])
            train_dataset.append((query_ids, trees, treatment_ids, outputs))
    return train_dataset


def create_test_df(test_predictions, test_outputs, single_outcome=True, outcomes_parallel=False):
    def process_data(data, prefix):
        return [
            {
                "query_id": query_id,
                f"y0_{prefix}": output["y0"] if single_outcome or outcomes_parallel else output["y0"][0],
                f"y1_{prefix}": output["y1"] if single_outcome or outcomes_parallel else output["y1"][0]
            }
            for query_id, output in data.items()
        ]

    pred_df = pd.DataFrame(process_data(test_predictions, "pred"))
    true_df = pd.DataFrame(process_data(test_outputs, "true"))
    
    return pd.merge(pred_df, true_df, on="query_id")


def plot_and_save_results(test_df, plot_folder, results_folder, split, sampling, biasing_covariate, bias_strength, nn_architecture_type, single_outcome, experiment_type="sample_size", plot_flag=True):
    ite_pred = test_df["y1_pred"] - test_df["y0_pred"]
    ite_true = test_df["y1_true"] - test_df["y0_true"]
    if plot_flag:
        general_scatter_plot(ite_true, ite_pred, "True ITE", "Predicted ITE", "Test Predictions vs Test Outputs for ITE", "{}/test_predictions_vs_outputs_ite_{}_{}_{}_single_outcome_{}.png".format(plot_folder, nn_architecture_type, split, bias_strength, single_outcome))

    # save test_df as csv
    test_df.to_csv("{}/df_outcomes_{}_{}_{}_{}_{}_single_outcome_{}.csv".format(results_folder, split, sampling, biasing_covariate, bias_strength, nn_architecture_type, single_outcome), index=False)
    pehe = np.mean((ite_true - ite_pred) ** 2)
    r2 = r2_score(ite_true, ite_pred)
    
    return pehe, r2

def load_query_id_to_treatment(dirs, biasing_covariate, bias_strength):
    obs_data_folder = f"{dirs['csvs']}/observational_data/{biasing_covariate}_{bias_strength}"
    json_file = f"{obs_data_folder}/treatment_assignments.json"
    # read json file
    with open(json_file, "r") as f:
        query_id_to_treatment = json.load(f)
    return query_id_to_treatment

def load_train_test_split_info(dirs, split):
    filepath = f"{dirs['csvs']}/{split}/train_test_split_qids.json"
    with open(filepath, "r") as f:
        train_test_split = json.load(f)
    return train_test_split

def load_query_id_depth_info(dirs, domain):
    # load csv file
    filepath = f"{dirs['csvs']}/{domain}_data_high_level_features.csv"
    df = pd.read_csv(filepath)
    query_id_depth_info = dict(zip(df["query_id"], df["tree_depth"]))
    return query_id_depth_info

def get_input_output_scalers(dirs, biasing_covariate, bias_strength, split_type):
    obs_csv_folder = f"{dirs['csvs']}/observational_data"
    split_folder = "{}/{}_{}/{}".format(obs_csv_folder, biasing_covariate, bias_strength, split_type)
    scaler_folder = "{}/scalers".format(split_folder)

    input_scalers = {}
    output_scalers = {}
    # get all .pkl files in the folder
    pkl_files = [f for f in os.listdir(scaler_folder) if f.endswith('.pkl')]
    for pkl_file in pkl_files:
        with open(os.path.join(scaler_folder, pkl_file), "rb") as f:
            if "input" in pkl_file:
                module_name = pkl_file.split("_")[2].split(".")[0]
                input_scalers[module_name] = pickle.load(f)
            else:
                if "high_level" in pkl_file:
                    output_scalers["high_level"] = pickle.load(f)
                else:
                    module_name = pkl_file.split("_")[2].split(".")[0]
                    output_scalers[module_name] = pickle.load(f)
    return input_scalers, output_scalers

def general_scatter_plot(true, predicted, true_label, predicted_label, title, plot_path):
    plt.figure(figsize=(10, 10))
    plt.scatter(x=true, y=predicted)
    plt.xlabel("True {}".format(true_label))
    plt.ylabel("Predicted {}".format(predicted_label))
    plt.title(title)
    # plot 45 degree line
    min_value = min(min(true), min(predicted))
    max_value = max(max(true), max(predicted))
    plt.plot([min_value, max_value], [min_value, max_value], color="red")
    # print r2 score
    r2 = r2_score(true, predicted)
    plt.text(0.1, 0.9, "R2 Score: {:.2f}".format(r2), fontsize=12, transform=plt.gcf().transFigure)
    plt.savefig(plot_path)

def get_ground_truth_effects(data, qids, treatment_col='treatment_id', outcome_col='query_output'):
    covariates = [x for x in data.columns if "feature" in x] + [treatment_col] + [outcome_col]
    
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
    return np.mean((ground_truth - estimated)**2)
    
def get_r2_score(ground_truth, estimated):
    return r2_score(ground_truth, estimated)

def scale_df(df_apo, df_sampled, scaler_path, csv_path, composition_type, covariates):

    if composition_type == "parallel":
        module_files = os.listdir(f"{csv_path}/")
        module_files = [x for x in module_files if "module" in x]
        module_output_columns = []
        for module_file in module_files:
            module_id = module_file.split(".")[0].split("_")[-1]
            module_input_scaler = pickle.load(open(f"{scaler_path}/input_scaler_{module_id}.pkl", "rb"))
            module_output_scaler = pickle.load(open(f"{scaler_path}/output_scaler_{module_id}.pkl", "rb"))
            module_features = [x for x in df_apo.columns if f"module_{module_id}_feature" in x]
            
            module_output = f"module_{module_id}_output"
            # print(df_apo[["query_id", "treatment_id", f"module_{module_id}_output"]].head())
            df_apo[[module_output]] = module_output_scaler.transform(df_apo[[module_output]].values.reshape(-1, 1))
            # print(df_apo[["query_id", "treatment_id", f"module_{module_id}_output"]].head())
            df_apo[module_features] = module_input_scaler.transform(df_apo[module_features].values)
            df_sampled[[module_output]] = module_output_scaler.transform(df_sampled[[module_output]].values.reshape(-1, 1))
            df_sampled[module_features] = module_input_scaler.transform(df_sampled[module_features].values)
            module_output_columns.append(module_output)

        # re-assign the query_output by adding the output of all modules
        
        df_apo["query_output"] = df_apo[module_output_columns].sum(axis=1)
        df_sampled["query_output"] = df_sampled[module_output_columns].sum(axis=1)

    else:
        # just scale the covariates
        df_apo[covariates] = df_apo[covariates].apply(lambda x: (x - x.mean()) / x.std())
        df_sampled[covariates] = df_sampled[covariates].apply(lambda x: (x - x.mean()) / x.std())


    return df_apo, df_sampled
            


