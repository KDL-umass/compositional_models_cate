from statsmodels.distributions.empirical_distribution import ECDF
import random
import numpy as np
import csv
import json
import os
import hashlib
import pandas as pd
import matplotlib.pyplot as plt
from domains.tree_data_structures import ExpressionNode, QueryPlanNode, Node
import torch
import torch.nn as nn

def generate_input_trees(num_modules, feature_dim=3, seed=42, num_trees=1000, max_depth=5, fixed_structure=False, data_dist="uniform", covariates_shared=False):
    """
    Generate different input structures in the form of trees.

    Args:
        num_modules (int): The number of distinct modules.
        feature_dim (int): The dimension of the feature vectors.
        seed (int): The seed value for random number generation.
        num_trees (int): The number of input trees to generate.
        max_depth (int): The maximum depth of the input trees.
        fixed_structure (bool): Whether to generate input trees with a fixed structure.
        data_dist (str): The distribution of the data. Can be "uniform" or "normal".
        """
    random.seed(seed)
    np.random.seed(seed)
    input_trees = []
    module_means = {}
    module_covs = {}

    def build_tree(module_id, depth, data_dist="uniform", covariates_shared=False, input_features=None):
        """
        Recursively build an input tree.

        Args:
            module_id (int): The ID of the current module.
            depth (int): The depth of the current node in the tree.

        Returns:
            Node: The input tree represented as a Node object.
        """
        # # Sample mean and covariance for the module's features if not already done
        # if module_id not in module_means:
        #     module_means[module_id] = np.random.uniform(0, 1, feature_dim)
        #     module_covs[module_id] = np.random.uniform(0, 3, (feature_dim, feature_dim))
        #     module_covs[module_id] = module_covs[module_id] @ module_covs[module_id].T + np.eye(feature_dim)

        # feature = np.random.multivariate_normal(module_means[module_id], module_covs[module_id])
        # feature = np.random.uniform(0, 1, feature_dim)
        # sample features with different means and covariance
        # feature = np.random.multivariate_normal(module_means[module_id], module_covs[module_id])
        # if data_dist == "normal":
        #     means = np.random.uniform(0, 1, feature_dim)
        #     covs = np.random.uniform(0, 3, (feature_dim, feature_dim))
        #     covs = covs @ covs.T + np.eye(feature_dim)
        #     feature = np.random.multivariate_normal(means, covs)
        if data_dist == "uniform":
            if not covariates_shared:
                feature = np.random.uniform(0, 1, feature_dim)
            else:
                feature = input_features
        if depth >= max_depth or (random.random() < 0.2 and depth > 3):
            return Node(module_id, module_id, feature.tolist())

        children = []
        child_module_id = random.randint(1,num_modules)
        child_node = build_tree(child_module_id, depth + 1, data_dist=data_dist)
        children.append(child_node)
        return Node(module_id, module_id, feature.tolist(), children=children)

    def build_tree_exactly_once(module_id, depth, used_modules, data_dist="uniform", covariates_shared=False, input_features=None):
        # print("module_id: ", module_id)
        used_modules.add(module_id)
        # print("used_modules: ", used_modules)
        """
        Recursively build an input tree with each module appearing exactly once.

        Args:
            module_id (int): The ID of the current module.
            depth (int): The depth of the current node in the tree.
            used_modules (set): A set of module IDs that have already been used in the tree.

        Returns:
            Node: The input tree represented as a Node object.
        """
        # # Sample mean and covariance for the module's features if not already done
        # if module_id not in module_means:
        #     module_means[module_id] = np.random.uniform(0, 1, feature_dim)
        #     module_covs[module_id] = np.random.uniform(0, 3, (feature_dim, feature_dim))
        #     module_covs[module_id] = module_covs[module_id] @ module_covs[module_id].T + np.eye(feature_dim)

        # feature = np.random.multivariate_normal(module_means[module_id], module_covs[module_id])
        if data_dist == "normal":
            if not covariates_shared:
                means = np.random.uniform(0, 1, feature_dim)
                covs = np.random.uniform(0, 3, (feature_dim, feature_dim))
                covs = covs @ covs.T + np.eye(feature_dim)
                feature = np.random.multivariate_normal(means, covs)
            else:
                feature = input_features
        elif data_dist == "uniform":
            if not covariates_shared:
                feature = np.random.uniform(0, 1, feature_dim)
            else:
                feature = input_features
        
        # end tree if max depth is reached or all modules are used or randomly choose to end tree
        if depth >= max_depth or len(used_modules) == num_modules:
            return Node(module_id, module_id, feature.tolist())

        children = []
        unused_modules = set(range(1, num_modules + 1)) - used_modules - {module_id}
        # print("unused_modules: ", unused_modules)
        if len(unused_modules) != 0:
            child_module_id = random.choice(list(unused_modules))
            used_modules.add(child_module_id)
            child_node = build_tree_exactly_once(child_module_id, depth + 1, used_modules, data_dist=data_dist, covariates_shared=covariates_shared, input_features=input_features)
            children.append(child_node)

        return Node(module_id, module_id, feature.tolist(), children=children)

    # Generate a desired number of input trees
    for i in range(num_trees):
        # print("Tree: ", _)
        root_module_id = random.randint(1, num_modules)
        if covariates_shared:
            if data_dist == "uniform":
                input_features = np.random.uniform(0, 1, feature_dim)
            elif data_dist == "normal":
                means = np.random.uniform(0, 1, feature_dim)
                covs = np.random.uniform(0, 3, (feature_dim, feature_dim))
                covs = covs @ covs.T + np.eye(feature_dim)
                input_features = np.random.multivariate_normal(means, covs)
        else:
            input_features = None
            
        if fixed_structure:
            # print("Generating trees with each module appearing exactly once.")
            input_trees.append(build_tree_exactly_once(root_module_id, 1, set(), data_dist=data_dist, covariates_shared=covariates_shared, input_features=input_features))
        else:
            # print("Generating trees with modules appearing multiple times.")
            input_trees.append(build_tree(root_module_id, 1, data_dist=data_dist, covariates_shared=covariates_shared, input_features=input_features))


    return input_trees, module_means, module_covs

def simulate_outcome(input_tree, treatment_id, module_functions, module_params_dict, max_depth=float('inf'), feature_dim=3, composition_type="hierarchical"):
    def propagate(node, depth=1):
        if depth > max_depth:
            return 0.0
        module_id = node['module_id']
        
        inputs = node['features'] 
        if node['children'] is not None and composition_type == "hierarchical":
            # include the output of the children as inputs
            inputs = inputs + [child['output'] for child in node['children']]
        else:
            inputs = inputs + [0.0]
        
        module_function = module_functions.get(module_id)
        if module_function is None:
            print(f"Module function not found for module ID: {module_id}")
            return 0.0

        module_params = module_params_dict.get(module_id, {})
        output = module_function(*inputs, **module_params)
        node['output'] = output
        # total_output is the output of the module including the output of its children if composition_type is parallel
        if node['children'] is not None and composition_type == "parallel":
            # print(module_id, output, [child['total_output'] for child in node['children']])
            node['total_output'] = output + sum([child['total_output'] for child in node['children']])
        else:
            node['total_output'] = output
        node['treatment_id'] = treatment_id
        return output

    def traverse_tree(node):
        if node['children'] is not None:
            for child in node['children']:
                traverse_tree(child)
        propagate(node)

    # Convert the input tree to a dictionary format
    input_tree_dict = input_tree.to_dict()

    # Traverse the tree and propagate values
    traverse_tree(input_tree_dict)
    query_output = input_tree_dict['total_output']

    # Return the output of the root module
    return input_tree_dict, query_output

def expand_features(tree_dict):
    if tree_dict is None:
        return None
    expanded_features = tree_dict["features"].copy()

    if tree_dict["children"] is None:
        # expand by adding 0
        expanded_features.append(0.0)
        tree_dict["features"] = expanded_features
        return tree_dict

    for child in tree_dict["children"]:
        expand_features(child)
        expanded_features.append(child["output"])

    tree_dict["features"] = expanded_features

    return tree_dict
    

def process_high_level_features(tree, query_id, treatment_id,tree_depth, query_output, sorted_module_names, module_feature_names, module_feature_counts, exactly_one_occurrence):
    module_counts = {}
    module_features = {}
    module_outputs = {}

    # initialize module_feature_names
    for name in sorted_module_names:
        module_features[name] = [0] * module_feature_counts[name]
        module_counts[name] = 0
        module_outputs[name] = 0

    def collect_features(node):
        # print(node)
        module_name = node['module_name']
        module_counts[module_name] += 1
        for i, feature in enumerate(node['features']):
            if isinstance(feature, (int, float)):
                # print(node['features'], module_features[module_name])
                module_features[module_name][i] += feature if not np.isnan(feature) else 0
            else:
                module_features[module_name][i] = feature if feature is not None else 'None'
        module_outputs[module_name] += node['output']
        
        if node['children'] is not None:
            for child in node['children']:
                collect_features(child)

    collect_features(tree)
    
    high_level_row = [query_id, treatment_id, tree_depth]
    high_level_row += [module_counts[module] for module in sorted_module_names]
    
    if exactly_one_occurrence:
        features = [feature for module in sorted_module_names for feature in module_features[module]]
        outputs = [module_outputs[module] for module in sorted_module_names]
    else:
        features = []
        outputs = []
        for module in sorted_module_names:
            count = module_counts[module]
            if count > 0:
                features.extend([feature / count if isinstance(feature, (int, float)) else feature 
                                 for feature in module_features[module]])
                outputs.append(module_outputs[module] / count)
            else:
                features.extend([0 if isinstance(feature, (int, float)) else 'None'
                                 for feature in module_features[module]])
                outputs.append(0)

    high_level_row.extend(features)
    high_level_row.extend(outputs)
    high_level_row.append(query_output)

    return high_level_row


def process_tree(node, treatment_id, csv_data, query_id, module_feature_names):
    module_name = node["module_name"]
    features = node["features"]
    output = node["output"]

    # Update module feature names
    if module_name not in module_feature_names:
        module_feature_names[module_name] = node["feature_names"]


    # Prepare CSV data for the current module
    csv_row = [query_id, treatment_id] + features + [output]
    if module_name in csv_data:
        csv_data[module_name].append(csv_row)
    else:
        csv_data[module_name] = [csv_row]

    # Process child nodes recursively
    if node["children"]:
        for child_node in node["children"]:
            process_tree(child_node, treatment_id, csv_data, query_id, module_feature_names)

    return module_feature_names

def process_trees_and_create_module_csvs(data_folders, csv_folder):
    csv_data = {}
    module_feature_names = {}
    all_modules = set()

    for folder in data_folders:
        for filename in os.listdir(folder):
            if filename.endswith(".json"):
                file_path = os.path.join(folder, filename)
                
                with open(file_path, "r") as file:
                    json_tree = json.load(file)
                    treatment_id = json_tree["treatment_id"]
                    query_id = json_tree["query_id"]
                    tree = json_tree["json_tree"]

                module_feature_names = process_tree(tree, treatment_id, csv_data, query_id, module_feature_names)
                all_modules.update(csv_data.keys())

    # Write CSV data to files for each module
    print("Writing CSV data to files for each module.")
    print(module_feature_names)
    for module_name, data in csv_data.items():
        csv_filename = f"module_{module_name}.csv"
        csv_path = os.path.join(csv_folder, csv_filename)
        with open(csv_path, "w", newline='') as file:
            writer = csv.writer(file)
            columns = ["query_id", "treatment_id"] + module_feature_names[module_name] + ["output"]
            writer.writerow(columns)
            writer.writerows(data)

    sorted_module_names = sorted(all_modules)
    module_feature_counts = {module: len(feature_names) for module, feature_names in module_feature_names.items()}
    
    return sorted_module_names, module_feature_names, module_feature_counts


def process_trees_and_create_high_level_csv(data_folders, csv_folder, sorted_module_names, module_feature_names, module_feature_counts, exactly_one_occurrence=False, domain="synthetic_data"):
    high_level_data = []

    for folder in data_folders:
        for filename in os.listdir(folder):
            if filename.endswith(".json"):
                file_path = os.path.join(folder, filename)
                
                with open(file_path, "r") as file:
                    json_tree = json.load(file)
                    treatment_id = json_tree["treatment_id"]
                    query_id = json_tree["query_id"]
                    query_output = json_tree["query_output"]
                    tree = json_tree["json_tree"]
                    # get tree depth 
                    # first convert the tree to a Node object which depends on the domain
                    if domain == "maths_evaluation":
                        tree_node = ExpressionNode.from_dict(tree)
                    elif domain == "query_execution":
                        tree_node = QueryPlanNode.from_dict(tree)
                    elif domain == "synthetic_data":
                        tree_node = Node.from_dict(tree)
                    
                    tree_depth = tree_node.get_depth()

                high_level_row = process_high_level_features(tree, query_id, treatment_id, tree_depth, query_output, sorted_module_names, module_feature_names, module_feature_counts, exactly_one_occurrence)
                
                if domain == "maths_evaluation":
                    matrix_size = int(query_id.split("_")[1])
                    high_level_row.append(matrix_size)

                high_level_data.append(high_level_row)

    # Write the high-level CSV
    high_level_csv_filename = f"{domain}_data_high_level_features.csv"
    high_level_csv_path = os.path.join(csv_folder, high_level_csv_filename)
    with open(high_level_csv_path, "w", newline='') as file:
        writer = csv.writer(file)
        columns = ["query_id", "treatment_id", "tree_depth"]
        columns += [f"num_module_{module_name}" for module_name in sorted_module_names]
        columns += [f"module_{module_name}_feature_{feature_name}" for module_name in sorted_module_names for feature_name in module_feature_names[module_name]]
        columns += [f"module_{module_name}_output" for module_name in sorted_module_names]
        columns += ["query_output"]
        if domain == "maths_evaluation":
            columns += ["matrix_size"]
        writer.writerow(columns)
        writer.writerows(high_level_data)

def process_trees_and_create_csvs(data_folders, csv_folder, config_folder, exactly_one_occurrence=False, domain="synthetic_data"):
    # first process the trees and create module csvs
    sorted_module_names, module_feature_names, module_feature_counts = process_trees_and_create_module_csvs(data_folders, csv_folder)
    # then process the trees and create high level csv
    process_trees_and_create_high_level_csv(data_folders, csv_folder, sorted_module_names, module_feature_names, module_feature_counts, exactly_one_occurrence, domain=domain)

    # create a config file with the domain name and save module names and feature names and feature counts
    config = {}
    config["domain"] = domain
    config["sorted_module_names"] = sorted_module_names
    config["module_feature_names"] = module_feature_names
    config["module_feature_counts"] = module_feature_counts

    # save the config file
    config_filename = f"{domain}_config.json"
    config_path = os.path.join(config_folder, config_filename)
    with open(config_path, "w") as file:
        json.dump(config, file)


def observational_sampling(df_apo, biasing_covariate, bias_strength, plot_folder=None):
    df_apo.sort_values(by=["query_id", "treatment_id"], inplace=True)
    
    treatment_ids = list(df_apo["treatment_id"].unique())
    if biasing_covariate == "feature_sum":
        feature_names = [col for col in df_apo.columns if "feature" in col]
        df_apo["feature_sum"] = df_apo[feature_names].sum(axis=1)
        cov = df_apo["feature_sum"].values
        # drop feature_sum from df_apo
        _ = df_apo.drop(columns=["feature_sum"], inplace=True)
    else:
        cov = df_apo[biasing_covariate].values
    
    cov = cov[::2]
    ecdf = ECDF(cov)
    cov_ecdf = ecdf(cov)
    cov_ecdf = cov_ecdf - np.mean(cov_ecdf)
    coefficients = np.repeat(bias_strength, len(cov))
    prob_values = 1 / (1 + np.exp(-coefficients * cov_ecdf))
    prob_values = np.clip(prob_values, 0.001, 0.999)
    
    assigned_treatment_ids = np.random.binomial(1, prob_values)
    assigned_treatment_ids = np.where(assigned_treatment_ids == 1, treatment_ids[0], treatment_ids[1])
    assigned_treatment_ids = np.repeat(assigned_treatment_ids, 2)
    df_apo["assigned_treatment_id"] = assigned_treatment_ids

    # if plot_folder is not None:
    #     plt.figure(figsize=(10, 10))
    #     plt.scatter(cov, prob_values)
    #     plt.xlabel(biasing_covariate)
    #     plt.ylabel("prob_values")
    #     plot_dir = "{}/{}".format(plot_folder, "prob_values_vs_covariate_values")
    #     os.makedirs(plot_dir, exist_ok=True)
    #     plt.savefig(f"{plot_dir}/prob_values_vs_covariate_values_bias_strength_{bias_strength}.png")

    df_sampled = df_apo[df_apo["treatment_id"] == df_apo["assigned_treatment_id"]]
    df_cf_sampled = df_apo[~df_apo.index.isin(df_sampled.index)]
    # drop treatment_id from df_sampled and df_cf_sampled
    df_sampled.drop(columns=["treatment_id"], inplace=True)
    df_cf_sampled.drop(columns=["treatment_id"], inplace=True)
    # rename assigned_treatment_id to treatment_id
    df_sampled.rename(columns={"assigned_treatment_id": "treatment_id"}, inplace=True)
    df_cf_sampled.rename(columns={"assigned_treatment_id": "treatment_id"}, inplace=True)

    # drop assigned_treatment_id from df
    df_apo.drop(columns=["assigned_treatment_id"], inplace=True)
    # also return a dictionary of the treatment assignments
    # make dict with query_id as key and treatment_id as value
    treatment_assignments = df_sampled[["query_id", "treatment_id"]].set_index("query_id").to_dict()["treatment_id"]
    return df_sampled, df_cf_sampled, treatment_assignments

