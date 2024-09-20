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
from itertools import combinations

def generate_feature(feature_dim=3, data_dist="uniform"):
    if data_dist == "normal":
        means = np.random.uniform(0, 1, feature_dim)
        covs = np.random.uniform(0, 3, (feature_dim, feature_dim))
        covs = covs @ covs.T + np.eye(feature_dim)
        feature = np.random.multivariate_normal(means, covs)
    elif data_dist == "uniform":
        feature = np.random.uniform(0, 1, feature_dim)
    return feature

def build_tree_systematically(modules, depth, used_modules, data_dist="uniform", covariates_shared=False, input_features=None, feature_dim=3, max_depth=5):
    
    if used_modules is None:
        used_modules = set()
    
    unused_modules = set(modules) - used_modules
    if not unused_modules:
        return None
    
    module_id = random.choice(list(unused_modules))
    if covariates_shared:
        feature = input_features
    else:
        feature = generate_feature(data_dist=data_dist, feature_dim=feature_dim)
    used_modules.add(module_id)
    
    if depth >= max_depth or len(used_modules) == len(modules):
        return Node(module_id, module_id, feature.tolist())
    
    child_node = build_tree_systematically(modules, depth + 1, used_modules, data_dist=data_dist, covariates_shared=covariates_shared, input_features=input_features, feature_dim=feature_dim, max_depth=max_depth)
    children = [child_node] if child_node else []
    
    return Node(module_id, module_id, feature.tolist(), children=children)

def build_tree_random(module_id, depth, data_dist="uniform", covariates_shared=False, input_features=None, num_modules=10, max_depth=5):
    """
    Recursively build an input tree.

    Args:
        module_id (int): The ID of the current module.
        depth (int): The depth of the current node in the tree.

    Returns:
        Node: The input tree represented as a Node object.
    """
    if covariates_shared:
        feature = input_features
    else:
        feature = generate_feature(data_dist=data_dist, feature_dim=feature_dim)
    if depth >= max_depth or (random.random() < 0.2 and depth > 3):
        return Node(module_id, module_id, feature.tolist())

    children = []
    child_module_id = random.randint(1,num_modules)
    child_node = build_tree_random(child_module_id, depth + 1, data_dist=data_dist, covariates_shared=covariates_shared, input_features=input_features, num_modules=num_modules, max_depth=max_depth)
    children.append(child_node)
    return Node(module_id, module_id, feature.tolist(), children=children)


def build_tree_exactly_once(module_id, depth, used_modules, data_dist="uniform", covariates_shared=False, input_features=None, num_modules=10, max_depth=5):
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
    if covariates_shared:
        feature = input_features
    else:
        feature = generate_feature(data_dist=data_dist, feature_dim=feature_dim)
    
    # end tree if max depth is reached or all modules are used or randomly choose to end tree
    if depth >= max_depth or len(used_modules) == num_modules:
        return Node(module_id, module_id, feature.tolist())

    children = []
    unused_modules = set(range(1, num_modules + 1)) - used_modules - {module_id}
    # print("unused_modules: ", unused_modules)
    if len(unused_modules) != 0:
        child_module_id = random.choice(list(unused_modules))
        used_modules.add(child_module_id)
        child_node = build_tree_exactly_once(child_module_id, depth + 1, used_modules, data_dist=data_dist, covariates_shared=covariates_shared, input_features=input_features, num_modules=num_modules, max_depth=max_depth)
        children.append(child_node)

    return Node(module_id, module_id, feature.tolist(), children=children)

def generate_fixed_variable_structure_trees(num_modules, feature_dim=3, seed=42, num_trees=1000, max_depth=5, data_dist="uniform", covariates_shared=False, fixed_structure=True):
    # Generate a desired number of input trees
    input_trees = []
    for i in range(num_trees):
        # print("Tree: ", _)
        root_module_id = random.randint(1, num_modules)
        input_features = generate_feature(data_dist=data_dist, feature_dim=feature_dim)
            
        if fixed_structure:
            # print("Generating trees with each module appearing exactly once.")
            input_trees.append(build_tree_exactly_once(root_module_id, 1, set(), data_dist=data_dist, covariates_shared=covariates_shared, input_features=input_features, num_modules=num_modules, max_depth=max_depth))
        else:
            # print("Generating trees with modules appearing multiple times.")
            input_trees.append(build_tree_random(root_module_id, 1, data_dist=data_dist, covariates_shared=covariates_shared, input_features=input_features, num_modules=num_modules, max_depth=max_depth))


    return input_trees

def generate_systematic_trees(num_modules, feature_dim=3, seed=42, num_trees=1000, max_depth=5, data_dist="uniform", covariates_shared=False, systematic=False, trees_per_group=100):
    grouped_trees = {}
    all_modules = list(range(1, num_modules + 1))
    
    for group_size in range(2, num_modules + 1):
        print(f"Generating trees for group size: {group_size}")
        group_trees = []
        combinations_list = list(combinations(all_modules, group_size))
        print(f"Number of combinations: {len(combinations_list)}")
        trees_per_combination = trees_per_group // len(combinations_list)
        
        for module_combination in combinations_list:
            for _ in range(trees_per_combination):
                input_features = generate_feature(data_dist=data_dist, feature_dim=feature_dim)
                
                tree = build_tree_systematically(module_combination, 1, set(), data_dist=data_dist, covariates_shared=covariates_shared, input_features=input_features, feature_dim=feature_dim, max_depth=max_depth)
                if tree:
                    group_trees.append(tree)
        
        grouped_trees[group_size] = group_trees

        # count the total number of trees
        total_trees = sum(len(trees) for trees in grouped_trees.values())
        # print tree for each group size
        for group_size, trees in grouped_trees.items():
            print(f"Group size: {group_size}, Number of trees: {len(trees)}")
        # have a single list of trees
        input_trees = [tree for trees in grouped_trees.values() for tree in trees]
    
    return input_trees

def generate_input_trees(num_modules, feature_dim=3, seed=42, num_trees=1000, max_depth=5, fixed_structure=False, data_dist="uniform", covariates_shared=False, systematic=False, trees_per_group=100):
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

    if systematic:
        input_trees = generate_systematic_trees(num_modules, feature_dim=feature_dim, seed=seed, num_trees=num_trees, max_depth=max_depth, data_dist=data_dist, covariates_shared=covariates_shared, systematic=systematic, trees_per_group=trees_per_group)
    else:
        input_trees =  generate_fixed_variable_structure_trees(num_modules, feature_dim=feature_dim, seed=seed, num_trees=num_trees, max_depth=max_depth, data_dist=data_dist, covariates_shared=covariates_shared, fixed_structure=fixed_structure)
    print(f"Number of input trees generated: {len(input_trees)}")
    return input_trees

# def generate_input_trees(num_modules, feature_dim=3, seed=42, num_trees=1000, max_depth=5, 
#                          fixed_structure=False, data_dist="uniform", covariates_shared=False, 
#                          systematic=False, trees_per_group=100):
#     """
#     Generate different input structures in the form of trees.

#     Args:
#         num_modules (int): The number of distinct modules.
#         feature_dim (int): The dimension of the feature vectors.
#         seed (int): The seed value for random number generation.
#         num_trees (int): The number of input trees to generate (used when systematic=False).
#         max_depth (int): The maximum depth of the input trees.
#         fixed_structure (bool): Whether to generate input trees with a fixed structure.
#         data_dist (str): The distribution of the data. Can be "uniform" or "normal".
#         covariates_shared (bool): Whether to use shared covariates across nodes in a tree.
#         systematic (bool): Whether to generate trees systematically with all possible module combinations.
#         trees_per_group (int): The number of trees to generate for each group size when systematic=True.

#     Returns:
#         If systematic=False: 
#             tuple: (list of input trees, module_means, module_covs)
#         If systematic=True:
#             dict: A dictionary where keys are the number of modules and values are lists of input trees for all combinations.
#     """
#     random.seed(seed)
#     np.random.seed(seed)

#     def generate_feature(module_id=None, shared_features=None, feature_dim=3, data_dist="uniform"):
#         if data_dist == "normal":
#             if not covariates_shared:
#                 means = np.random.uniform(0, 1, feature_dim)
#                 covs = np.random.uniform(0, 3, (feature_dim, feature_dim))
#                 covs = covs @ covs.T + np.eye(feature_dim)
#                 return np.random.multivariate_normal(means, covs)
#             else:
#                 return shared_features
#         elif data_dist == "uniform":
#             if not covariates_shared:
#                 return np.random.uniform(0, 1, feature_dim)
#             else:
#                 return shared_features

#     def build_tree(module_id, depth, used_modules=None, shared_features=None, data_dist="uniform", feature_dim=3):
#         feature = generate_feature(module_id, shared_features=shared_features, feature_dim=feature_dim, data_dist=data_dist)
#         if depth >= max_depth or (random.random() < 0.2 and depth > 3):
#             return Node(module_id, module_id, feature.tolist())

#         children = []
#         child_module_id = random.randint(1, num_modules)
#         child_node = build_tree(child_module_id, depth + 1)
#         children.append(child_node)
#         return Node(module_id, module_id, feature.tolist(), children=children)

#     def build_tree_fixed(module_id, depth, used_modules, shared_features=None):
#         if module_id in used_modules:
#             return None
        
#         feature = generate_feature(module_id, shared_features=shared_features)
#         used_modules.add(module_id)
        
#         if depth >= max_depth or len(used_modules) == num_modules:
#             return Node(module_id, module_id, feature.tolist())

#         children = []
#         unused_modules = set(range(1, num_modules + 1)) - used_modules
#         if unused_modules:
#             child_module_id = random.choice(list(unused_modules))
#             child_node = build_tree_fixed(child_module_id, depth + 1, used_modules)
#             if child_node:
#                 children.append(child_node)

#         return Node(module_id, module_id, feature.tolist(), children=children)

#     def build_tree_systematically(modules, depth=1, used_modules=None, shared_features=None):
#         if used_modules is None:
#             used_modules = set()
        
#         unused_modules = set(modules) - used_modules
#         if not unused_modules:
#             return None
        
#         module_id = random.choice(list(unused_modules))
#         feature = generate_feature(module_id, shared_features=shared_features)
#         used_modules.add(module_id)
        
#         if depth >= max_depth or len(used_modules) == len(modules):
#             return Node(module_id, module_id, feature.tolist())
        
#         child_node = build_tree_systematically(modules, depth + 1, used_modules)
#         children = [child_node] if child_node else []
        
#         return Node(module_id, module_id, feature.tolist(), children=children)

#     if systematic:
#         grouped_trees = {}
#         all_modules = list(range(1, num_modules + 1))
        
#         for group_size in range(2, num_modules + 1):
#             print(f"Generating trees for group size: {group_size}")
#             group_trees = []
#             combinations_list = list(combinations(all_modules, group_size))
#             print(f"Number of combinations: {len(combinations_list)}")
#             trees_per_combination = trees_per_group // len(combinations_list)
            
#             for module_combination in combinations_list:
#                 for _ in range(trees_per_combination):
#                     if covariates_shared:
#                         shared_features = generate_feature()
#                     else:
#                         shared_features = None
                    
#                     tree = build_tree_systematically(module_combination, shared_features=shared_features)
#                     if tree:
#                         group_trees.append(tree)
            
#             grouped_trees[group_size] = group_trees

#             # count the total number of trees
#             total_trees = sum(len(trees) for trees in grouped_trees.values())
#             # print tree for each group size
#             for group_size, trees in grouped_trees.items():
#                 print(f"Group size: {group_size}, Number of trees: {len(trees)}")
#             # have a single list of trees
#             all_trees = [tree for trees in grouped_trees.values() for tree in trees]
        
#         return all_trees
#     else:
#         input_trees = []
#         # module_means = {}
#         # module_covs = {}

#         for _ in range(num_trees):
#             root_module_id = random.randint(1, num_modules)
#             if covariates_shared:
#                 shared_features = generate_feature()
#             else:
#                 shared_features = None
            
#             if fixed_structure:
#                 input_trees.append(build_tree_fixed(root_module_id, 1, set(), shared_features=shared_features))
#             else:
#                 input_trees.append(build_tree(root_module_id, 1, shared_features=shared_features))

#         return input_trees

# def count_systematic_trees(grouped_trees):
#     """
#     Count the total number of trees in the grouped_trees dictionary.
#     """
#     total_trees = sum(len(trees) for trees in grouped_trees.values())
#     return total_trees
    
def simulate_outcome(input_tree, treatment_id, module_functions, module_params_dict, max_depth=float('inf'), feature_dim=3, composition_type="hierarchical", use_subset_features=False):
    def propagate(node, depth=1):
        if depth > max_depth:
            return 0.0
        module_id = node['module_id']
        

        if not use_subset_features:
            inputs = node['features'] 
        else:
            # get total number of modules 
            num_modules = len(module_functions)
            # divide the features into equal parts for each module
            num_features_per_module = feature_dim // num_modules
            # get the index of the first feature for the current module
            start_index = (module_id - 1) * num_features_per_module
            # get the index of the last feature for the current module
            end_index = start_index + num_features_per_module
            inputs = node['features'][start_index:end_index]
            
            

        if composition_type == "hierarchical":
            # include the output of the children as inputs
            if node['children'] is not None:
                inputs = inputs + [child['output'] for child in node['children']]
            else:
                inputs = inputs + [0.0]
        else:
            inputs = inputs
        
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

def process_trees_and_create_module_csvs(data_folders, csv_folder, source="processed"):
    csv_data = {}
    module_feature_names = {}
    all_modules = set()
    if source == "json":
        for folder in data_folders:
            for filename in os.listdir(folder):
                if filename.endswith(".jsonl"):
                    file_path = os.path.join(folder, filename)
                    
                    with open(file_path, "r") as file:
                        for line in file:
                            json_tree = json.loads(line)
                            treatment_id = json_tree["treatment_id"]
                            query_id = json_tree["query_id"]
                            tree = json_tree["json_tree"]

                            process_tree(tree, treatment_id, csv_data, query_id, module_feature_names)
                            all_modules.update(csv_data.keys())
    elif source == "processed":
        for i, treatment_tree_dicts in enumerate(data_folders):
            for input_dict in treatment_tree_dicts:
                treatment_id = input_dict["treatment_id"]
                query_id = input_dict["query_id"]
                tree = input_dict["json_tree"]

                process_tree(tree, treatment_id, csv_data, query_id, module_feature_names)
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


def process_trees_and_create_high_level_csv(data_folders, csv_folder, sorted_module_names, module_feature_names, module_feature_counts, exactly_one_occurrence=False, domain="synthetic_data", source="processed"):
    high_level_data = []

    if source == "json":
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
    elif source == "processed":
        for i, treatment_tree_dicts in enumerate(data_folders):
            for input_dict in treatment_tree_dicts:
                treatment_id = input_dict["treatment_id"]
                query_id = input_dict["query_id"]
                query_output = input_dict["query_output"]
                tree = input_dict["json_tree"]
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

def process_trees_and_create_csvs(data_folders, csv_folder, config_folder, exactly_one_occurrence=False, domain="synthetic_data", source="processed"):
    # first process the trees and create module csvs
    sorted_module_names, module_feature_names, module_feature_counts = process_trees_and_create_module_csvs(data_folders, csv_folder, source=source)
    # then process the trees and create high level csv
    process_trees_and_create_high_level_csv(data_folders, csv_folder, sorted_module_names, module_feature_names, module_feature_counts, exactly_one_occurrence, domain=domain, source=source)

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

