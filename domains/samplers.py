import sys 
sys.path.append("../")
from domains.data_utils import generate_input_trees, simulate_outcome, process_trees_and_create_csvs, expand_features, observational_sampling, add_order_of_modules
from domains.tree_data_structures import Node, ExpressionNode, QueryPlanNode
import numpy as np
import random
import os
import shutil
import json
import pandas as pd
import pickle
# import standard scaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
import torch.nn as nn

# Assuming 'data' is your dataset
# Convert to float64 for higher precision

# script to generate synthetic data for the causal effect estimation experiments

coeffs = {
 '1': {0: [ 0.21667   , -0.63903564,  9.00922865],
  1: [0.12936905, 0.2799703 , 6.57174534]},
 '2': {0: [ 0.30264717, -1.25032706,  9.94655612],
  1: [0.16302046, 1.00160131, 4.64935063]},
 '3': {0: [0.01939968, 0.02394197, 9.47095968],
  1: [0.02878614, 0.03671678, 8.68403881]},
 '4': {0: [ 0.22816887, -0.77317096,  9.38126938],
  1: [0.14832781, 0.1123053 , 6.89397014]},
 '5': {0: [ 0.21146381, -0.56553982,  8.49733234],
  1: [ 0.29130791, -0.41326397,  8.49733234]},
 '6': {0: [0.13604885, 0.41398028, 9.26971561],
  1: [ 0.27753554, -0.14679641,  9.01205926]},
 '7': {0: [ 0.21259568, -0.43276121, 10.05956736],
  1: [0.20716442, 0.60688892, 6.5829069 ]},
 '8': {0: [ 0.19895964, -0.46852413, 10.8638669 ],
  1: [ 0.33923464, -1.1632891 , 11.31139315]},
 '9': {0: [ 0.03409934, -0.05973316,  8.11280246],
  1: [-0.03727203,  0.68611027,  5.90218791]},
 '10': {0: [0.04776045, 1.34701657, 8.06038158],
  1: [0.16713525, 0.98506548, 7.36645402]}
}

def f1_module(*inputs, w):
    X = np.array(inputs)
    Mj = len(X)
    return np.dot(X, w[:Mj]) + w[-1]

def f2_module(*inputs, w):
    X = np.array(inputs)
    Mj = len(X)
    return np.dot(X**2, w[:Mj]) + np.dot(X, w[Mj:2*Mj]) + w[-1]

def f3_module(*inputs, w):
    X = np.array(inputs)
    Mj = len(X)
    return w[0] * (np.sin(np.pi * np.dot(X, w[1:Mj+1])) / 2 + 0.5) + w[-1]

def f4_module(*inputs, w):
    X = np.array(inputs)
    Mj = len(X)
    return w[0] / (1 + np.exp(-np.dot(X, w[1:Mj+1]))) + w[-1]

def f5_module(*inputs, w):
    X = np.array(inputs)
    Mj = len(X)
    return w[0] * np.sqrt(np.dot(X, w[1:Mj+1])) + w[-1]

def f6_module(*inputs, w):
    X = np.array(inputs)
    Mj = len(X)
    return w[0] * (1 - np.dot(X**3, w[1:Mj+1])) + w[-1]

def f7_module(*inputs, w):
    X = np.array(inputs)
    Mj = len(X)
    return w[0] * (0.5 * np.cos(2 * np.pi * np.dot(X, w[1:Mj+1])) + 0.5) + w[-1]

def f8_module(*inputs, w):
    X = np.array(inputs)
    Mj = len(X)
    return w[0] * np.exp(-np.dot(X, w[1:Mj+1])) / (1 + np.exp(-np.dot(X, w[1:Mj+1]))) + w[-1]

def f9_module(*inputs, w):
    X = np.array(inputs)
    Mj = len(X)
    return w[0] * np.log(np.dot(X, w[1:Mj+1]) + 1) / np.log(2) + w[-1]

def f10_module(*inputs, w):
    X = np.array(inputs)
    Mj = len(X)
    return w[0] * (0.5 * np.tanh(np.dot(X, w[1:Mj+1]) - 2) + 0.5) + w[-1]

def polyval_module(*inputs, w):
    return np.polyval(w, inputs)[0]

def quadratic_module(*inputs, w):
    # take all the inputs
    X = np.array(inputs)
    Mj = len(X)
    w = np.array(w)
    
    # quadratic outcome function
    x_squared = X**2
    y = np.dot(X**2, w[:Mj]) 
    y += np.dot(X, w[Mj:2*Mj])
    y += w[-1]
    return y

def linear_module(*inputs, w):
    X = np.array(inputs)
    Mj = len(X)
    # linear outcome function
    y = np.dot(X, w[:Mj]) + w[-1]
    return y

def mlp_module(*inputs, model):
    X = np.array(inputs)
    X = X.reshape(1, -1)
    model.eval()
    with torch.no_grad():
        output = model(torch.tensor(X, dtype=torch.float32))
    return output.item()

def logarithmic_module(*inputs, w):
    X = np.array(inputs)
    Mj = len(X)
    # take log of absolute value of X + 1
    X_log = np.log(np.abs(X) + 1)
    w = np.array(w)
    y = np.dot(X_log, w[:Mj]) + w[-1]
    # exponential function
    y = np.exp(y)
    return y

def sigmoid_module(*inputs, w):
    X = np.array(inputs)
    Mj = len(X)
    y = 1 / (1 + np.exp(-np.dot(X, w[:Mj]) - w[-1]))
    return y

def exponential_module(*inputs, w):
    X = np.array(inputs)
    Mj = len(X)
    y = np.exp(np.dot(X, w[:Mj])) + w[-1]
    return y



# TODO: add more modules

class SyntheticDataSampler:
    def __init__(
        self, 
        num_modules,
        feature_dim, 
        composition_type,
        fixed_structure,
        max_depth=float('inf'), 
        num_trees=1000,
        seed=42,
        data_dist="uniform",
        module_function_types = None,
        domain = "synthetic_data",
        resample = False,
        heterogeneity = 1,
        covariates_shared = False,
        use_subset_features = False,
        run_env = "local",
        systematic = False,
        trees_per_group = 2000,
        test_size = 0.2
    ):
        self.num_modules = num_modules
        print("num_modules: ", num_modules)
        self.feature_dim = feature_dim
        self.composition_type = composition_type
        self.fixed_structure = fixed_structure
        self.domain = domain
        # self.module_functions = module_functions
        # self.module_params_dict = module_params_dict
        self.max_depth = max_depth
        self.num_trees = num_trees
        self.seed = seed
        self.data_dist = data_dist
        self.input_trees, self.module_means, self.module_covs = None, None, None
        self.input_trees_dict = {}
        self.module_function_types = module_function_types or ['quadratic'] * num_modules  # Default to all linear if not specified
        self.module_params_dict = {}
        self.module_functions = None
        self.resample = resample
        self.run_env = run_env
        if self.run_env == "local":
            self.base_dir = "/Users/ppruthi/research/compositional_models/compositional_models_cate/domains"
        else:
            self.base_dir = "/work/pi_jensen_umass_edu/ppruthi_umass_edu/compositional_models_cate/domains"
        self.heterogeneity = heterogeneity
        self.covariates_shared = covariates_shared
        self.use_subset_features = use_subset_features
        self.trees_per_group = trees_per_group
        self.systematic = systematic
        
        self.initialize_folders()

        
    def initialize_folders(self):
            # make path if doesn't exist
        self.data_path = "{}/{}/jsons/fixed_structure_{}_outcomes_{}_systematic_{}".format(self.base_dir, self.domain, self.fixed_structure, self.composition_type, self.systematic)
        folder_0 = "data_0"
        folder_1 = "data_1"
        self.path_0 = "{}/{}".format(self.data_path, folder_0)
        self.path_1 = "{}/{}".format(self.data_path, folder_1)
        if not os.path.exists(self.path_0):
            os.makedirs(self.path_0)
        else:
            # remove whole folder
            if self.resample:
                print("Reinitializing path 0")
                shutil.rmtree(self.path_0)
                os.makedirs(self.path_0, exist_ok=True)
        if not os.path.exists(self.path_1):
            os.makedirs(self.path_1)
        else:
            if self.resample:
                print("Reinitializing path 1")
                shutil.rmtree(self.path_1)
                os.makedirs(self.path_1, exist_ok=True)

        self.csv_folder = "{}/{}/csvs/fixed_structure_{}_outcomes_{}_systematic_{}".format(self.base_dir, self.domain, self.fixed_structure, self.composition_type, self.systematic)
        if not os.path.exists(self.csv_folder):
            os.makedirs(self.csv_folder)
        else:
            if self.resample:
                print("Reinitializing csv folder")
                shutil.rmtree(self.csv_folder)
                os.makedirs(self.csv_folder, exist_ok=True)
        self.json_folders = [self.path_0, self.path_1]

        self.obs_csv_folder = "{}/{}/observational_data/fixed_structure_{}_outcomes_{}_systematic_{}".format(self.base_dir, self.domain, self.fixed_structure, self.composition_type, self.systematic)
        if not os.path.exists(self.obs_csv_folder):
            os.makedirs(self.obs_csv_folder)

        self.plot_folder = "{}/{}/plots/fixed_structure_{}_outcomes_{}_systematic_{}".format(self.base_dir, self.domain, self.fixed_structure, self.composition_type, self.systematic)
        if not os.path.exists(self.plot_folder):
            os.makedirs(self.plot_folder)

        self.config_folder = "{}/{}/config/fixed_structure_{}_outcomes_{}_systematic_{}".format(self.base_dir, self.domain, self.fixed_structure, self.composition_type, self.systematic)
        if not os.path.exists(self.config_folder):
            os.makedirs(self.config_folder)


    def generate_trees(self):
        print("Generating input trees")
        self.input_trees = generate_input_trees(self.num_modules, self.feature_dim, self.seed, self.num_trees, self.max_depth, self.fixed_structure, self.data_dist, self.covariates_shared, self.systematic, self.trees_per_group)
        
        
    def generate_module_weights(self, treatment_id):
        base_seed = self.seed + treatment_id * 100
        if self.composition_type == "hierarchical":
            input_dim = self.feature_dim + 1
        else:
            input_dim = self.feature_dim
        # Generate a base set of weights
        np.random.seed(base_seed)
        
        if not self.use_subset_features:
            base_weights = np.random.uniform(0.1, 1, 2 * (input_dim) + 1)
            module_feature_dim = input_dim
        else:
            # set feature dim equally across all modules
            num_modules = self.num_modules
            feature_dim = input_dim
            feature_dim_per_module = int(feature_dim / num_modules)
            base_weights = np.random.uniform(0.1, 1, 2 * (feature_dim_per_module) + 1)
            module_feature_dim = feature_dim_per_module
        
        # Determine how many modules will have the same weights
        num_same_weight_modules = int(self.num_modules * (1 - self.heterogeneity))
        
        # Define the MLP architecture if needed
        
        input_dim = module_feature_dim
        if "mlp" in self.module_function_types:
            base_mlp = self.create_mlp(input_dim)
        
        for module_id in range(1, self.num_modules + 1):
            Mj = module_feature_dim # feature dim for this module
            
            if module_id <= num_same_weight_modules:
                # Use exactly the same weights for these modules
                module_type = self.module_function_types[0]
                if module_type == 'mlp':
                    model = copy.deepcopy(base_mlp)
                else:
                    w = base_weights
                
            else:
                # Use distinct weights for the remaining modules
                np.random.seed(base_seed + module_id)
                module_type = self.module_function_types[module_id - num_same_weight_modules - 1]
                if module_type == 'mlp':
                    model = self.create_mlp(input_dim)
                    # Initialize the weights randomly
                    for layer in model.modules():
                        if isinstance(layer, nn.Linear):
                            nn.init.xavier_uniform_(layer.weight)
                            nn.init.zeros_(layer.bias)
                else:
                   w = np.random.uniform(0.1, 1, 2 * (Mj) + 1)
                
            
            if module_type == 'mlp':
                self.module_params_dict[module_id] = {"model": model}
            elif module_type == 'polyval':
                    w = coeffs[str(module_id)][treatment_id]
                    self.module_params_dict[module_id] = {"w": w}
            else:
                self.module_params_dict[module_id] = {"w": w.tolist()}

    def create_mlp(self, input_dim):
        hidden_dim1 = 2 * input_dim
        hidden_dim2 = 2 * input_dim
        output_dim = 1
        
        model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, output_dim)
        )
        
        # Initialize the weights randomly
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        return model
            
    def simulate_potential_outcomes(self, treatment_id, use_subset_features = False, batch_size=10000):
        module_functions = {}
        # revisit this logic again when heterogeneity is not 1
        for module_id in range(1, self.num_modules + 1):
            module_type = self.module_function_types[module_id - 1]
            module_function = globals()[f"{module_type}_module"]
            module_functions[module_id] = module_function
        self.module_functions = module_functions
        self.generate_module_weights(treatment_id)
        batch = []
        batch_counter = 0
        file_counter = 0
        if treatment_id == 0:
            self.processed_trees_0 = []
        else:
            self.processed_trees_1 = []
        for i, tree in enumerate(self.input_trees):
            input_tree = tree
            # sample module parameters
            
            input_processed_tree_dict, query_output = simulate_outcome(input_tree, treatment_id, self.module_functions, self.module_params_dict, self.max_depth, self.feature_dim, self.composition_type, self.use_subset_features)
            if self.composition_type == "hierarchical":
                input_processed_tree_dict = expand_features(input_processed_tree_dict)
                order_of_modules = add_order_of_modules(input_processed_tree_dict)
            input_dict = {}
            
            # save weights for all modules except mlp
            input_dict["module_params_dict"] = {k: v["w"] for k, v in self.module_params_dict.items() if self.module_function_types[k - 1] != "mlp"}
            
            input_dict["query_id"] = i
            input_dict["treatment_id"] = treatment_id
            input_dict["json_tree"] = input_processed_tree_dict
            input_dict["query_output"] = query_output
            input_dict["feature_dim"] = self.feature_dim
            if self.composition_type == "hierarchical":
                input_dict["order_of_modules"] = order_of_modules
            if treatment_id == 0:
                self.processed_trees_0.append(input_dict)
            else:
                self.processed_trees_1.append(input_dict)
            if treatment_id == 0:
                path = self.path_0
            else:
                path = self.path_1
            
            # batch.append(input_dict)
            # batch_counter += 1
            
            # if batch_counter >= batch_size:
            #     self._save_batch(batch, treatment_id, file_counter)
            #     batch = []
            #     batch_counter = 0
            #     file_counter += 1
        
            # # Save any remaining items in the last batch
            # if len(batch) > 0:
            #     self._save_batch(batch, treatment_id, file_counter)

    def _save_batch(self, batch, treatment_id, file_counter):
        path = self.path_0 if treatment_id == 0 else self.path_1
        filename = f"{path}/batch_{file_counter}.jsonl"
        
        with open(filename, 'w') as f:
            for item in batch:
                json.dump(item, f)
                f.write('\n')
        
        # batch save the input trees
            

            
            # with open("{}/input_tree_{}.json".format(path, i), "w") as f:
            #     json.dump(input_dict, f, indent=4)
    def save_csvs(self, source="processed"):
        if source == "json":
            process_trees_and_create_csvs(self.json_folders, self.csv_folder, self.config_folder, exactly_one_occurrence=False, domain=self.domain, source="json",composition_type=self.composition_type)
        elif source == "processed":
            process_trees_and_create_csvs([self.processed_trees_0, self.processed_trees_1], self.csv_folder, self.config_folder, exactly_one_occurrence=False, domain=self.domain,source="processed", composition_type=self.composition_type)

    def simulate_data(self):
        if self.resample:
            print("Resampling data")
            print("Generating input trees")
            self.generate_trees()
            print("Simulating potential outcomes for treatment 0")
            self.simulate_potential_outcomes(0, use_subset_features=self.use_subset_features)
            print("Simulating potential outcomes for treatment 1")
            self.simulate_potential_outcomes(1, use_subset_features=self.use_subset_features)
            self.save_csvs()

    def load_dataset(self, treatment_id, custom_path=None):
        self.input_trees_dict[treatment_id] = []
        if treatment_id == 0:
            if custom_path:
                data_folder = custom_path
            else:
                data_folder = self.path_0
        else:
            if custom_path:
                data_folder = custom_path
            else:
                data_folder = self.path_1

        files = os.listdir(data_folder)
        files = [file for file in files if "json" in file]
        for file in files:
            with open(data_folder + "/" + file, "r") as f:
                data = json.load(f)
                input_tree_dict = data["json_tree"]
                input_tree = Node.from_dict(input_tree_dict)
                query_id = data["query_id"]
                query_output = data["query_output"]
                self.input_trees_dict[treatment_id].append((query_id, input_tree, treatment_id, query_output))
    
    def create_observational_data(self, biasing_covariate, bias_strength):
        high_level_csv_filename = "{}_data_high_level_features.csv".format(self.domain)
        # open the file
        filepath = "{}/{}".format(self.csv_folder, high_level_csv_filename)
        
        with open(filepath, "r") as f:
            df_apo = pd.read_csv(filepath)
        
        df_sampled, df_cf_sampled, treatment_assignments = observational_sampling(df_apo, biasing_covariate, bias_strength, plot_folder = self.plot_folder)
        obs_folder = "{}/{}_{}/".format(self.obs_csv_folder, biasing_covariate, bias_strength)
        if not os.path.exists(obs_folder):
            os.makedirs(obs_folder)
        df_sampled.to_csv("{}/df_sampled.csv".format(obs_folder), index=False)
        df_cf_sampled.to_csv("{}/df_cf_sampled.csv".format(obs_folder), index=False)

        # save treatment assignments as json
        with open("{}/treatment_assignments.json".format(obs_folder), "w") as f:
            json.dump(treatment_assignments, f, indent=4)

    def create_iid_ood_split(self, split_type = "iid", test_size = 0.4, test_on_last_depth = False):
        high_level_csv_filename = "{}_data_high_level_features.csv".format(self.domain)
        # iid split just involves evenly splitting the data for each depth
        # open the file
        filepath = "{}/{}".format(self.csv_folder, high_level_csv_filename)
        with open(filepath, "r") as f:
            df_apo = pd.read_csv(filepath)
        
        grouped = df_apo.groupby('tree_depth')['query_id'].agg(list).reset_index()
        # all depths 
        depths = grouped["tree_depth"].unique()
        split_idx = int(len(depths) * (1 - test_size))
        train_depths, test_depths = depths[:split_idx], depths[split_idx:]
        if test_on_last_depth:
            test_depths = [depths[-1]]

        print("train depths: ", train_depths)
        print("test depths: ", test_depths)
        split_dict = {"train": [], "test": []}
        for i, row in grouped.iterrows():
            query_ids = row["query_id"]
            if split_type == "iid":
                train_ids = random.sample(query_ids, int(0.8*len(query_ids)))
                test_ids = [query_id for query_id in query_ids if query_id not in train_ids]
                split_dict["train"].extend(train_ids)
                split_dict["test"].extend(test_ids)
            elif split_type == "ood":
                # sort all depth query ids
                if row["tree_depth"] in train_depths:
                    train_ids = query_ids
                    split_dict["train"].extend(train_ids)
                else:
                    test_ids = query_ids
                    split_dict["test"].extend(test_ids)
            

        # save it in main folder
        split_folder = "{}/{}".format(self.csv_folder, split_type)
        if not os.path.exists(split_folder):
            os.makedirs(split_folder)

        with open("{}/train_test_split_qids.json".format(split_folder), "w") as f:
            json.dump(split_dict, f, indent=4)


class ManufacturingDataSampler:
    # make it inherit from the SyntheticDataSampler
    def __init__(
        self, 
        domain = "simulation_manufacturing",
        run_env = "local",
        
        trees_per_group = 2000,
        test_size = 0.2,
        resample = False
    ):
        
        self.domain = domain
        self.run_env = run_env
        if self.run_env == "local":
            self.base_dir = "/Users/ppruthi/research/compositional_models/compositional_models_cate/domains"
        else:
            self.base_dir = "/work/pi_jensen_umass_edu/ppruthi_umass_edu/compositional_models_cate/domains"
        self.trees_per_group = trees_per_group
        
        self.test_size = test_size
        self.resample = resample
        self.initialize_folders()

        
    def initialize_folders(self):
            # make path if doesn't exist
        self.data_path = "{}/{}/jsons/".format(self.base_dir, self.domain)
        folder_0 = "data_0"
        folder_1 = "data_1"
        self.path_0 = "{}/{}".format(self.data_path, folder_0)
        self.path_1 = "{}/{}".format(self.data_path, folder_1)
        if not os.path.exists(self.path_0):
            os.makedirs(self.path_0)
        else:
            # remove whole folder
            if self.resample:
                print("Reinitializing path 0")
                shutil.rmtree(self.path_0)
                os.makedirs(self.path_0, exist_ok=True)
        if not os.path.exists(self.path_1):
            os.makedirs(self.path_1)
        else:
            if self.resample:
                print("Reinitializing path 1")
                shutil.rmtree(self.path_1)
                os.makedirs(self.path_1, exist_ok=True)

        self.csv_folder = "{}/{}/csvs/".format(self.base_dir, self.domain)
        if not os.path.exists(self.csv_folder):
            os.makedirs(self.csv_folder)
        else:
            if self.resample:
                print("Reinitializing csv folder")
                shutil.rmtree(self.csv_folder)
                os.makedirs(self.csv_folder, exist_ok=True)
        self.json_folders = [self.path_0, self.path_1]

        self.obs_csv_folder = "{}/{}/observational_data/".format(self.base_dir, self.domain)
        if not os.path.exists(self.obs_csv_folder):
            os.makedirs(self.obs_csv_folder)

        self.plot_folder = "{}/{}/plots/".format(self.base_dir, self.domain)
        if not os.path.exists(self.plot_folder):
            os.makedirs(self.plot_folder)

        self.config_folder = "{}/{}/config/".format(self.base_dir, self.domain)
        if not os.path.exists(self.config_folder):
            os.makedirs(self.config_folder)

    def preprocess_and_save_jsons(self, output_key):
        def assign_scalar_output(node, key):
            node.output = node.output[key]
            for child in node.children:
                assign_scalar_output(child, key)
        
        def multiply_features_by_output(node):
            new_features = []
            for feature in node.features:
                new_features.append(feature * (node.output["Total_WIPs_produced"] + node.output["total_scrap_produced"]))
            node.features = new_features
            for child in node.children:
                multiply_features_by_output(child)
        
        orig_data_dir = "{}/{}/raw_data".format(self.base_dir, self.domain)
        dirs = os.listdir(orig_data_dir)
        out_dir = self.data_path
        for dir in dirs:
            if dir == "workers05":
                out_dir_name = self.path_0
            elif dir == "workers10":
                out_dir_name = self.path_1
            scenarios = os.listdir(os.path.join(orig_data_dir, dir))
            for scenario in scenarios:
                # json files
                json_files = os.listdir(os.path.join(orig_data_dir, dir, scenario))
                for jf in json_files:
                    if jf.endswith('.json'):
                        with open(os.path.join(orig_data_dir, dir, scenario, jf)) as f:
                            data = json.load(f)
                        demand = jf.split('_')[2].split('.')[0]
                        scenario_id = scenario.split('_')[1]
                        data["query_id"] = f"{scenario_id}_{demand}"
                        data["scenario_id"] = scenario_id
                        data["treatment_id"] = 0 if dir == "workers05" else 1
                        data["demand"] = demand

                        tree = data["json_tree"]
                        root = Node.from_dict(tree)
                        multiply_features_by_output(root)
                        assign_scalar_output(root, output_key)
                        
                        data["json_tree"] = root.to_dict()
                        data["query_output"] = root.output
                        

                        
                    # save json
                    out_path = os.path.join(out_dir, out_dir_name)
                    if not os.path.exists(out_path):
                        os.makedirs(out_path)
                    out_file = os.path.join(out_dir, out_dir_name, f"input_tree_{scenario_id}_{demand}.json")
                    with open(out_file, 'w') as f:
                        json.dump(data, f, indent=4)

    def create_csvs(self):
        process_trees_and_create_csvs(self.json_folders, self.csv_folder, self.config_folder, exactly_one_occurrence=False, domain=self.domain, source="json")

    def create_observational_data(self, biasing_covariate, bias_strength):
        high_level_csv_filename = "{}_data_high_level_features.csv".format(self.domain)
        # open the file
        filepath = "{}/{}".format(self.csv_folder, high_level_csv_filename)
        
        with open(filepath, "r") as f:
            df_apo = pd.read_csv(filepath)
        
        df_sampled, df_cf_sampled, treatment_assignments = observational_sampling(df_apo, biasing_covariate, bias_strength, plot_folder = self.plot_folder)
        obs_folder = "{}/{}_{}/".format(self.obs_csv_folder, biasing_covariate, bias_strength)
        if not os.path.exists(obs_folder):
            os.makedirs(obs_folder)
        df_sampled.to_csv("{}/df_sampled.csv".format(obs_folder), index=False)
        df_cf_sampled.to_csv("{}/df_cf_sampled.csv".format(obs_folder), index=False)
        print(obs_folder)

        # save treatment assignments as json
        with open("{}/treatment_assignments.json".format(obs_folder), "w") as f:
            json.dump(treatment_assignments, f, indent=4)

    def create_iid_ood_split(self, split_type = "iid", test_size = 0.2):
        high_level_csv_filename = "{}_data_high_level_features.csv".format(self.domain)
        # iid split just involves evenly splitting the data for each depth
        # open the file
        filepath = "{}/{}".format(self.csv_folder, high_level_csv_filename)
        with open(filepath, "r") as f:
            df_apo = pd.read_csv(filepath)
        
        grouped = df_apo.groupby('tree_depth')['query_id'].agg(list).reset_index()
        # all depths 
        depths = grouped["tree_depth"].unique()
        split_idx = int(len(depths) * (1 - test_size))
        train_depths, test_depths = depths[:split_idx], depths[split_idx:]
        print("train depths: ", train_depths)
        print("test depths: ", test_depths)
        split_dict = {"train": [], "test": []}
        for i, row in grouped.iterrows():
            query_ids = row["query_id"]
            if split_type == "iid":
                train_ids = random.sample(query_ids, int(0.8*len(query_ids)))
                test_ids = [query_id for query_id in query_ids if query_id not in train_ids]
                split_dict["train"].extend(train_ids)
                split_dict["test"].extend(test_ids)
            elif split_type == "ood":
                # sort all depth query ids
                if row["tree_depth"] in train_depths:
                    train_ids = query_ids
                    split_dict["train"].extend(train_ids)
                else:
                    test_ids = query_ids
                    split_dict["test"].extend(test_ids)
            

        # save it in main folder
        split_folder = "{}/{}".format(self.csv_folder, split_type)
        if not os.path.exists(split_folder):
            os.makedirs(split_folder)

        with open("{}/train_test_split_qids.json".format(split_folder), "w") as f:
            json.dump(split_dict, f, indent=4)

class MathsEvaluationDataSampler:
    def __init__(
        self, 
        outcomes_parallel=True,
        log_transform=False,
        domain = "maths_evaluation"
    ):
    
        self.outcomes_parallel = outcomes_parallel
        self.log_transform = log_transform
        self.input_trees_dict = {}
        self.domain = domain

    def initialize_folders(self):
        self.data_path = "jsons"
        folder_0 = "{}/jsons/data_0".format(self.domain)
        folder_1 = "{}/jsons/data_1".format(self.domain)
        self.path_0 = folder_0
        self.path_1 = folder_1
        if not os.path.exists(self.path_0):
            os.makedirs(self.path_0)
        if not os.path.exists(self.path_1):
            os.makedirs(self.path_1)
        self.csv_folder = "{}/csvs/".format(self.domain)
        if not os.path.exists(self.csv_folder):
            os.makedirs(self.csv_folder)
        self.json_folders = [self.path_0, self.path_1]
        self.obs_csv_folder = "{}/observational_data/".format(self.csv_folder)
        if not os.path.exists(self.obs_csv_folder):
            os.makedirs(self.obs_csv_folder)

        self.plot_folder = "{}/plots/".format(self.domain)
        if not os.path.exists(self.plot_folder):
            os.makedirs(self.plot_folder)

        self.config_folder = "{}/config/".format(self.domain)
        if not os.path.exists(self.config_folder):
            os.makedirs(self.config_folder)


    def load_expressions_list(self):
        path = "{}/expressions_list.json".format(self.domain)
        with open(path, "r") as f:
            expressions = json.load(f)["expressions"]
        self.expressions = expressions

    def preprocess_dataset(self, data_folder, treatment_id):
        files = os.listdir(data_folder)
        files = [file for file in files if "json" in file]
        for i, file in enumerate(files):
            with open(data_folder + "/" + file, "r") as f:
                data = json.load(f)
                expr_string = list(data.keys())[0]
                for exp in data[expr_string]:
                    matrix_size = exp["matrix_size"]
                    run_time = exp["total_runtime"]
                    expr_tree_dict = exp["expression_tree"]
                    expr_no = self.expressions.index(expr_string)
                    query_id =  "{}_{}".format(expr_no, matrix_size)
                    
                    expression_tree = ExpressionNode.from_dict_custom(expr_tree_dict)
                    
                    # outputs = expression_tree.return_outputs_as_list_preorder(outcomes_parallel = self.outcomes_parallel)
                    # if self.log_transform:
                    #     outputs = [np.log(out + 1) for out in outputs]
                    input_dict = {}
                    input_dict["query_id"] = query_id
                    input_dict["treatment_id"] = treatment_id
                    input_dict["json_tree"] = expression_tree.to_dict()
                    input_dict["query_output"] = run_time
                    if treatment_id == 0:
                        path = self.path_0
                    else:
                        path = self.path_1
                    
                    with open("{}/input_tree_{}.json".format(path, query_id), "w") as f:
                        json.dump(input_dict, f, indent=4)
                    

    def load_dataset(self, treatment_id, custom_path=None):
        self.input_trees_dict[treatment_id] = []
        if treatment_id == 0:
            if custom_path:
                data_folder = custom_path
            else:
                data_folder = self.path_0
        else:
            if custom_path:
                data_folder = custom_path
            else:
                data_folder = self.path_1

        files = os.listdir(data_folder)
        files = [file for file in files if "json" in file]
        for file in files:
            with open(data_folder + "/" + file, "r") as f:
                data = json.load(f)
                expr_tree_dict = data["json_tree"]
                expression_tree = ExpressionNode.from_dict(expr_tree_dict)
                query_id = data["query_id"]
                query_output = data["query_output"]
                self.input_trees_dict[treatment_id].append((query_id, treatment_id, expression_tree, query_output))

    def save_csvs(self):
        process_trees_and_create_csvs(self.json_folders, self.csv_folder, self.config_folder, exactly_one_occurrence=False, domain=self.domain)

    def create_observational_data(self, biasing_covariate, bias_strength):
        high_level_csv_filename = "{}_data_high_level_features.csv".format(self.domain)
        # open the file
        filepath = "{}/{}".format(self.csv_folder, high_level_csv_filename)
        
        with open(filepath, "r") as f:
            df_apo = pd.read_csv(filepath)
        
        df_sampled, df_cf_sampled, treatment_assignments = observational_sampling(df_apo, biasing_covariate, bias_strength, plot_folder = self.plot_folder)
        obs_folder = "{}/{}_{}/".format(self.obs_csv_folder, biasing_covariate, bias_strength)
        if not os.path.exists(obs_folder):
            os.makedirs(obs_folder)
        df_sampled.to_csv("{}/df_sampled.csv".format(obs_folder), index=False)
        df_cf_sampled.to_csv("{}/df_cf_sampled.csv".format(obs_folder), index=False)
        print(obs_folder)

        # save treatment assignments as json
        with open("{}/treatment_assignments.json".format(obs_folder), "w") as f:
            json.dump(treatment_assignments, f, indent=4)

    def create_iid_ood_split(self, split_type = "iid", test_size = 0.2):
        high_level_csv_filename = "{}_data_high_level_features.csv".format(self.domain)
        # iid split just involves evenly splitting the data for each depth
        # open the file
        filepath = "{}/{}".format(self.csv_folder, high_level_csv_filename)
        with open(filepath, "r") as f:
            df_apo = pd.read_csv(filepath)
        
        grouped = df_apo.groupby('tree_depth')['query_id'].agg(list).reset_index()
        # all depths 
        depths = grouped["tree_depth"].unique()
        split_idx = int(len(depths) * (1 - test_size))
        train_depths, test_depths = depths[:split_idx], depths[split_idx:]
        print("train depths: ", train_depths)
        print("test depths: ", test_depths)
        split_dict = {"train": [], "test": []}
        for i, row in grouped.iterrows():
            query_ids = row["query_id"]
            if split_type == "iid":
                train_ids = random.sample(query_ids, int(0.8*len(query_ids)))
                test_ids = [query_id for query_id in query_ids if query_id not in train_ids]
                split_dict["train"].extend(train_ids)
                split_dict["test"].extend(test_ids)
            elif split_type == "ood":
                # sort all depth query ids
                if row["tree_depth"] in train_depths:
                    train_ids = query_ids
                    split_dict["train"].extend(train_ids)
                else:
                    test_ids = query_ids
                    split_dict["test"].extend(test_ids)
            

        # save it in main folder
        split_folder = "{}/{}".format(self.csv_folder, split_type)
        if not os.path.exists(split_folder):
            os.makedirs(split_folder)

        with open("{}/train_test_split_qids.json".format(split_folder), "w") as f:
            json.dump(split_dict, f, indent=4)

    def create_scalers(self, split_type, biasing_covariate = None, bias_strength = None):
        # load all module files fro .csv folder
        module_files = [file for file in os.listdir(self.csv_folder) if "module" in file]
        high_level_csv_filename = "{}_data_high_level_features.csv".format(self.domain)

        # load split info
        split_folder = "{}/{}".format(self.csv_folder, split_type)
        with open("{}/train_test_split_qids.json".format(split_folder), "r") as f:
            split_dict = json.load(f)

        # load treatment info
        query_id_to_treatment = None
        if biasing_covariate and bias_strength:
            obs_folder = "{}/{}_{}/".format(self.obs_csv_folder, biasing_covariate, bias_strength)
            with open("{}/treatment_assignments.json".format(obs_folder), "r") as f:
                query_id_to_treatment = json.load(f)

        # high_level_output_scaler 
        high_level_df = pd.read_csv("{}/{}".format(self.csv_folder, high_level_csv_filename))
        high_level_output_scaler = StandardScaler()

        # filter based on split and treatment
        high_level_df = high_level_df[high_level_df["query_id"].isin(split_dict["train"])]
        if query_id_to_treatment:
            high_level_df["assigned_treatment"] = high_level_df["query_id"].apply(lambda x: query_id_to_treatment[str(x)])
            high_level_df = high_level_df[high_level_df["treatment_id"] == high_level_df["assigned_treatment"]]
        high_level_output_scaler.fit(np.log(high_level_df["query_output"].values).reshape(-1, 1))

        # save the high level output scaler
        split_folder = "{}/{}_{}/{}".format(self.obs_csv_folder, biasing_covariate, bias_strength, split_type)
        scaler_folder = "{}/scalers".format(split_folder)
        if not os.path.exists(scaler_folder):
            os.makedirs(scaler_folder)

        with open("{}/high_level_output_scaler.pkl".format(scaler_folder), "wb") as f:
            pickle.dump(high_level_output_scaler, f)
        # save the scalers
        
        # read module_features from config file
        with open(f"{self.config_folder}/{self.domain}_config.json", "r") as f:
            config = json.load(f)
            all_feature_names = config["module_feature_names"]


        for module_file in module_files:
            module_df = pd.read_csv("{}/{}".format(self.csv_folder, module_file))

            # replace nan or inf with 0
            module_df = module_df.replace([np.inf, -np.inf], np.nan)
            module_df = module_df.fillna(0)
            
            module_name = module_file.split("_")[1].split(".")[0]
            module_feature_names = all_feature_names[module_name]
            # filter out the train and test data
            module_df = module_df[module_df["query_id"].isin(split_dict["train"])]
            if query_id_to_treatment:
                module_df["assigned_treatment"] = module_df["query_id"].apply(lambda x: query_id_to_treatment[str(x)])
                module_df = module_df[module_df["treatment_id"] == module_df["assigned_treatment"]]
                

            # sort by module feature names
            module_df = module_df.sort_values(by=module_feature_names)
            # now pre-process the data
            input_scaler = RobustScaler()
            # for output, also do log transform and then scale
            output_scaler = StandardScaler()
            # fit the scalers
            input_scaler.fit(module_df[module_feature_names].values)
            output_scaler.fit(np.log(module_df["output"].values).reshape(-1, 1))

            # make the scalers in the split folder
            
            with open("{}/input_scaler_{}.pkl".format(scaler_folder, module_name), "wb") as f:
                pickle.dump(input_scaler, f)

            with open("{}/output_scaler_{}.pkl".format(scaler_folder, module_name), "wb") as f:
                pickle.dump(output_scaler, f)

class QueryExecutionDataSampler:
    def __init__(
        self, 
        outcomes_parallel=True,
        log_transform=False,
        domain = "query_execution",
        input_output_schema = None
    ):
    
        self.outcomes_parallel = outcomes_parallel
        self.log_transform = log_transform
        self.input_trees_dict = []
        self.domain = domain
        self.input_output_schema = input_output_schema
        

    def initialize_folders(self):
        self.data_path = "jsons"
        folder_0 = "{}/jsons/data_0".format(self.domain)
        folder_1 = "{}/jsons/data_1".format(self.domain)
        self.path_0 = folder_0
        self.path_1 = folder_1
        if not os.path.exists(self.path_0):
            os.makedirs(self.path_0)
        if not os.path.exists(self.path_1):
            os.makedirs(self.path_1)
        self.csv_folder = "{}/csvs/".format(self.domain)
        if not os.path.exists(self.csv_folder):
            os.makedirs(self.csv_folder)
        self.json_folders = [self.path_0, self.path_1]
        self.config_folder = "{}/config/".format(self.domain)
        if not os.path.exists(self.config_folder):
            os.makedirs(self.config_folder)

    def preprocess_dataset(self, data_folder, treatment_id):
        files = os.listdir(data_folder)
        files = [file for file in files if "json" in file]
        for i, file in enumerate(files):
            # if file doesn't end with json, skip
            if "json" not in file:
                continue
            with open(data_folder + "/" + file, "r") as f:
                data = json.load(f)
                # file name is postgres_query_query_id_tid_runid.json
                # get query id, tid and runid from file
                filename = file.split(".")[0]
                query_id = int(filename.split("_")[2])
                tid = int(filename.split("_")[3])
                runid = int(filename.split("_")[4])
                run_time = data["json_result"][0]["Execution Time"]
                query_plan_tree_dict = data["json_result"][0]["Plan"]
                query_plan_tree = QueryPlanNode.from_dict_custom(query_plan_tree_dict, self.input_output_schema)
                # outputs = query_plan_tree.return_outputs_as_list_preorder(outcomes_parallel = self.outcomes_parallel)
                # if self.log_transform:
                #     outputs = [np.log(out + 1) for out in outputs]
                #     run_time = np.log(run_time + 1)
                input_dict = {}
                input_dict["query_id"] = query_id
                input_dict["treatment_id"] = treatment_id
                input_dict["json_tree"] = query_plan_tree.to_dict(input_output_features_schema=self.input_output_schema)
                input_dict["query_output"] = run_time
                if treatment_id == 0:
                    path = self.path_0
                else:
                    path = self.path_1
                
                with open("{}/input_tree_{}.json".format(path, query_id), "w") as f:
                    json.dump(input_dict, f, indent=4)
                    

    def load_dataset(self, treatment_id, custom_path=None):
        self.input_trees_dict[treatment_id] = []
        if treatment_id == 0:
            if custom_path:
                data_folder = custom_path
            else:
                data_folder = self.path_0
        else:
            if custom_path:
                data_folder = custom_path
            else:
                data_folder = self.path_1

        files = os.listdir(data_folder)
        files = [file for file in files if "json" in file]
        for file in files:
            with open(data_folder + "/" + file, "r") as f:
                data = json.load(f)
                query_plan_dict = data["json_tree"]
                query_plan_tree = QueryPlanNode.from_dict(query_plan_dict)
                query_id = data["query_id"]
                query_output = data["query_output"]
                self.input_trees_dict[treatment_id]((query_id, treatment_id, query_plan_tree, query_output))

    def save_csvs(self):
        process_trees_and_create_csvs(self.json_folders, self.csv_folder, self.config_folder, exactly_one_occurrence=False, domain=self.domain)

    


if __name__ == "__main__":

    # test the manufacturing data sampler
    sampler = ManufacturingDataSampler(resample=True)
    sampler.preprocess_and_save_jsons("throughput")
    sampler.create_csvs()

    # # test the synthetic data sampler
    # num_modules = 10
    # feature_dim = 3
    # composition_type = "parallel"
    # fixed_structure = True
    # max_depth = num_modules
    # num_trees = 5000
    # seed = 42
    # module_function_type = "mlp"
    # resample = True
    # covariates_shared = True
    # use_subset_features = False
    # run_env = "local"
    # heterogeneity = 1
    # data_dist = "uniform"
    # systematic = True
    # domain = "synthetic_data"
    # sampler = SyntheticDataSampler(num_modules=num_modules, feature_dim=feature_dim, composition_type=composition_type, fixed_structure=fixed_structure, max_depth=max_depth, num_trees=num_trees, seed=seed, module_function_type=module_function_type, resample=resample, covariates_shared=covariates_shared, use_subset_features=use_subset_features, run_env=run_env, heterogeneity=heterogeneity, data_dist=data_dist, systematic=systematic)
    # sampler.simulate_data()

    # if run_env == "local":
    #     base_dir = "/Users/ppruthi/research/compositional_models/compositional_models_cate/domains"
    # else:
    #     base_dir = "/work/pi_jensen_umass_edu/ppruthi_umass_edu/compositional_models_cate/domains"

    # main_dir = f"{base_dir}/{domain}"
    # csv_path = f"{main_dir}/csvs/fixed_structure_{fixed_structure}_outcomes_{composition_type}_systematic_{systematic}"
    # obs_data_path = f"{main_dir}/observational_data/fixed_structure_{fixed_structure}_outcomes_{composition_type}_systematic_{systematic}"
    # data_path = f"{csv_path}/{domain}_data_high_level_features.csv"
    # data = pd.read_csv(data_path)

    # # print unique values of tree depth
    # print(data["tree_depth"].unique())



    # # test the maths evaluation data sampler
    # data_sampler = MathsEvaluationDataSampler()
    # data_sampler.initialize_folders()
    # data_sampler.load_expressions_list()
    # data_sampler.save_csvs()
    # data_sampler.create_observational_data("matrix_size", 10)
    # data_sampler.create_iid_ood_split("iid")
    # data_sampler.create_iid_ood_split("ood")
    # create scalers for each module for each split
    # data_sampler.create_scalers("iid", "matrix_size", 10)
    # data_sampler.create_scalers("ood", "matrix_size", 10)
    # print(data_sampler)
    # main_dir = "/Users/ppruthi/research/novelty_accommodation/causal_effect_estimation_experiments/maths_evaluation"
    # data_folder_0 = "{}/jsons/results_test".format(main_dir)
    # data_folder_1 = "{}/jsons/results_test_manjaro".format(main_dir)
    # data_sampler.preprocess_dataset(data_folder_0, 0)
    # data_sampler.preprocess_dataset(data_folder_1, 1)
    # data_sampler.load_dataset(0)
    # data_sampler.load_dataset(1)
    # data_sampler.save_csvs()
    # print(len(data_sampler.expression_trees))
    

    # test the query execution data sampler
    # input_output_schema_file_path = "/Users/ppruthi/research/novelty_accommodation/compositional_models_cate/domains/query_execution/jsons/input_output_features_schema.json"
    # with open(input_output_schema_file_path, "r") as f:
    #     schema = json.load(f)
    # data_sampler = QueryExecutionDataSampler(input_output_schema = schema)
    # data_sampler.initialize_folders()
    # print(data_sampler)
    # main_dir = "/Users/ppruthi/research/novelty_accommodation/postgres_data/queries/data/data_v1/post_processed_queries"
    # data_folder_0 = "{}/index_0_memory_0_page_0".format(main_dir)
    # data_folder_1 = "{}/index_0_memory_2_page_0".format(main_dir)
    # data_sampler.preprocess_dataset(data_folder_0, 0)
    # data_sampler.preprocess_dataset(data_folder_1, 1)
    # data_sampler.load_dataset(0)
    # data_sampler.load_dataset(1)
    # data_sampler.save_csvs()
    # print(len(data_sampler.query_plans))

        



