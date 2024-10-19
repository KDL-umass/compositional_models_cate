# TODO: add more modules
# Create a Gaussian Process Regressor


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
from domains.synthetic_modules import *
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
        test_size = 0.2,
        noise = 0.0
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
        self.noise = noise
        
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
        
        input_dim = self.feature_dim

        # Generate a base set of weights
        np.random.seed(base_seed)
        
        if not self.use_subset_features:
            if self.composition_type == "hierarchical":
                input_dim = self.feature_dim + 1
            base_weights = np.random.uniform(0.1, 1, 2 * input_dim + 1)
            module_feature_dim = input_dim
        else:
            num_modules = self.num_modules
            feature_dim = input_dim
            feature_dim_per_module = int(feature_dim / num_modules)
            if self.composition_type == "hierarchical":
                feature_dim_per_module += 1
            base_weights = np.random.uniform(0.1, 1, 2 * feature_dim_per_module + 1)
            module_feature_dim = feature_dim_per_module

        num_same_weight_modules = int(self.num_modules * (1 - self.heterogeneity))

        input_dim = module_feature_dim
        if "mlp" in self.module_function_types:
            base_mlp = self.create_mlp(input_dim)

        for module_id in range(1, self.num_modules + 1):
            Mj = module_feature_dim
            
            if module_id <= num_same_weight_modules:
                module_type = self.module_function_types[0]
                if module_type == 'mlp':
                    model = copy.deepcopy(base_mlp)
                else:
                    w = base_weights
                
            else:
                np.random.seed(base_seed + module_id)
                module_type = self.module_function_types[module_id - num_same_weight_modules - 1]
                if module_type == 'mlp':
                    model = self.create_mlp(input_dim)
                    for layer in model.modules():
                        if isinstance(layer, nn.Linear):
                            nn.init.xavier_uniform_(layer.weight)
                            nn.init.zeros_(layer.bias)
                # elif module_type == 'linear':
                #     slope = np.random.uniform(-1, 1)
                #     intercept = np.random.uniform(-1, 1)
                #     model = LinearFunction(slope, intercept)
                # elif module_type == 'affine':
                #     matrix = np.random.uniform(-1, 1, size=(Mj,))
                #     vector = np.random.uniform(-1, 1, size=(1,))
                #     model = AffineFunction(matrix, vector)
                # elif module_type == 'polynomial':
                #     coefficients = np.random.uniform(-1, 1, size=Mj+1)
                #     model = PolynomialFunction(coefficients)
                else:
                    w = np.random.uniform(0.1, 1, 2 * (Mj) + 1)
                
            if module_type == 'mlp':
                self.module_params_dict[module_id] = {"model": model}
            # elif module_type in ['linear', 'affine', 'polynomial']:
            #     self.module_params_dict[module_id] = {"model": model}
            else:
                self.module_params_dict[module_id] = {"w": w.tolist()}

        print(self.module_params_dict[module_id])
        
        
    # def generate_module_weights(self, treatment_id):
    #     base_seed = self.seed + treatment_id * 100
    #     if self.composition_type == "hierarchical":
    #         input_dim = self.feature_dim + 1
    #     else:
    #         input_dim = self.feature_dim
    #     # Generate a base set of weights
    #     np.random.seed(base_seed)
        
    #     if not self.use_subset_features:
    #         base_weights = np.random.uniform(0.1, 1, 2 * (input_dim) + 1)
    #         module_feature_dim = input_dim
    #     else:
    #         # set feature dim equally across all modules
    #         num_modules = self.num_modules
    #         feature_dim = input_dim
    #         feature_dim_per_module = int(feature_dim / num_modules)
    #         base_weights = np.random.uniform(0.1, 1, 2 * (feature_dim_per_module) + 1)
    #         module_feature_dim = feature_dim_per_module
        
    #     # Determine how many modules will have the same weights
    #     num_same_weight_modules = int(self.num_modules * (1 - self.heterogeneity))
        
    #     # Define the MLP architecture if needed
        
    #     input_dim = module_feature_dim
    #     if "mlp" in self.module_function_types:
    #         base_mlp = self.create_mlp(input_dim)
        
    #     for module_id in range(1, self.num_modules + 1):
    #         Mj = module_feature_dim # feature dim for this module
            
    #         if module_id <= num_same_weight_modules:
    #             # Use exactly the same weights for these modules
    #             module_type = self.module_function_types[0]
    #             if module_type == 'mlp':
    #                 model = copy.deepcopy(base_mlp)
    #             else:
    #                 w = base_weights
                
    #         else:
    #             # Use distinct weights for the remaining modules
    #             np.random.seed(base_seed + module_id)
    #             module_type = self.module_function_types[module_id - num_same_weight_modules - 1]
    #             if module_type == 'mlp':
    #                 model = self.create_mlp(input_dim)
    #                 # Initialize the weights randomly
    #                 for layer in model.modules():
    #                     if isinstance(layer, nn.Linear):
    #                         nn.init.xavier_uniform_(layer.weight)
    #                         nn.init.zeros_(layer.bias)
    #             else:
    #                w = np.random.uniform(0.1, 1, 2 * (Mj) + 1)
                
            
    #         if module_type == 'mlp':
    #             self.module_params_dict[module_id] = {"model": model}
    #         elif module_type == 'polyval':
    #                 w = coeffs[str(module_id)][treatment_id]
    #                 self.module_params_dict[module_id] = {"w": w}
    #         else:
    #             self.module_params_dict[module_id] = {"w": w.tolist()}

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

    def simulate_potential_outcomes(self, treatment_id, use_subset_features=False, batch_size=10000):
        module_functions = {}
        
        self.generate_module_weights(treatment_id)
        for module_id in range(1, self.num_modules + 1):
            module_type = self.module_function_types[module_id - 1]
            module_function = globals()[f"{module_type}_module"]
            module_functions[module_id] = module_function
        self.module_functions = module_functions
        
        
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
            
            input_processed_tree_dict, query_output = simulate_outcome(input_tree, treatment_id, self.module_functions, self.module_params_dict, self.max_depth, self.feature_dim, self.composition_type, self.use_subset_features, noise=self.noise)
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
        print("Processing trees and creating csvs here: ", self.csv_folder)
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

    def create_iid_ood_split(self, split_type = "iid", test_size = 0.4, test_on_last_depth = False, ood_type = "depth"):
        high_level_csv_filename = "{}_data_high_level_features.csv".format(self.domain)
        # iid split just involves evenly splitting the data for each depth
        # open the file
        filepath = "{}/{}".format(self.csv_folder, high_level_csv_filename)
        with open(filepath, "r") as f:
            df_apo = pd.read_csv(filepath)
        
        # find the query ids for each depth to split based on iid or ood
        grouped = df_apo.groupby('tree_depth')['query_id'].agg(list).reset_index()
        # all depths 
        depths = grouped["tree_depth"].unique()
        if split_type == "ood":
            split_idx = int(len(depths) * (1 - test_size))
            train_depths, test_depths = depths[:split_idx], depths[split_idx:]
            if test_on_last_depth:
                test_depths = [depths[-1]]

            print("train depths: ", train_depths)
            print("test depths: ", test_depths)




        split_dict = {"train": [], "test": []}
        for i, row in grouped.iterrows():
            query_ids = row["query_id"]
            # remove duplicates
            query_ids = list(set(query_ids))
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
        print(len(split_dict["train"]), len(split_dict["test"]))
        # sort the query ids
        split_dict["train"] = sorted(split_dict["train"])
        split_dict["test"] = sorted(split_dict["test"])
        # save it in main folder
        split_folder = "{}/{}".format(self.csv_folder, split_type)
        if not os.path.exists(split_folder):
            os.makedirs(split_folder)

        with open("{}/train_test_split_qids.json".format(split_folder), "w") as f:
            json.dump(split_dict, f, indent=4)


