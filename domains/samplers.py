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
from sklearn.gaussian_process import GaussianProcessRegressor
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
    # sampler = ManufacturingDataSampler(resample=True)
    # sampler.preprocess_and_save_jsons("throughput")
    # sampler.create_csvs()

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
    input_output_schema_file_path = "/Users/ppruthi/research/novelty_accommodation/compositional_models_cate/domains/query_execution/jsons/input_output_features_schema.json"
    with open(input_output_schema_file_path, "r") as f:
        schema = json.load(f)
    data_sampler = QueryExecutionDataSampler(input_output_schema = schema)
    data_sampler.initialize_folders()
    print(data_sampler)
    main_dir = "/Users/ppruthi/research/novelty_accommodation/postgres_data/queries/data/data_v1/post_processed_queries"
    data_folder_0 = "{}/index_0_memory_0_page_0".format(main_dir)
    data_folder_1 = "{}/index_0_memory_2_page_0".format(main_dir)
    data_sampler.preprocess_dataset(data_folder_0, 0)
    data_sampler.preprocess_dataset(data_folder_1, 1)
    data_sampler.load_dataset(0)
    data_sampler.load_dataset(1)
    data_sampler.save_csvs()
    # print(len(data_sampler.query_plans))

        



