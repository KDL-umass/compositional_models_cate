import os
import json
import pickle
import pandas as pd
import numpy as np
from models.MoE import *
from models.utils import  get_ground_truth_effects

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle

def load_module_data(csv_path):
    module_files = [f for f in os.listdir(csv_path) if "module" in f]
    return {f: pd.read_csv(os.path.join(csv_path, f)) for f in module_files}

def get_estimated_effects(df, qids):
    return df[df["query_id"].isin(qids)].set_index("query_id")["estimated_effect"].to_dict()

def load_treatment_assignments(obs_data_path, bias_strength):
    path = f"{obs_data_path}/feature_sum_{bias_strength}/treatment_assignments.json"
    with open(path, "r") as f:
        return json.load(f)
# Example usage:
# scaled_data = scale_module_data(
#     csv_folder="/path/to/csv",
#     obs_csv_folder="/path/to/obs_csv",
#     config_folder="/path/to/config",
#     domain="your_domain",
#     split_type="train",
#     biasing_covariate="some_covariate",
#     bias_strength="some_strength"
# )
def scale_module_data(module_df, scaler_path, module_id):
    input_scaler = pickle.load(open(f"{scaler_path}/input_scaler_{module_id}.pkl", "rb"))
    output_scaler = pickle.load(open(f"{scaler_path}/output_scaler_{module_id}.pkl", "rb"))
    
    feature_names = [x for x in module_df.columns if "feature" in x]
    module_df = module_df.sort_values(by=feature_names)
    module_df[["output"]] = output_scaler.transform(module_df[["output"]].values.reshape(-1, 1))
    module_df[feature_names] = input_scaler.transform(module_df[feature_names])
    
    return module_df

def get_additive_model_effects(csv_path, obs_data_path, train_qids, test_qids, hidden_dim=32, epochs=100, batch_size=64, output_dim=1, underlying_model_class="MLP", scale = True, scaler_path=None, bias_strength=0, domain="synthetic_data", hl_train_df=None, hl_test_df=None, hl_covariates=None, use_high_level_features=False, num_modules=3, num_feature_dimensions=3):
    module_files = os.listdir(f"{csv_path}/")
    
    module_files = [x for x in module_files if "module" in x]

    # read all the module files
    module_data = {}
    for module_file in module_files:
        module_data[module_file] = pd.read_csv(f"{csv_path}/{module_file}")
        # sort the data by query_id
        module_data[module_file] = module_data[module_file].sort_values("query_id")
    

    query_id_treatment_id_json_path = f"{obs_data_path}/feature_sum_{bias_strength}/treatment_assignments.json"
    with open(query_id_treatment_id_json_path, "r") as f:
        query_id_treatment_id = json.load(f)
    train_data = {}
    test_data = {}
    for module_file in module_files:
        module_df = module_data[module_file]
        if domain == "simulation_manufacturing":
            module_df["demand"] = module_df["query_id"].apply(lambda x: int(x.split("_")[1]))

        if scale == True:
            # load the input and output scalers
            module_id = module_file.split(".")[0].split("_")[-1]
            module_input_scaler = pickle.load(open(f"{scaler_path}/input_scaler_{module_id}.pkl", "rb"))
            module_output_scaler = pickle.load(open(f"{scaler_path}/output_scaler_{module_id}.pkl", "rb"))
            module_feature_names = [x for x in module_df.columns if "feature" in x]
            # module_df = module_df.sort_values(by=module_feature_names)
            # print(module_df[["query_id", "treatment_id", "output"]].head())
            module_df[["output"]] = module_output_scaler.transform(module_df[["output"]].values.reshape(-1, 1))
            # print(module_df[["query_id", "treatment_id", "output"]].head())
            # print(module_file, module_feature_names)
            module_df[module_feature_names] = module_input_scaler.transform(module_df[module_feature_names])
            

        # print(module_df["query_id"].head())
        # print(query_id_treatment_id.keys())
        module_df["assigned_treatment_id"] = module_df["query_id"].apply(lambda x: query_id_treatment_id[str(x)])
        module_df = module_df[module_df["treatment_id"] == module_df["assigned_treatment_id"]]
        # drop the assigned treatment id
        module_df.drop("assigned_treatment_id", axis=1, inplace=True)
        # sort the data by query_id
        module_df = module_df.sort_values("query_id")
        # split the data into train and test
        train_data[module_file] = module_df[module_df["query_id"].isin(train_qids)]
        test_data[module_file] = module_df[module_df["query_id"].isin(test_qids)]
    
    for module_file in module_files:
        train_df = train_data[module_file]
        test_df = test_data[module_file]
        if domain == "synthetic_data":
            if use_high_level_features:
                # use hl_covariates from  hl_train_df for same query_id
                covariates = hl_covariates 
                train_df = train_df.merge(hl_train_df, on="query_id", how="left")
                # rename treatment_id_x to treatment_id
                train_df.rename(columns={"treatment_id_x": "treatment_id"}, inplace=True)
                test_df = test_df.merge(hl_test_df, on="query_id", how="left")
                # rename treatment_id_x to treatment_id
                test_df.rename(columns={"treatment_id_x": "treatment_id"}, inplace=True)
            else:
                covariates = [x for x in train_df.columns if "feature" in x]
        else:
            covariates = [x for x in train_df.columns if "output" not in x and "query_id" not in x and "treatment_id" not in x]
        print(module_file, covariates)
        treatment = "treatment_id"
        outcome = "output"
        input_dim = len(covariates)
        if input_dim > hidden_dim:
            hidden_dim = (input_dim + 1)*2
        if underlying_model_class == "MLP":
            expert_model = BaselineModel(input_dim + 1, hidden_dim, output_dim)
        else:
            expert_model = BaselineLinearModel(input_dim + 1, output_dim)
        
        expert_model, train_losses, val_losses = train_model(expert_model, train_df, covariates, treatment, outcome, epochs, batch_size, num_modules=num_modules, num_feature_dimensions=num_feature_dimensions,model_name="Baseline")
        causal_effect_estimates_train = predict_model(expert_model, train_df, covariates, num_modules=num_modules, num_feature_dimensions=num_feature_dimensions, model_name="Baseline")
        causal_effect_estimates_test = predict_model(expert_model, test_df, covariates, num_modules=num_modules, num_feature_dimensions=num_feature_dimensions, model_name="Baseline")
        train_df["estimated_effect"] = causal_effect_estimates_train
        test_df["estimated_effect"] = causal_effect_estimates_test
        train_data[module_file] = train_df
        test_data[module_file] = test_df
        # save the trained model
        # torch.save(expert_model.state_dict(), f"{scaler_path}/trained_models_hl_{use_high_level_features}/{module_file}.pt")

    # now for each module, get the ground truth and estimated effects
    additive_ground_truth_effects_train = {}
    additive_estimated_effects_train = {}
    additive_ground_truth_effects_test = {}
    additive_estimated_effects_test = {}
    modules_csvs_train = {}
    modules_csvs_test = {}
    modules_train_sizes = {}
    for module_file in module_files:
        module_name = module_file.split(".")[0]
        train_df = train_data[module_file]
        modules_train_sizes[module_name] = len(train_df)
        test_df = test_data[module_file]
       
        module_causal_effect_dict_train = get_ground_truth_effects(module_data[module_file], train_qids, treatment_col="treatment_id", outcome_col="output")
        module_causal_effect_dict_test = get_ground_truth_effects(module_data[module_file], test_qids, treatment_col="treatment_id", outcome_col="output")
        module_estimated_effects_train = get_estimated_effects(train_df, train_qids)
        module_estimated_effects_test = get_estimated_effects(test_df, test_qids)
        # have a combined df with ground truth and estimated effects based on the query_id
        module_gt_effect_df_train = pd.DataFrame.from_dict(module_causal_effect_dict_train, orient="index", columns=["ground_truth_effect"])
        module_gt_effect_df_test = pd.DataFrame.from_dict(module_causal_effect_dict_test, orient="index", columns=["ground_truth_effect"])

        # add the estimated effects based on query ids with same order
        module_estimated_effects_train_df = pd.DataFrame.from_dict(module_estimated_effects_train, orient="index", columns=["estimated_effect"])
        module_estimated_effects_test_df = pd.DataFrame.from_dict(module_estimated_effects_test, orient="index", columns=["estimated_effect"])
        # merge 
        module_combined_train_df = pd.concat([module_gt_effect_df_train, module_estimated_effects_train_df], axis=1)
        module_combined_test_df = pd.concat([module_gt_effect_df_test, module_estimated_effects_test_df], axis=1)
        modules_csvs_train[module_name] = module_combined_train_df
        modules_csvs_test[module_name] = module_combined_test_df
        

        
        if len(additive_estimated_effects_test) == 0:
            additive_ground_truth_effects_test = module_causal_effect_dict_test
            additive_estimated_effects_test = module_estimated_effects_test
        else:
            # add the effects
            # handle the case where the query ids are not the same, check if k is in the other dict oyherwise add 0
            for k, v in module_causal_effect_dict_test.items():
                if k not in additive_ground_truth_effects_test:
                    additive_ground_truth_effects_test[k] = v
                else:
                    additive_ground_truth_effects_test[k] += v
                if k not in additive_estimated_effects_test:
                    additive_estimated_effects_test[k] = module_estimated_effects_test[k]
                else:
                    additive_estimated_effects_test[k] += module_estimated_effects_test[k]

        if len(additive_estimated_effects_train) == 0:
            additive_ground_truth_effects_train = module_causal_effect_dict_train
            additive_estimated_effects_train = module_estimated_effects_train
        else:
            for k, v in module_causal_effect_dict_train.items():
                if k not in additive_ground_truth_effects_train:
                    additive_ground_truth_effects_train[k] = v
                else:
                    additive_ground_truth_effects_train[k] += v
                if k not in additive_estimated_effects_train:
                    additive_estimated_effects_train[k] = module_estimated_effects_train[k]
                else:
                    additive_estimated_effects_train[k] += module_estimated_effects_train[k]

            
    additive_gt_effect_train_df = pd.DataFrame.from_dict(additive_ground_truth_effects_train, orient="index", columns=["ground_truth_effect"])
    additive_estimated_effect_train_df = pd.DataFrame.from_dict(additive_estimated_effects_train, orient="index", columns=["estimated_effect"])
    additive_combined_train_df = pd.concat([additive_gt_effect_train_df, additive_estimated_effect_train_df], axis=1)
    # have query_id as index
    additive_combined_train_df.index.name = "query_id"
    # make it a column
    additive_combined_train_df.reset_index(inplace=True)

    additive_gt_effect_test_df = pd.DataFrame.from_dict(additive_ground_truth_effects_test, orient="index", columns=["ground_truth_effect"])
    additive_estimated_effect_test_df = pd.DataFrame.from_dict(additive_estimated_effects_test, orient="index", columns=["estimated_effect"])
    additive_combined_test_df = pd.concat([additive_gt_effect_test_df, additive_estimated_effect_test_df], axis=1)
    # have query_id as index
    additive_combined_test_df.index.name = "query_id"
    # make it a column
    additive_combined_test_df.reset_index(inplace=True)


    return additive_combined_train_df, additive_combined_test_df, modules_csvs_train, modules_csvs_test, modules_train_sizes


import os
import json
import pickle
import pandas as pd
import numpy as np
from models.MoE import *
from models.utils import  get_ground_truth_effects
# import tqdm notebook
from tqdm import tqdm

def create_module_scalers(csv_path, obs_data_path, scaler_path, bias_strength, composition_type, data_dist):
    # Load module data and treatment assignments
    scalers = {}
    module_data = load_module_data(csv_path)
    query_id_treatment_id = load_treatment_assignments(obs_data_path, bias_strength)

    # Create scaler directory
    # scaler_path = f"{obs_data_path}/feature_sum_{bias_strength}/scalers"
    os.makedirs(scaler_path, exist_ok=True)

    for module_file, module_df in module_data.items():
        module_id = module_file.split(".")[0].split("_")[-1]
        scalers[module_id] = {}
        
        # Replace inf with nan and then fill nan with 0
        module_df = module_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # # Filter data based on train_qids and treatment assignments
        # module_df = module_df[module_df["query_id"].isin(train_qids)]
        # module_df["assigned_treatment_id"] = module_df["query_id"].apply(lambda x: query_id_treatment_id[str(x)])
        # module_df = module_df[module_df["treatment_id"] == module_df["assigned_treatment_id"]]
        
        # Identify feature columns
        if composition_type == "parallel":
            feature_names = [x for x in module_df.columns if "feature" in x]
        else:
            feature_names = [x for x in module_df.columns if "feature" in x or x == "child_output"]
        
        # Sort by feature names
        module_df = module_df.sort_values(by=feature_names)
        
        # Create and fit scalers
        if data_dist == "normal":
            input_scaler = StandardScaler()
        else:
            input_scaler = MinMaxScaler()
        output_scaler = StandardScaler()
        
        input_scaler.fit(module_df[feature_names].values)
        output_scaler.fit(module_df[["output"]].values)
        
        # Save scalers
        with open(f"{scaler_path}/input_scaler_{module_id}.pkl", "wb") as f:
            pickle.dump(input_scaler, f)
        
        if composition_type == "parallel":
            with open(f"{scaler_path}/output_scaler_{module_id}.pkl", "wb") as f:
                pickle.dump(output_scaler, f)

        scalers[module_id]["input_scaler"] = input_scaler
        scalers[module_id]["output_scaler"] = output_scaler
            
            
        print(f"Created scalers for module {module_id}")

    print(f"All scalers created and saved in {scaler_path}")
    return scalers

def split_modular_data(csv_path, obs_data_path, train_qids, test_qids, scalers, scale, bias_strength, composition_type):
    # get the names of all the module files
    module_files = os.listdir(f"{csv_path}/")
    module_files = [x for x in module_files if "module" in x]
    # Read all the module files
    module_data = {f: pd.read_csv(os.path.join(csv_path, f)).sort_values("query_id") for f in module_files}

    # Load treatment assignments
    with open(os.path.join(obs_data_path, f"feature_sum_{bias_strength}", "treatment_assignments.json"), "r") as f:
        query_id_treatment_id = json.load(f)

    # split the data into train and test
    train_data, test_data = {}, {}
    for module_file in module_files:
        module_df = module_data[module_file]
        module_id = module_file.split(".")[0].split("_")[-1]
        # skip scaling for outputs for the sequential model
        if scale:
            module_id = module_file.split(".")[0].split("_")[-1]
            module_input_scaler = scalers[module_id]["input_scaler"]
            module_output_scaler = scalers[module_id]["output_scaler"]
            module_feature_names = [x for x in module_df.columns if "feature" in x or x == "child_output"]
            module_df[["output"]] = module_output_scaler.transform(module_df[["output"]].values.reshape(-1, 1))
            module_df[module_feature_names] = module_input_scaler.transform(module_df[module_feature_names])

        # Filter data based on train_qids and treatment assignments
        module_df["assigned_treatment_id"] = module_df["query_id"].apply(lambda x: query_id_treatment_id[str(x)])
        module_df = module_df[module_df["treatment_id"] == module_df["assigned_treatment_id"]].drop("assigned_treatment_id", axis=1)
        
        train_data[module_file] = module_df[module_df["query_id"].isin(train_qids)]
        test_data[module_file] = module_df[module_df["query_id"].isin(test_qids)]
    
    return train_data, test_data, module_files


def train_modular_model_with_po(train_data, module_files, domain, model_misspecification, underlying_model_class, hidden_dim, epochs, batch_size, output_dim, hl_train_df, hl_covariates, use_high_level_features, num_modules=3, num_feature_dimensions=3):
    # Train models with access to fine-grained potential outcomes
    trained_models = {}
    for module_file in module_files:
        # consists of already scaled 
        train_df = train_data[module_file]
        if domain == "synthetic_data":
            if use_high_level_features:
                # use hl_covariates from  hl_train_df for same query_id
                covariates = hl_covariates + [x for x in train_df.columns if "child_output" in x]
                train_df = train_df.merge(hl_train_df, on="query_id", how="left")
                # rename treatment_id_x to treatment_id
                train_df.rename(columns={"treatment_id_x": "treatment_id"}, inplace=True)
            else:
                # if sequential composition, use child_output (parent) as a feature
                covariates = [x for x in train_df.columns if "feature" in x or x == "child_output"]

            # try removing feature_0 to induce model misspecification # other way is to use the linear model
            if model_misspecification:
                covariates = [x for x in covariates if "feature_0" not in x]
        else:
            covariates = [x for x in train_df.columns if x not in ["output", "query_id", "treatment_id"]]

        treatment = "treatment_id"
        outcome = "output"
        input_dim = len(covariates)
        print(input_dim, hidden_dim)
        if input_dim > hidden_dim:
            hidden_dim = (input_dim + 1) * 2

        # underlying baseline model (just models E[Y|X, T] for now), later replace with probablistic model that models P(Y|X, T)
        if underlying_model_class == "MLP":
            expert_model = BaselineModel(input_dim + 1, hidden_dim, output_dim)
        else:
            expert_model = BaselineLinearModel(input_dim + 1, output_dim)

        # Train the model
        expert_model, _, _ = train_model(expert_model, train_df, covariates, treatment, outcome, epochs, batch_size, num_modules=num_modules, num_feature_dimensions=num_feature_dimensions, model_name="Baseline")
        # store the trained model
        trained_models[module_file] = expert_model
    return trained_models



def predict(node, query_id, treatment_id, trained_models, test_data_dict, hl_test_df, hl_covariates, use_high_level_features, num_modules=3, num_feature_dimensions=3):
    module_id = node["module_id"]
    module_file = f"module_{module_id}.csv"
    model = trained_models[module_file]
    df = test_data_dict[module_file]
    module_row = df[df["query_id"] == query_id]

    if node["children"] is not None and len(node["children"]) > 0:
        child = node["children"][0]
        child_output = predict(child, query_id, treatment_id, trained_models, test_data_dict, hl_test_df, hl_covariates, use_high_level_features, num_modules, num_feature_dimensions)
        module_row = module_row.assign(child_output=child_output)
    
    if use_high_level_features:
        covariates = hl_covariates + [x for x in df.columns if x == "child_output"]
        module_row = module_row.merge(hl_test_df, on="query_id", how="left")
        # rename treatment_id_x to treatment_id
        module_row.rename(columns={"treatment_id_x": "treatment_id"}, inplace=True)
    else:
        covariates = [x for x in df.columns if "feature" in x or x == "child_output"]
    
    p1, p0 = predict_model(model, module_row, covariates, num_modules=num_modules, num_feature_dimensions=num_feature_dimensions, model_name="Baseline",return_po=True, return_effect=False)


    return p1 if treatment_id == 1 else p0

def predict_single_query(args):
    query_id, json_dict, trained_models, test_data_dict, hl_test_df, hl_covariates, use_high_level_features, num_modules, num_feature_dimensions = args
    tree_node = json_dict["json_tree"]
    p1 = predict(tree_node, query_id, 1, trained_models, test_data_dict, hl_test_df, hl_covariates, use_high_level_features, num_modules, num_feature_dimensions)
    p0 = predict(tree_node, query_id, 0, trained_models, test_data_dict, hl_test_df, hl_covariates, use_high_level_features, num_modules, num_feature_dimensions)
    return query_id, p1 - p0

def predict_modular_model_with_po(test_data, trained_models, jsons,  hl_test_df=None, hl_covariates=None, use_high_level_features=False, num_modules=3, num_feature_dimensions=3):
    predictions = {}

    # Convert test_data to a dictionary for faster access
    test_data_dict = {file: df for file, df in test_data.items()}

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for query_id, json_dict in jsons.items():
            futures.append(executor.submit(predict_single_query, (query_id, json_dict, trained_models, test_data_dict,  hl_test_df, hl_covariates, use_high_level_features, num_modules, num_feature_dimensions)))

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing queries"):
            query_id, prediction = future.result()
            predictions[query_id] = prediction

    return predictions

def get_ground_truth_effects_jsons(jsons_0, jsons_1, qids, scalers=None, scale=False):
    jsons_0 = {k:v for k,v in jsons_0.items() if k in qids}
    jsons_1 = {k:v for k,v in jsons_1.items() if k in qids}
    ground_truth_0 = {k:v["query_output"] for k,v in jsons_0.items()}
    ground_truth_1 = {k:v["query_output"] for k,v in jsons_1.items()}
    if scale:
        # get the last module id in each json and use the corresponding scaler 
        for k,v in jsons_0.items():
            module_id = str(v["json_tree"]["module_id"])
            output_scaler = scalers[module_id]["output_scaler"]
            output = v["query_output"]
            ground_truth_0[k] = output_scaler.transform(output.reshape(-1, 1)).squeeze()

        for k,v in jsons_1.items():
            module_id = str(v["json_tree"]["module_id"])
            output_scaler = scalers[module_id]["output_scaler"]
            output = v["query_output"]
            ground_truth_1[k] = output_scaler.transform(output.reshape(-1, 1)).squeeze()
 
    return {k: ground_truth_1[k] - ground_truth_0[k] for k in qids}

# Prepare results
def prepare_results(qids, predictions, ground_truth_data):
    results = []
    for qid in qids:
        results.append({
            "query_id": qid,
            "ground_truth_effect": ground_truth_data[qid],
            "estimated_effect": predictions[qid].squeeze()
        })
    return pd.DataFrame(results)


def get_sequential_model_effects(csv_path, obs_data_path, train_qids, test_qids, jsons_0, jsons_1, hidden_dim=32, epochs=100, batch_size=64, output_dim=1, underlying_model_class="MLP", scale=True, scaler_path=None, bias_strength=0, domain="synthetic_data", model_misspecification=False, composition_type="parallel", evaluate_train=False, train_df=None, test_df=None, covariates=None, use_high_level_features=False, data_dist="normal", num_modules=3, num_feature_dimensions=3):
    if scale:
        scalers = create_module_scalers(csv_path, obs_data_path, scaler_path, bias_strength, composition_type, data_dist=data_dist)
    else:
        scalers = None
    # split the data into train and test
    train_data, test_data, module_files = split_modular_data(csv_path, obs_data_path, train_qids, test_qids, scalers, scale, bias_strength, composition_type)
    # Train models with access to fine-grained potential outcomes
    trained_models = train_modular_model_with_po(train_data, module_files, domain, model_misspecification, underlying_model_class, hidden_dim, epochs, batch_size, output_dim, train_df,  covariates, use_high_level_features)

    # save the trained models as .pt files
    # for module_file, model in trained_models.items():
    #     torch.save(model.state_dict(), f"{scaler_path}/trained_models_hl_{use_high_level_features}/{module_file}.pt")
    # Compute predictions
    train_jsons = {k:v for k,v in jsons_0.items() if k in train_qids}
    test_jsons = {k:v for k,v in jsons_0.items() if k in test_qids}
    if evaluate_train:
        train_predictions = predict_modular_model_with_po(train_data, trained_models, train_jsons, train_df, covariates, use_high_level_features, num_modules, num_feature_dimensions)
        # Get ground truth effects        
        train_ground_truth = get_ground_truth_effects_jsons(jsons_0, jsons_1, train_qids, scalers, scale)
        train_results = prepare_results(train_qids, train_predictions, train_ground_truth)
    test_predictions = predict_modular_model_with_po(test_data, trained_models, test_jsons, test_df, covariates, use_high_level_features, num_modules, num_feature_dimensions)
    test_ground_truth = get_ground_truth_effects_jsons(jsons_0, jsons_1, test_qids, scalers, scale)
    test_results = prepare_results(test_qids, test_predictions, test_ground_truth)
    if evaluate_train:
        return train_results, test_results
    else:
        return test_results


   