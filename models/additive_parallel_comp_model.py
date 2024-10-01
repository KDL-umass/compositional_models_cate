import os
import json
import pickle
import pandas as pd
import numpy as np
from models.MoE import *
from models.utils import  get_ground_truth_effects


import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
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

def create_module_scalers(csv_path, obs_data_path, scaler_path, train_qids, bias_strength):
    # Load module data and treatment assignments
    module_data = load_module_data(csv_path)
    query_id_treatment_id = load_treatment_assignments(obs_data_path, bias_strength)

    # Create scaler directory
    # scaler_path = f"{obs_data_path}/feature_sum_{bias_strength}/scalers"
    os.makedirs(scaler_path, exist_ok=True)

    for module_file, module_df in module_data.items():
        module_id = module_file.split(".")[0].split("_")[-1]
        
        # Replace inf with nan and then fill nan with 0
        module_df = module_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # # Filter data based on train_qids and treatment assignments
        # module_df = module_df[module_df["query_id"].isin(train_qids)]
        # module_df["assigned_treatment_id"] = module_df["query_id"].apply(lambda x: query_id_treatment_id[str(x)])
        # module_df = module_df[module_df["treatment_id"] == module_df["assigned_treatment_id"]]
        
        # Identify feature columns
        feature_names = [x for x in module_df.columns if "feature" in x]
        
        # Sort by feature names
        module_df = module_df.sort_values(by=feature_names)
        
        # Create and fit scalers
        input_scaler = StandardScaler(with_mean=False)
        output_scaler = StandardScaler(with_mean=False)
        
        input_scaler.fit(module_df[feature_names].values)
        output_scaler.fit(module_df[["output"]].values)
        
        # Save scalers
        with open(f"{scaler_path}/input_scaler_{module_id}.pkl", "wb") as f:
            pickle.dump(input_scaler, f)
        
        with open(f"{scaler_path}/output_scaler_{module_id}.pkl", "wb") as f:
            pickle.dump(output_scaler, f)
        
        print(f"Created scalers for module {module_id}")

    print(f"All scalers created and saved in {scaler_path}")


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




def get_additive_model_effects(csv_path, obs_data_path, train_qids, test_qids, hidden_dim=32, epochs=100, batch_size=64, output_dim=1, underlying_model_class="MLP", scale = True, scaler_path=None, bias_strength=0):
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
        covariates = [x for x in train_df.columns if "feature" in x]
        treatment = "treatment_id"
        outcome = "output"
        input_dim = len(covariates)
        if underlying_model_class == "MLP":
            expert_model = BaselineModel(input_dim + 1, hidden_dim, output_dim)
        else:
            expert_model = BaselineLinearModel(input_dim + 1, output_dim)
        
        expert_model, train_losses, val_losses = train_model(expert_model, train_df, covariates, treatment, outcome, epochs, batch_size)
        causal_effect_estimates_train = predict_model(expert_model, train_df, covariates)
        causal_effect_estimates_test = predict_model(expert_model, test_df, covariates)
        train_df["estimated_effect"] = causal_effect_estimates_train
        test_df["estimated_effect"] = causal_effect_estimates_test
        train_data[module_file] = train_df
        test_data[module_file] = test_df

    # now for each module, get the ground truth and estimated effects
    additive_ground_truth_effects_train = {}
    additive_estimated_effects_train = {}
    additive_ground_truth_effects_test = {}
    additive_estimated_effects_test = {}
    modules_csvs_train = {}
    modules_csvs_test = {}
    for module_file in module_files:
        module_name = module_file.split(".")[0]
        train_df = train_data[module_file]
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

    print(additive_combined_train_df.head())
    print(additive_combined_test_df.head())

    return additive_combined_train_df, additive_combined_test_df, modules_csvs_train, modules_csvs_test